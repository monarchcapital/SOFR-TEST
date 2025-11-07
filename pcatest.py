import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SOFR Futures PCA Analyzer")

# --- Helper Functions for Data Processing ---

def load_data(uploaded_file, file_type):
    """
    Loads CSV data into a DataFrame.
    file_type must be 'price' or 'expiry'.
    """
    if uploaded_file is None:
        return None
        
    try:
        if file_type == 'expiry':
            df = pd.read_csv(uploaded_file, sep=',')
            # Standardize column names based on common expiry file headers
            df = df.rename(columns={
                'MATURITY': 'Contract', 
                'DATE': 'ExpiryDate', 
                'Maturity': 'Contract', 
                'Date': 'ExpiryDate'
            })
            
            # Clean up the DataFrame
            if 'Contract' in df.columns and 'ExpiryDate' in df.columns:
                df = df.set_index('Contract')
                df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
                df.index.name = 'Contract'
                return df
            else:
                st.error("Expiry file must contain 'MATURITY'/'Contract' and 'DATE'/'ExpiryDate' columns.")
                return None

        elif file_type == 'price':
            # Price File (sofr rates.csv format: Date as index)
            df = pd.read_csv(
                uploaded_file, 
                index_col=0, 
                parse_dates=True,
                # Try common date formats
                date_format='mixed' 
            )
            
            # Ensure index is a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                 st.error("Price file index (first column) must be a recognizable date format.")
                 return None
                 
            # Drop any columns that are not numeric
            df = df.select_dtypes(include=np.number)
            return df

    except Exception as e:
        st.error(f"Error loading or parsing {file_type} file: {e}")
        return None

def get_analysis_contracts(price_data, expiry_data, analysis_date):
    """Filters price data to only include contracts that were trading on or after a given date."""
    
    # Find the first contract that has a non-NaN value on the analysis_date
    try:
        current_prices = price_data.loc[analysis_date].dropna()
        if current_prices.empty:
            st.warning(f"No contract prices found on {analysis_date.strftime('%Y-%m-%d')}. Try a different date.")
            return None, None
            
        # Get the first contract code
        first_contract = current_prices.index[0]
        
        # Filter the full historical price data to keep only contracts >= first_contract
        # Use a consistent contract order function
        all_contracts = contracts_in_order(price_data.columns)
        
        # Ensure only contracts that appear in the expiry data are considered
        valid_contracts = [c for c in all_contracts if c in expiry_data.index]

        price_data_filtered = price_data.loc[:, contracts_in_order(valid_contracts, start_contract=first_contract)].dropna(axis=0, how='all')
        
        # Filter expiry data to match the filtered price columns
        relevant_contracts = price_data_filtered.columns.tolist()
        expiry_data_filtered = expiry_data.loc[expiry_data.index.isin(relevant_contracts)].copy()
        
        return price_data_filtered, expiry_data_filtered
        
    except KeyError:
        st.error(f"The analysis date {analysis_date.strftime('%Y-%m-%d')} is outside the historical data range.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during contract filtering: {e}")
        return None, None

def contracts_in_order(all_contracts, start_contract=None):
    """Returns a list of contracts sorted by year/month code."""
    
    # Filter out non-string columns just in case
    str_contracts = [c for c in all_contracts if isinstance(c, str)]

    # Simple sorting based on contract code (e.g., H21 < M21 < U21 < Z21 < H22)
    # This works because the year comes second (H21 is March 2021).
    # The primary key is the year (last two digits), secondary key is the month letter.
    sorted_contracts = sorted(str_contracts, key=lambda x: (x[-2:], x[0]))
    
    if start_contract:
        try:
            start_index = sorted_contracts.index(start_contract)
            return sorted_contracts[start_index:]
        except ValueError:
            # start_contract not found
            return sorted_contracts
            
    return sorted_contracts
    
def transform_to_analysis_curve(price_data, expiry_data, analysis_date):
    """
    Transforms the historical price data into a curve where columns represent 
    contracts relative to the analysis date, and returns the nearest contract price 
    on the analysis date.
    """
    if price_data is None or expiry_data is None:
        return None, None

    # Get the nearest contract that was trading on or after the analysis date
    analysis_contracts = contracts_in_order(price_data.columns)
    
    # Re-index the price data columns to ensure proper order
    price_data = price_data[analysis_contracts]

    # Use the first contract's price on the analysis date as the anchor for reconstruction later
    nearest_contract_code = analysis_contracts[0]
    try:
        # Find the row for the analysis date, then select the price of the nearest contract
        nearest_contract_price = price_data.loc[analysis_date, nearest_contract_code]
    except KeyError:
        st.error(f"Could not find price for contract {nearest_contract_code} on {analysis_date.strftime('%Y-%m-%d')}.")
        return None, None

    return price_data, nearest_contract_price

# --- Core Derivative Calculation Functions (MODIFIED to accept gap_contracts) ---

def calculate_outright_spreads(prices_df, gap_contracts=1):
    """
    Calculates calendar spreads (C_N - C_{N + gap_contracts}).
    gap_contracts=1 is 3-month (standard IMM), 2 is 6-month, 4 is 12-month.
    """
    contracts = prices_df.columns
    spreads_df = pd.DataFrame(index=prices_df.index)

    for i in range(len(contracts) - gap_contracts):
        c1 = contracts[i]
        c2 = contracts[i + gap_contracts]
        
        # Determine the tenor for labeling
        tenor_months = gap_contracts * 3
        
        # Use a specific label to differentiate 6M and 12M legs from standard 3M
        if tenor_months == 3:
            name = f"{c1}-{c2}"
        elif tenor_months == 6:
            name = f"{c1}-{c2} (6M)"
        elif tenor_months == 12:
            name = f"{c1}-{c2} (12M)"
        else:
            name = f"{c1}-{c2} ({tenor_months}M)"
            
        spreads_df[name] = prices_df[c1] - prices_df[c2]

    return spreads_df

def calculate_butterflies(prices_df, gap_contracts=1):
    """
    Calculates butterfly spreads (C_N - 2*C_{N + gap_contracts} + C_{N + 2*gap_contracts}).
    gap_contracts=1 is a standard 3-month fly, 2 is a 6-month fly, 4 is a 12-month fly.
    """
    contracts = prices_df.columns
    butterflies_df = pd.DataFrame(index=prices_df.index)

    required_contracts = 2 * gap_contracts
    
    for i in range(len(contracts) - required_contracts):
        c1 = contracts[i]
        c2 = contracts[i + gap_contracts]
        c3 = contracts[i + required_contracts]

        # Determine the leg tenor for labeling
        tenor_months = gap_contracts * 3
        
        # Use a specific label to differentiate 6M and 12M legs from standard 3M
        if tenor_months == 3:
            name = f"{c1}-{c2}x2-{c3}"
        elif tenor_months == 6:
            name = f"{c1}-{c2}x2-{c3} (6M Leg)"
        elif tenor_months == 12:
            name = f"{c1}-{c2}x2-{c3} (12M Leg)"
        else:
            name = f"{c1}-{c2}x2-{c3} ({tenor_months}M Leg)"
        
        butterflies_df[name] = prices_df[c1] - 2 * prices_df[c2] + prices_df[c3]
        
    return butterflies_df

# --- PCA and Reconstruction Functions ---

def run_pca_and_get_loadings(data_df, n_components):
    """Runs PCA, returns fitted PCA object, scaled data, and loadings."""
    # Standardize the data
    data_mean = data_df.mean()
    data_std = data_df.std()
    
    # Handle zero std dev to prevent division by zero
    data_std[data_std == 0] = 1.0 
    data_scaled = (data_df - data_mean) / data_std
    
    # Run PCA
    # Use only rows that contain no NaNs for fitting the model
    data_fit = data_scaled.dropna()
    
    if data_fit.shape[0] < n_components:
        raise ValueError(f"Not enough clean data points ({data_fit.shape[0]}) to fit {n_components} components.")

    pca = PCA(n_components=n_components)
    pca.fit(data_fit)
    
    # Calculate Loadings (Eigenvectors)
    loadings = pd.DataFrame(
        pca.components_.T, 
        index=data_df.columns, 
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return pca, data_scaled, data_mean, data_std, loadings

def calculate_outright_loadings(derivatives_loadings, outright_contracts, derivatives_df):
    """
    Transforms spread loadings into outright price loadings using cumulative sum,
    anchored to the nearest contract price.
    """
    
    # 1. Initialize outright loadings
    outright_loadings = pd.DataFrame(0.0, index=outright_contracts, columns=derivatives_loadings.columns)
    
    # 2. Filter for 3M spreads. 
    # Use the spread calculation function to get the expected names of 3M spreads (without the (3M) suffix)
    temp_3m_spreads = calculate_outright_spreads(derivatives_df.iloc[:, :len(outright_contracts)], gap_contracts=1)
    spread_columns_3m = temp_3m_spreads.columns.tolist()
    
    # Filter the actual loadings using these names
    spread_loadings_3m = derivatives_loadings.loc[derivatives_loadings.index.intersection(spread_columns_3m)]
        
    for pc_name in derivatives_loadings.columns:
        # Calculate the cumulative sum of the negative 3M spread loadings
        cumulative_loadings = -spread_loadings_3m[pc_name].cumsum()
        
        # Apply the cumulative sum to the outright loadings, starting from the second contract
        outright_loadings.loc[outright_contracts[1:1+len(cumulative_loadings)], pc_name] = cumulative_loadings.values

    return outright_loadings

def transform_and_reconstruct(data_df, pca_model, data_mean, data_std, num_pcs):
    """Scales data, calculates scores, sets non-selected scores to zero, and reconstructs data."""
    
    # 1. Handle zero std dev to prevent division by zero
    data_std_safe = data_std.copy()
    data_std_safe[data_std_safe == 0] = 1.0
    
    # 1. Scale data
    data_scaled = (data_df - data_mean) / data_std_safe
    data_clean = data_scaled.dropna(how='all')
    
    # 2. Calculate PC Scores
    n_fitted_components = pca_model.components_.shape[0]
    
    # --- FIX: Defensive check for empty data_clean before transformation ---
    if data_clean.empty:
        # If no clean data, return empty structures with correct columns/index
        st.warning("No complete rows of derivative data found for PCA scoring. Returning empty score/reconstruction DataFrames.")
        empty_index = data_df.index
        empty_reconstructed = pd.DataFrame(np.nan, index=empty_index, columns=data_df.columns)
        empty_scores = pd.DataFrame(np.nan, index=empty_index, columns=[f'PC{i+1}' for i in range(n_fitted_components)])
        return empty_reconstructed, empty_scores
    # ----------------------------------------------------------------------
    
    scores = pd.DataFrame(
        pca_model.transform(data_clean),
        index=data_clean.index,
        columns=[f'PC{i+1}' for i in range(n_fitted_components)]
    )
    
    # 3. Apply PCA selection: set non-selected factor scores to zero
    selected_scores = scores.copy()
    if num_pcs < n_fitted_components:
        cols_to_zero = [f'PC{i+1}' for i in range(num_pcs, n_fitted_components)]
        selected_scores[cols_to_zero] = 0.0
    
    # 4. Reconstruct: Scores * Loadings.T + Mean (rescale)
    # Ensure components array matches the columns in selected_scores
    components_to_use = pca_model.components_[:selected_scores.shape[1], :]
    
    reconstructed_scaled = pd.DataFrame(
        selected_scores @ components_to_use, 
        index=selected_scores.index, 
        columns=data_df.columns
    )
    
    # Rescale back to original values: (Reconstructed_Scaled * Std) + Mean
    # Need to reindex and fill NaNs because reconstruction is only for dates in data_clean
    reconstructed_df = pd.DataFrame(np.nan, index=data_df.index, columns=data_df.columns)
    reconstructed_df.loc[reconstructed_scaled.index, reconstructed_scaled.columns] = reconstructed_scaled
    reconstructed_df = (reconstructed_df * data_std_safe) + data_mean
    
    # Pad the scores DataFrame to match the original index for consistency
    full_scores = pd.DataFrame(np.nan, index=data_df.index, columns=scores.columns)
    full_scores.loc[scores.index, scores.columns] = scores

    return reconstructed_df, full_scores

def reconstruct_prices_and_derivatives(reconstructed_derivatives, outright_contracts, nearest_contract_price):
    """
    Uses reconstructed derivative values (spreads/flies) to reconstruct the outright prices.
    """
    
    # 1. Filter out the reconstructed 3M spreads
    temp_3m_spreads = calculate_outright_spreads(reconstructed_derivatives.iloc[:, :len(outright_contracts)], gap_contracts=1)
    spread_cols_3m = temp_3m_spreads.columns.tolist()
    reconstructed_spreads_3m_full = reconstructed_derivatives[reconstructed_derivatives.columns.intersection(spread_cols_3m)]


    # Create the DataFrame for reconstructed outright prices
    reconstructed_prices = pd.DataFrame(index=reconstructed_derivatives.index, columns=outright_contracts)
    
    # Anchor the first contract price (C1) to the actual price (nearest_contract_price)
    nearest_contract_code = outright_contracts[0]
    # nearest_contract_price is a float, we need to broadcast it across all dates
    reconstructed_prices[nearest_contract_code] = nearest_contract_price
    
    # Use the 3M spreads to iteratively build the rest of the curve
    for i in range(len(outright_contracts) - 1):
        c1 = outright_contracts[i]
        c2 = outright_contracts[i+1]
        
        try:
            spread_col_name = f"{c1}-{c2}"
            
            # C2 = C1 - (C1 - C2) = C1 - Spread
            reconstructed_prices[c2] = reconstructed_prices[c1] - reconstructed_spreads_3m_full[spread_col_name]
        except KeyError:
            # This happens when we run out of 3M spread data
            break

    # 2. Calculate the fair spreads/flies from the reconstructed prices
    reconstructed_prices = reconstructed_prices.dropna(axis=1, how='all')

    reconstructed_spreads_3m = calculate_outright_spreads(reconstructed_prices, gap_contracts=1)
    reconstructed_spreads_6m = calculate_outright_spreads(reconstructed_prices, gap_contracts=2)
    reconstructed_spreads_12m = calculate_outright_spreads(reconstructed_prices, gap_contracts=4)
    reconstructed_spreads = pd.concat([reconstructed_spreads_3m, reconstructed_spreads_6m, reconstructed_spreads_12m], axis=1)

    reconstructed_butterflies_3m = calculate_butterflies(reconstructed_prices, gap_contracts=1)
    reconstructed_butterflies_6m = calculate_butterflies(reconstructed_prices, gap_contracts=2)
    reconstructed_butterflies_12m = calculate_butterflies(reconstructed_prices, gap_contracts=4)
    reconstructed_butterflies = pd.concat([reconstructed_butterflies_3m, reconstructed_butterflies_6m, reconstructed_butterflies_12m], axis=1)
    
    return reconstructed_prices, reconstructed_spreads, reconstructed_butterflies

def calculate_z_scores(scores_df):
    """Calculates the Z-score for each PC score."""
    z_scores = (scores_df - scores_df.mean()) / scores_df.std()
    return z_scores

# --- Streamlit Application ---

st.title("SOFR Futures PCA Risk Analyzer")
st.markdown("---")

# --- 1. File Upload and Date Selection ---
st.header("1. Data Loading and Analysis Date Selection")

col1, col2 = st.columns(2)
with col1:
    uploaded_price_file = st.file_uploader("Upload SOFR Price Data (CSV)", type="csv", key="price")
with col2:
    uploaded_expiry_file = st.file_uploader("Upload Contract Expiry Data (CSV)", type="csv", key="expiry")

# Load data using specific types
price_data = load_data(uploaded_price_file, 'price')
expiry_data = load_data(uploaded_expiry_file, 'expiry')

if price_data is not None and expiry_data is not None:
    
    # --- Check index type before calling .date() ---
    if isinstance(price_data.index, pd.DatetimeIndex):
        max_date = price_data.index.max().date()
        
        analysis_date = st.date_input(
            "Select Analysis Date (latest available date is recommended for snapshot)",
            value=max_date,
            max_value=max_date
        )
        
        analysis_date = datetime.combine(analysis_date, datetime.min.time())
    else:
        st.error("Price data index is not recognized as a DatetimeIndex. Please ensure the first column of your price file is a date.")
        st.stop()
    
    # --- 2. Data Filtering and Preprocessing ---
    
    price_data_filtered, expiry_data_filtered = get_analysis_contracts(price_data, expiry_data, analysis_date)
    
    if price_data_filtered is not None and expiry_data_filtered is not None:
        
        # Filter and order the price data for analysis
        price_data, nearest_contract_price = transform_to_analysis_curve(
            price_data_filtered, expiry_data_filtered, analysis_date
        )
        
        if price_data is None:
            st.stop()

        # The contract list is now the ordered list of all contracts from the nearest forward
        outright_contracts = price_data.columns.tolist()

        st.success(f"Data ready. Nearest contract: **{outright_contracts[0]}** (Price: **{nearest_contract_price:.4f}**)")
        st.dataframe(price_data.tail(), use_container_width=True)


        # --- 3. PCA Setup: Calculate Spreads and Butterflies ---
        
        # Calculate 3M, 6M, and 12M Spreads
        spreads_3m_df = calculate_outright_spreads(price_data, gap_contracts=1)
        spreads_6m_df = calculate_outright_spreads(price_data, gap_contracts=2)
        spreads_12m_df = calculate_outright_spreads(price_data, gap_contracts=4)
        spreads_df = pd.concat([spreads_3m_df, spreads_6m_df, spreads_12m_df], axis=1)

        # Calculate 3M, 6M, and 12M Butterflies
        butterflies_3m_df = calculate_butterflies(price_data, gap_contracts=1)
        butterflies_6m_df = calculate_butterflies(price_data, gap_contracts=2)
        butterflies_12m_df = calculate_butterflies(price_data, gap_contracts=4)
        butterflies_df = pd.concat([butterflies_3m_df, butterflies_6m_df, butterflies_12m_df], axis=1)
        
        # Combine all derivatives for PCA
        derivatives_df = pd.concat([spreads_df, butterflies_df], axis=1).dropna(how='all')
        
        
        if derivatives_df.shape[1] > 0 and derivatives_df.shape[0] > 100: # Check for minimum data
            
            # --- 3. PCA Execution and Factor Selection ---
            st.header("3. PCA Execution and Factor Selection")
            
            max_pcs = min(20, derivatives_df.shape[1])
            n_components = st.slider("Select Number of Principal Components to Keep", 
                                     min_value=1, 
                                     max_value=max_pcs, 
                                     value=min(3, max_pcs), 
                                     step=1)
            
            # Run PCA
            try:
                pca, data_scaled, data_mean, data_std, loadings = run_pca_and_get_loadings(
                    derivatives_df, n_components=max_pcs
                )
            except ValueError as e:
                st.error(f"PCA Error: {e}. This often means the input data has too few non-NaN rows or non-zero columns for the requested number of components.")
                st.stop()
                
            
            # Explained Variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            st.markdown("###### Explained Variance Ratio")
            var_df = pd.DataFrame({
                'PC': [f'PC{i+1}' for i in range(max_pcs)],
                'Variance (%)': (explained_variance_ratio * 100).round(2),
                'Cumulative (%)': (cumulative_variance * 100).round(2)
            }).set_index('PC')
            st.dataframe(var_df.T, use_container_width=True)
            
            
            # --- 4. Factor Interpretation: Loadings and Shapes ---
            st.header("4. Factor Interpretation: Loadings and Shapes")
            
            # Filter loadings to only the selected number of components
            selected_loadings = loadings.iloc[:, :n_components]
            
            # --- 4.1. Derivative Loadings Heatmap ---
            st.markdown("##### 4.1. Derivative Loadings Heatmap")
            fig_loadings, ax_loadings = plt.subplots(figsize=(12, 0.4 * selected_loadings.shape[0]))
            sns.heatmap(selected_loadings, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, linecolor='black', cbar_kws={'label': 'Loadings (Weights)'}, ax=ax_loadings)
            ax_loadings.set_title(f"PCA Factor Loadings on Derivatives (3M, 6M, 12M Spreads/Flies)")
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_loadings)
            
            # --- 4.2. Outright Price Loadings (Factor Shapes) ---
            st.markdown("##### 4.2. Outright Price Loadings (Factor Shapes)")
            outright_loadings = calculate_outright_loadings(selected_loadings, outright_contracts, derivatives_df)
            
            fig_shape, ax_shape = plt.subplots(figsize=(12, 6))
            outright_loadings.plot(kind='line', ax=ax_shape, marker='o')
            ax_shape.set_title("PCA Factor Impact on Outright Price Curve (Factor Shapes)")
            ax_shape.set_xlabel("Contract")
            ax_shape.set_ylabel("Loadings (Change in Price for 1 Std Dev PC Move)")
            ax_shape.grid(True, linestyle='--')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_shape)

            
            # --- 5. Factor Scores and Z-Scores ---
            st.header("5. Factor Scores Time Series and Z-Scores")
            
            # Calculate Scores time series (using max_pcs for historical context)
            _, scores = transform_and_reconstruct(
                derivatives_df, pca, data_mean, data_std, num_pcs=max_pcs
            )
            
            if scores.empty or scores.isnull().all().all():
                 st.error("Cannot generate Factor Scores: No complete historical data rows found.")
                 st.stop()
                 
            selected_scores = scores.iloc[:, :n_components]
            z_scores = calculate_z_scores(selected_scores)
            
            # Plot PC Scores Time Series
            for i in range(n_components):
                pc_name = f'PC{i+1}'
                fig_score, ax_score = plt.subplots(figsize=(12, 4))
                selected_scores[pc_name].plot(ax=ax_score, title=f"{pc_name} Score Time Series")
                ax_score.grid(True, linestyle='--')
                ax_score.set_ylabel("Score Value (in Std Devs)")
                plt.tight_layout()
                st.pyplot(fig_score)
                
            st.markdown("###### Factor Z-Scores (Current Deviation)")
            st.dataframe(z_scores.iloc[-1:].T.rename(columns={z_scores.index[-1]: 'Z-Score'}), use_container_width=True)
            
            # --- Pre-calculate Fair Values for Snapshots (only using selected PCs) ---
            reconstructed_derivatives_selected, _ = transform_and_reconstruct(
                derivatives_df, pca, data_mean, data_std, num_pcs=n_components
            )
            
            # Check for empty reconstruction before proceeding to snapshot analysis
            if reconstructed_derivatives_selected.empty or reconstructed_derivatives_selected.isnull().all().all():
                st.error("Cannot perform snapshot analysis: Reconstructed derivative data is empty or all NaN. Check data quality and time range.")
                st.stop()
            
            # Get the fair value curve (prices, spreads, flies)
            (
                pca_fair_prices, 
                pca_fair_spreads, 
                pca_fair_butterflies
            ) = reconstruct_prices_and_derivatives(
                reconstructed_derivatives_selected, 
                outright_contracts, 
                nearest_contract_price
            )


            # --- 6. Outright Price Mispricing Snapshot ---
            st.markdown("---")
            st.header("6. Outright Price Mispricing Snapshot")
            
            try:
                # Get prices for the analysis date
                original_prices_snap = price_data.loc[analysis_date].dropna()
                pca_fair_prices_snap = pca_fair_prices.loc[analysis_date].dropna()
                
                # Align and calculate mispricing
                outright_comparison = pd.DataFrame({
                    'Original': original_prices_snap,
                    'PCA Fair': pca_fair_prices_snap
                }).dropna()
                
                mispricing = outright_comparison['Original'] - outright_comparison['PCA Fair']
                
                # Plotting
                fig_outright, ax_outright = plt.subplots(figsize=(12, 6))
                outright_comparison.plot(ax=ax_outright, marker='o')
                mispricing.plot(ax=ax_outright.twinx(), kind='bar', color='grey', alpha=0.3, label='Mispricing (Original - PCA Fair)')
                ax_outright.set_title(f"Outright Curve Snapshot on {analysis_date.strftime('%Y-%m-%d')}")
                ax_outright.set_ylabel("Price")
                ax_outright.set_xlabel("Contract")
                ax_outright.grid(True, linestyle='--')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_outright)

                # Table
                st.markdown("###### Outright Mispricing")
                detailed_comparison_outright = outright_comparison.copy()
                detailed_comparison_outright.index.name = 'Outright Contract'
                detailed_comparison_outright['Mispricing (BPS)'] = mispricing * 10000
                st.dataframe(
                    detailed_comparison_outright.style.format({
                        'Original': "{:.4f}",
                        'PCA Fair': "{:.4f}",
                        'Mispricing (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )
                
            except KeyError:
                st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data. Please choose a different date within the historical range.")
            
            
            # --- 7. Calendar Spread Mispricing Snapshot ---
            st.markdown("---")
            st.header("7. Calendar Spread Mispricing Snapshot (3M, 6M, 12M)")
            
            if spreads_df.shape[1] > 0:
                try:
                    original_spreads_snap = spreads_df.loc[analysis_date].dropna()
                    pca_fair_spreads_snap = pca_fair_spreads.loc[analysis_date].dropna()
                    
                    spread_comparison = pd.DataFrame({
                        'Original': original_spreads_snap,
                        'PCA Fair': pca_fair_spreads_snap
                    }).dropna()

                    mispricing = spread_comparison['Original'] - spread_comparison['PCA Fair']
                    
                    # Plotting
                    fig_spread, ax_spread = plt.subplots(figsize=(12, 6))
                    spread_comparison.plot(ax=ax_spread, marker='o')
                    mispricing.plot(ax=ax_spread.twinx(), kind='bar', color='grey', alpha=0.3, label='Mispricing (Original - PCA Fair)')
                    ax_spread.set_title(f"Spread Snapshot on {analysis_date.strftime('%Y-%m-%d')} (3M, 6M, 12M)")
                    ax_spread.set_ylabel("Spread Value")
                    ax_spread.set_xlabel("Calendar Spread Contract")
                    ax_spread.grid(True, linestyle='--')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig_spread)

                    # --- Detailed Spread Table ---
                    st.markdown("###### Calendar Spread Mispricing (3M, 6M, 12M)")
                    detailed_comparison_spread = spread_comparison.copy()
                    detailed_comparison_spread.index.name = 'Calendar Spread'
                    detailed_comparison_spread['Mispricing (BPS)'] = mispricing * 10000
                    detailed_comparison_spread = detailed_comparison_spread.rename(
                        columns={'Original': 'Original Spread', 'PCA Fair': 'PCA Fair Spread'}
                    )
                    st.dataframe(
                        detailed_comparison_spread.style.format({
                            'Original Spread': "{:.4f}",
                            'PCA Fair Spread': "{:.4f}",
                            'Mispricing (BPS)': "{:.2f}"
                        }),
                        use_container_width=True
                    )

                except KeyError:
                    st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for Spreads. Please choose a different date within the historical range.")
            else:
                st.info("Not enough contracts (need 2 or more) to calculate and plot spread snapshot.")
                

            # --- 8. Butterfly Mispricing Snapshot ---
            st.markdown("---")
            st.header("8. Butterfly Mispricing Snapshot (3M, 6M, 12M Leg)")
            
            if butterflies_df.shape[1] > 0:
                try:
                    original_fly_snap = butterflies_df.loc[analysis_date].dropna()
                    pca_fair_fly_snap = pca_fair_butterflies.loc[analysis_date].dropna()
                    
                    fly_comparison = pd.DataFrame({
                        'Original': original_fly_snap,
                        'PCA Fair': pca_fair_fly_snap
                    }).dropna()
                    
                    mispricing = fly_comparison['Original'] - fly_comparison['PCA Fair']
                    
                    # Plotting
                    fig_fly, ax_fly = plt.subplots(figsize=(12, 6))
                    fly_comparison.plot(ax=ax_fly, marker='o')
                    mispricing.plot(ax=ax_fly.twinx(), kind='bar', color='grey', alpha=0.3, label='Mispricing (Original - PCA Fair)')
                    ax_fly.set_title(f"Butterfly Snapshot on {analysis_date.strftime('%Y-%m-%d')} (3M, 6M, 12M Leg)")
                    ax_fly.set_ylabel("Butterfly Value")
                    ax_fly.set_xlabel("Butterfly Contract")
                    ax_fly.grid(True, linestyle='--')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig_fly)

                    # --- Detailed Fly Table ---
                    st.markdown("###### Butterfly (Fly) Mispricing (3M, 6M, 12M)")
                    detailed_comparison_fly = fly_comparison.copy()
                    detailed_comparison_fly.index.name = 'Butterfly Contract'
                    detailed_comparison_fly['Mispricing (BPS)'] = mispricing * 10000
                    detailed_comparison_fly = detailed_comparison_fly.rename(
                        columns={'Original': 'Original Fly', 'PCA Fair': 'PCA Fair Fly'}
                    )
                    
                    st.dataframe(
                        detailed_comparison_fly.style.format({
                            'Original Fly': "{:.4f}",
                            'PCA Fair Fly': "{:.4f}",
                            'Mispricing (BPS)': "{:.2f}"
                        }),
                        use_container_width=True
                    )

                except KeyError:
                    st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for Butterflies. Please choose a different date within the historical range.")
            else:
                st.info("Not enough contracts (need 3 or more) to calculate and plot butterfly snapshot.")
                
        else:
            st.error("PCA failed. Please check your data quantity and quality. Need at least 100 non-NaN rows to run PCA effectively.")
