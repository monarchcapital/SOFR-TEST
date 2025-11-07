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

def load_data(uploaded_file):
    """Loads CSV data into a DataFrame, adapting to price or expiry file formats."""
    if uploaded_file is None:
        return None
        
    try:
        # Read the uploaded file content to inspect the header for format identification
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue().decode("utf-8")
        uploaded_file.seek(0)
            
        # --- Case 1: Expiry File (EXPIRY (2).csv format: MATURITY, DATE) ---
        if 'MATURITY,DATE' in file_content.split('\n')[0].upper():
            df = pd.read_csv(uploaded_file, sep=',')
            df = df.rename(columns={'MATURITY': 'Contract', 'DATE': 'ExpiryDate'})
            # Ensure Contract is the index and Date is datetime
            df = df.set_index('Contract')
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
            df.index.name = 'Contract'
            return df

        # --- Case 2: Price File (sofr rates.csv format: Date as index) ---
        df = pd.read_csv(
            uploaded_file, 
            index_col=0, 
            parse_dates=True,
            date_format='%Y-%m-%d' # Assuming standard date format
        )
        # Drop any columns that are not numeric (e.g., if there are non-price columns)
        df = df.select_dtypes(include=np.number)
        return df

    except Exception as e:
        st.error(f"Error loading or parsing file: {e}")
        return None

def get_analysis_contracts(price_data, expiry_data, analysis_date):
    """Filters price data to only include contracts that were trading on or after a given date."""
    
    # 1. Determine the relevant IMM contracts (March, June, September, December)
    # The price_data columns are the contract codes (e.g., H21, M21, U21, Z21)
    
    # Find the first contract that has a non-NaN value on the analysis_date
    try:
        current_prices = price_data.loc[analysis_date].dropna()
        if current_prices.empty:
            st.warning(f"No contract prices found on {analysis_date.strftime('%Y-%m-%d')}. Try a different date.")
            return None, None
            
        first_contract = current_prices.index[0]
        
        # Filter the full historical price data to keep only contracts >= first_contract
        price_data_filtered = price_data.loc[:, contracts_in_order(price_data.columns, start_contract=first_contract)].dropna(axis=0, how='all')
        
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
    
    # Simple sorting based on contract code (e.g., H21 < M21 < U21 < Z21 < H22)
    # This works because the year comes second.
    sorted_contracts = sorted(all_contracts, key=lambda x: (x[-2:], x[0]))
    
    if start_contract:
        try:
            start_index = sorted_contracts.index(start_contract)
            return sorted_contracts[start_index:]
        except ValueError:
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
    gap_contracts=1 is 3-month, 2 is 6-month, 4 is 12-month.
    """
    contracts = prices_df.columns
    spreads_df = pd.DataFrame(index=prices_df.index)

    for i in range(len(contracts) - gap_contracts):
        c1 = contracts[i]
        c2 = contracts[i + gap_contracts]
        
        # Determine the tenor for labeling
        tenor_months = gap_contracts * 3
        name_suffix = f" ({tenor_months}M)" if tenor_months != 3 else "" # 3M spread name is already C1-C2
            
        spread_name = f"{c1}-{c2}{name_suffix}"
        spreads_df[spread_name] = prices_df[c1] - prices_df[c2]

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
        name_suffix = f" ({tenor_months}M Leg)" if tenor_months != 3 else ""
        
        butterfly_name = f"{c1}-{c2}x2-{c3}{name_suffix}"
        butterflies_df[butterfly_name] = prices_df[c1] - 2 * prices_df[c2] + prices_df[c3]
        
    return butterflies_df

# --- PCA and Reconstruction Functions ---

def run_pca_and_get_loadings(data_df, n_components):
    """Runs PCA, returns fitted PCA object, scaled data, and loadings."""
    # Standardize the data
    data_mean = data_df.mean()
    data_std = data_df.std()
    data_scaled = (data_df - data_mean) / data_std
    
    # Run PCA
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled.dropna())
    
    # Calculate Loadings (Eigenvectors)
    loadings = pd.DataFrame(
        pca.components_.T, 
        index=data_scaled.columns, 
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return pca, data_scaled, data_mean, data_std, loadings

def calculate_outright_loadings(derivatives_loadings, outright_contracts, nearest_contract_price=100.0):
    """
    Transforms spread/fly loadings into outright price loadings using cumulative sum,
    anchored to a reference price (e.g., nearest contract price).
    """
    # Create a DataFrame for outright loadings, starting with the nearest contract
    # Initialize all PC's impact to zero for the anchor contract
    outright_loadings = pd.DataFrame(0.0, index=outright_contracts, columns=derivatives_loadings.columns)
    
    # Find the nearest contract column index in the derivatives loadings
    nearest_contract_code = outright_contracts[0]
    
    # The spread between the first two contracts (C1 - C2) is the first row in the 3M spread section
    # We assume the spread columns are ordered C1-C2, C2-C3, C3-C4, etc.
    
    # Use the 3M Spread loadings (C_N - C_{N+1}) to calculate cumulative loadings
    
    # Filter for 3M spreads. Assuming the 3M spread names don't have "(3M)" in the modified code.
    spread_columns = [col for col in derivatives_loadings.index if col not in butterflies_df.columns and "M)" not in col]
    spread_loadings_3m = derivatives_loadings.loc[spread_columns]
    
    if len(spread_loadings_3m) < len(outright_contracts) - 1:
        st.warning("Not enough 3M spreads found to fully calculate outright loadings.")
        
    for pc_name in derivatives_loadings.columns:
        # The outright price of C2 is C1 - Spread(C1-C2)
        # Therefore, the outright loading of C2 is Load_C1 - Load_Spread(C1-C2)
        # Since we anchor C1 (the nearest contract) loading at 0.0, the loading for C2 is:
        # Load_C2 = Load_C1 - Load_Spread(C1-C2) = 0.0 - Load_Spread(C1-C2)
        
        # Calculate the cumulative sum of the negative 3M spread loadings
        # Load_Cn = Load_C(n-1) - Load_Spread(C(n-1)-Cn)
        # This is equivalent to: -CumulativeSum(Load_Spread)
        
        # The cumulative sum of the negative loadings of the 3M spreads
        cumulative_loadings = -spread_loadings_3m[pc_name].cumsum()
        
        # Apply the cumulative sum to the outright loadings, starting from the second contract
        # The index of outright_loadings starts at 0 (nearest_contract_code), the cumulative_loadings starts at C2's change
        outright_loadings.loc[outright_contracts[1:1+len(cumulative_loadings)], pc_name] = cumulative_loadings.values

    return outright_loadings

def transform_and_reconstruct(data_df, pca_model, data_mean, data_std, num_pcs):
    """Scales data, calculates scores, sets non-selected scores to zero, and reconstructs data."""
    
    # 1. Scale data
    data_scaled = (data_df - data_mean) / data_std
    data_clean = data_scaled.dropna(how='all')
    
    # 2. Calculate PC Scores
    scores = pd.DataFrame(
        pca_model.transform(data_clean),
        index=data_clean.index,
        columns=[f'PC{i+1}' for i in range(len(pca_model.components_))]
    )
    
    # 3. Apply PCA selection: set non-selected factor scores to zero
    selected_scores = scores.copy()
    if num_pcs < len(pca_model.components_):
        cols_to_zero = [f'PC{i+1}' for i in range(num_pcs, len(pca_model.components_))]
        selected_scores[cols_to_zero] = 0.0
    
    # 4. Reconstruct: Scores * Loadings.T + Mean (rescale)
    # Reconstructed scaled data = Scores @ Loadings
    reconstructed_scaled = pd.DataFrame(
        selected_scores @ pca_model.components_[:len(selected_scores.columns)], 
        index=selected_scores.index, 
        columns=data_df.columns
    )
    
    # Rescale back to original values: (Reconstructed_Scaled * Std) + Mean
    reconstructed_df = (reconstructed_scaled * data_std) + data_mean
    
    return reconstructed_df, scores

def reconstruct_prices_and_derivatives(reconstructed_derivatives, outright_contracts, nearest_contract_price):
    """
    Uses reconstructed derivative values (spreads/flies) to reconstruct the outright prices.
    """
    
    # 1. Filter out the reconstructed 3M spreads
    # Assuming 3M spread names don't have "M)" in the modified code (as determined in the helper)
    spread_cols = [col for col in reconstructed_derivatives.columns if col not in butterflies_df.columns and "M)" not in col]
    reconstructed_spreads_3m = reconstructed_derivatives[spread_cols]

    # Create the DataFrame for reconstructed outright prices
    reconstructed_prices = pd.DataFrame(index=reconstructed_derivatives.index, columns=outright_contracts)
    
    # Anchor the first contract price (C1) to the actual price (nearest_contract_price)
    nearest_contract_code = outright_contracts[0]
    reconstructed_prices[nearest_contract_code] = nearest_contract_price
    
    # Use the 3M spreads to iteratively build the rest of the curve
    for i in range(len(outright_contracts) - 1):
        c1 = outright_contracts[i]
        c2 = outright_contracts[i+1]
        
        try:
            spread_col_name = f"{c1}-{c2}"
            
            # C2 = C1 - (C1 - C2) = C1 - Spread
            reconstructed_prices[c2] = reconstructed_prices[c1] - reconstructed_spreads_3m[spread_col_name]
        except KeyError:
            # This happens when we run out of 3M spread data
            break

    # 2. Calculate the fair spreads/flies from the reconstructed prices
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

if uploaded_price_file and uploaded_expiry_file:
    price_data = load_data(uploaded_price_file)
    expiry_data = load_data(uploaded_expiry_file)
    
    if price_data is not None and expiry_data is not None:
        
        # Determine the latest date in the price data for the default
        max_date = price_data.index.max().date()
        
        analysis_date = st.date_input(
            "Select Analysis Date (latest available date is recommended for snapshot)",
            value=max_date,
            max_value=max_date
        )
        
        analysis_date = datetime.combine(analysis_date, datetime.min.time())
        
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


            # --- 3. PCA Setup: Calculate Spreads and Butterflies (MODIFIED to include 6M, 12M) ---
            
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
                
                # --- 4. PCA Execution and Factor Selection ---
                st.header("2. PCA Execution and Factor Selection")
                
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
                
                
                # --- 5. Factor Interpretation: Loadings and Shapes ---
                st.header("3. Factor Interpretation: Loadings and Shapes")
                
                # Filter loadings to only the selected number of components
                selected_loadings = loadings.iloc[:, :n_components]
                
                # --- 5.1. Derivative Loadings Heatmap ---
                st.markdown("##### 3.1. Derivative Loadings Heatmap")
                fig_loadings, ax_loadings = plt.subplots(figsize=(12, 0.4 * selected_loadings.shape[0]))
                sns.heatmap(selected_loadings, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, linecolor='black', cbar_kws={'label': 'Loadings (Weights)'}, ax=ax_loadings)
                ax_loadings.set_title(f"PCA Factor Loadings on Derivatives (3M, 6M, 12M Spreads/Flies)")
                plt.xticks(rotation=0)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig_loadings)
                
                # --- 5.2. Outright Price Loadings (Factor Shapes) ---
                st.markdown("##### 3.2. Outright Price Loadings (Factor Shapes)")
                outright_loadings = calculate_outright_loadings(selected_loadings, outright_contracts, nearest_contract_price)
                
                fig_shape, ax_shape = plt.subplots(figsize=(12, 6))
                outright_loadings.plot(kind='line', ax=ax_shape, marker='o')
                ax_shape.set_title("PCA Factor Impact on Outright Price Curve (Factor Shapes)")
                ax_shape.set_xlabel("Contract")
                ax_shape.set_ylabel("Loadings (Change in Price for 1 Std Dev PC Move)")
                ax_shape.grid(True, linestyle='--')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_shape)

                
                # --- 6. Factor Scores and Z-Scores ---
                st.header("4. Factor Scores Time Series and Z-Scores")
                
                reconstructed_df, scores = transform_and_reconstruct(
                    derivatives_df, pca, data_mean, data_std, n_components=max_pcs # Use max for scores time series
                )
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


                # --- 7. Curve Snapshot Analysis (Mispricing) (UPDATED to use combined data) ---
                st.header("5. Curve Snapshot Analysis (Mispricing)")
                
                # Recalculate Reconstruction using ONLY the selected PCs
                reconstructed_derivatives_selected, _ = transform_and_reconstruct(
                    derivatives_df, pca, data_mean, data_std, num_pcs=n_components
                )
                
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

                
                # --- 7.1. Outright Price Mispricing ---
                st.markdown("### Outright Prices")
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
                
                
                # --- 7.2. Spread Mispricing ---
                st.markdown("---")
                st.markdown("### Calendar Spreads (3M, 6M, 12M)")
                
                # Check if there are any spreads calculated (should always be if > 1 contract)
                if spreads_df.shape[1] > 0:
                    try:
                        original_spreads_snap = spreads_df.loc[analysis_date].dropna()
                        pca_fair_spreads_snap = pca_fair_spreads.loc[analysis_date].dropna()
                        
                        # Align and calculate mispricing
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
                        st.markdown("###### Calendar Spread Mispricing (3M, 6M, 12M)") # Updated header
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
                    

                # --- 7.3. Butterfly Mispricing ---
                st.markdown("---")
                st.markdown("### Butterflies (3M, 6M, 12M)")
                
                # Check if there are any butterflies calculated (should always be if > 2 contracts)
                if butterflies_df.shape[1] > 0:
                    try:
                        original_fly_snap = butterflies_df.loc[analysis_date].dropna()
                        pca_fair_fly_snap = pca_fair_butterflies.loc[analysis_date].dropna()
                        
                        # Align and calculate mispricing
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
                        st.markdown("###### Butterfly (Fly) Mispricing (3M, 6M, 12M)") # Updated header
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
