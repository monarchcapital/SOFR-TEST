import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression # NEW IMPORT for generalized hedging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SOFR Futures PCA Analyzer")

# --- Helper Functions for Data Processing ---

# Use st.cache_data for performance as file loading is idempotent
@st.cache_data
def load_data(uploaded_file):
    """Loads CSV data into a DataFrame, adapting to price or expiry file formats."""
    if uploaded_file is None:
        return None
        
    try:
        # Read the uploaded file content to inspect the header for format identification
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue().decode("utf-8")
        uploaded_file.seek(0)
            
        # --- Case 1: Expiry File (MATURITY, DATE) ---
        if 'MATURITY,DATE' in file_content.split('\n')[0].upper():
            df = pd.read_csv(uploaded_file, sep=',')
            df = df.rename(columns={'MATURITY': 'Contract', 'DATE': 'ExpiryDate'})
            df = df.set_index('Contract')
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
            df.index.name = 'Contract'
            return df

        # --- Case 2: Price File (Date as index) ---
        df = pd.read_csv(
            uploaded_file, 
            index_col=0, 
            parse_dates=True,
            sep=',', 
            header=0 
        )
        
        df.index.name = 'Date'
        df = df.dropna(axis=1, how='all')
        
        for col in df.columns:
            # Ensure price columns are numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(how='all')
        df = df[df.index.notna()]

        if df.empty or df.shape[1] == 0:
             raise ValueError("DataFrame is empty after processing or has no data columns.")
             
        return df
        
    except Exception as e:
        st.error(f"Error loading and processing data from {uploaded_file.name}: {e}")
        return None


@st.cache_data
def get_analysis_contracts(expiry_df, analysis_date):
    """Filters contract codes that expire on or after the analysis date."""
    if expiry_df is None:
        return pd.DataFrame()
    future_expiries = expiry_df[expiry_df['ExpiryDate'] >= analysis_date].copy()
    future_expiries = future_expiries.sort_values(by='ExpiryDate')
    
    if future_expiries.empty:
        st.warning(f"No contracts found expiring on or after {analysis_date.strftime('%Y-%m-%d')}.")
    
    return future_expiries

@st.cache_data
def transform_to_analysis_curve(price_df, future_expiries_df):
    """Selects and orders historical prices for relevant contracts."""
    if price_df is None or future_expiries_df.empty:
        return pd.DataFrame(), []
    contract_order = future_expiries_df.index.tolist()
    valid_contracts = [c for c in contract_order if c in price_df.columns]
    if not valid_contracts:
        st.warning("No matching contract columns found in price data for the selected analysis date range.")
        return pd.DataFrame(), []
    analysis_curve_df = price_df[valid_contracts]
    return analysis_curve_df, valid_contracts


# --- GENERALIZED DERIVATIVE CALCULATION FUNCTIONS (k-step) ---

@st.cache_data
def calculate_k_step_spreads(analysis_curve_df, k):
    """
    Calculates spreads between contracts separated by 'k' steps (e.g., k=1 for 3M, k=2 for 6M, k=4 for 12M).
    CME Basis: C_i - C_{i+k}
    """
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    spreads_data = {}
    
    for i in range(num_contracts - k):
        short_maturity = analysis_curve_df.columns[i]
        long_maturity = analysis_curve_df.columns[i+k]
        
        spread_label = f"{short_maturity}-{long_maturity}"
        # Spread = C_i - C_{i+k}
        spreads_data[spread_label] = analysis_curve_df.iloc[:, i] - analysis_curve_df.iloc[:, i+k]
        
    return pd.DataFrame(spreads_data)

@st.cache_data
def calculate_k_step_butterflies(analysis_curve_df, k):
    """
    Calculates butterflies using contracts separated by 'k' steps (e.g., k=1 for 3M fly, k=2 for 6M fly, k=4 for 12M fly).
    Formula: C_i - 2 * C_{i+k} + C_{i+2k}
    Label Format: C_i-2xC_{i+k}+C_{i+2k}
    """
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < 2 * k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    flies_data = {}

    for i in range(num_contracts - 2 * k):
        short_maturity = analysis_curve_df.columns[i]      # C_i
        center_maturity = analysis_curve_df.columns[i+k]   # C_{i+k}
        long_maturity = analysis_curve_df.columns[i+2*k]   # C_{i+2k}

        # Fly = C_i - 2*C_{i+k} + C_{i+2k}
        fly_label = f"{short_maturity}-2x{center_maturity}+{long_maturity}"

        flies_data[fly_label] = analysis_curve_df.iloc[:, i] - 2 * analysis_curve_df.iloc[:, i+k] + analysis_curve_df.iloc[:, i+2*k]

    return pd.DataFrame(flies_data)
# --- END GENERALIZED DERIVATIVE CALCULATION FUNCTIONS ---


def perform_pca(data_df):
    """Performs PCA on the input DataFrame (expected to be spreads for Fair Curve)."""
    data_df_clean = data_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None, None, None, None

    # Standardize the data (PCA on Correlation Matrix - preferred for spread PCA)
    data_mean = data_df_clean.mean()
    data_std = data_df_clean.std()
    data_scaled = (data_df_clean - data_mean) / data_std
    
    n_components = min(data_scaled.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    
    # Loadings (Eigenvectors on Correlation Matrix)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    # Get Eigenvalues (Variance of the principal components)
    eigenvalues = pca.explained_variance_
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    scores = pd.DataFrame(
        pca.transform(data_scaled),
        index=data_df_clean.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings, explained_variance_ratio, eigenvalues, scores, data_df_clean

# --- PCA ON PRICES (FOR NON-UNIFORM PC1 VISUALIZATION) ---
def perform_pca_on_prices(price_df):
    """
    Performs PCA directly on Outright Price Levels using the COVARIANCE MATRIX 
    (unstandardized data), which results in a NON-UNIFORM PC1.
    """
    data_df_clean = price_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None
        
    # Center the data, but DO NOT scale/standardize it (PCA on Covariance Matrix)
    data_centered = data_df_clean - data_df_clean.mean() 
    
    n_components = min(data_centered.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_centered)
    
    # Loadings (Eigenvectors - the raw sensitivities)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    
    explained_variance = pca.explained_variance_ratio_
    
    return loadings, explained_variance

# --- RECONSTRUCTION LOGIC (No Change) ---

def _reconstruct_derivative(original_df, reconstructed_prices, derivative_type='spread'):
    """
    Helper to reconstruct a derivative from the reconstructed price curve.
    """
    if original_df.empty:
        return pd.DataFrame()

    # Align the original data index with the reconstructed prices index
    valid_indices = reconstructed_prices.index.intersection(original_df.index)
    original_df_aligned = original_df.loc[valid_indices]
    reconstructed_prices_aligned = reconstructed_prices.loc[valid_indices]
    
    reconstructed_data = {}
    
    for label in original_df_aligned.columns:
        
        try:
            if derivative_type == 'spread':
                # Spread: C_i - C_{i+k}. Label is C_i-C_{i+k} (e.g., Z25-M26)
                c1, c_long = label.split('-')
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - reconstructed_prices_aligned[c_long + ' (PCA)']
                )
            
            elif derivative_type == 'fly':
                # Fly: C_i - 2 * C_{i+k} + C_{i+2k}. Label format: C_i-2xC_{i+k}+C_{i+2k}
                # Example: Z25-2xM26+Z26
                
                # 1. Split on the first '-' to get ['C_i', '2xC_{i+k}+C_{i+2k}']
                parts = label.split('-', 1) 
                c1 = parts[0] # C_i
                
                # 2. Split the second part by '+' to get ['2xC_{i+k}', 'C_{i+2k}']
                sub_parts = parts[1].split('+')
                
                # 3. Extract C_{i+k} from '2xC_{i+k}'
                c2_label = sub_parts[0].split('x')[1] 
                
                # 4. Extract C_{i+2k}
                c3_label = sub_parts[1] 
                
                # Reconstruct the derivative
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - 
                    2 * reconstructed_prices_aligned[c2_label + ' (PCA)'] + 
                    reconstructed_prices_aligned[c3_label + ' (PCA)']
                )
            
        except Exception as e:
             # Skip if reconstruction fails due to malformed label or missing price
             # print(f"Skipping {derivative_type} reconstruction for {label}. Error: {e}")
             continue 
    
    reconstructed_df = pd.DataFrame(reconstructed_data, index=reconstructed_prices_aligned.index)
    
    original_rename = {col: col + ' (Original)' for col in original_df_aligned.columns}
    original_df_renamed = original_df_aligned.rename(columns=original_rename)
    
    return pd.merge(original_df_renamed, reconstructed_df, left_index=True, right_index=True)


def reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_3M_df, spreads_3M_df, spreads_6M_df, butterflies_3M_df, butterflies_6M_df, spreads_12M_df, butterflies_12M_df):
    """
    Reconstructs Outright Prices and all derivative types based on the 
    reconstructed 3M spreads (PCA result) and the original nearest contract price anchor.
    """
    # Filter the analysis_curve_df to match the index of the reconstructed 3M spreads
    analysis_curve_df_aligned = analysis_curve_df.loc[reconstructed_spreads_3M_df.index]
    
    # --- 1. Reconstruct Outright Prices (Anchor) ---
    nearest_contract_original = analysis_curve_df_aligned.iloc[:, 0]
    nearest_contract_label = analysis_curve_df_aligned.columns[0]
    
    reconstructed_prices_df = pd.DataFrame(index=analysis_curve_df_aligned.index)
    reconstructed_prices_df[nearest_contract_label + ' (PCA)'] = nearest_contract_original # Anchor
    
    # Reconstruct all subsequent contracts using the reconstructed 3M spreads (k=1)
    for i in range(1, len(analysis_curve_df_aligned.columns)):
        prev_maturity = analysis_curve_df_aligned.columns[i-1]
        current_maturity = analysis_curve_df_aligned.columns[i]
        spread_label = f"{prev_maturity}-{current_maturity}" # This is always the 3M spread label
        
        if spread_label in reconstructed_spreads_3M_df.columns:
            # P_i = P_i-1 (PCA) - S_i-1,i (PCA)
            reconstructed_prices_df[current_maturity + ' (PCA)'] = (
                reconstructed_prices_df[prev_maturity + ' (PCA)'] - reconstructed_spreads_3M_df[spread_label]
            )
        else:
            # Fallback if the 3M spread is missing for that contract roll
             reconstructed_prices_df[current_maturity + ' (PCA)'] = reconstructed_prices_df[prev_maturity + ' (PCA)']

    original_price_rename = {col: col + ' (Original)' for col in analysis_curve_df_aligned.columns}
    original_prices_df = analysis_curve_df_aligned.rename(columns=original_price_rename)
    historical_outrights = pd.merge(original_prices_df, reconstructed_prices_df, left_index=True, right_index=True)


    # --- 2. Reconstruct Derivatives from Reconstructed Prices ---
    
    historical_spreads_3M = _reconstruct_derivative(spreads_3M_df, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_3M = _reconstruct_derivative(butterflies_3M_df, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_6M = _reconstruct_derivative(spreads_6M_df, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_6M = _reconstruct_derivative(butterflies_6M_df, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_12M = _reconstruct_derivative(spreads_12M_df, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_12M = _reconstruct_derivative(butterflies_12M_df, reconstructed_prices_df, derivative_type='fly')
    
    return historical_outrights, historical_spreads_3M, historical_butterflies_3M, historical_spreads_6M, historical_butterflies_6M, historical_spreads_12M, historical_butterflies_12M


# --- NEW GENERALIZED HEDGING LOGIC (Replaces the previous hedging section) ---

def calculate_derivatives_covariance_generalized(all_derivatives_df, scores_df, eigenvalues, pc_count):
    """
    Calculates the Raw Covariance Matrix for ALL derivatives (Spreads, Flies) 
    by projecting their standardized time series onto the standardized 3M Spread PC scores.
    """
    # 1. Align and clean data - ensure all derivatives are aligned with the PC scores index
    aligned_index = all_derivatives_df.index.intersection(scores_df.index)
    derivatives_aligned = all_derivatives_df.loc[aligned_index].dropna(axis=1)
    scores_aligned = scores_df.loc[aligned_index]
    
    if derivatives_aligned.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 2. Standardize all derivatives
    derivatives_mean = derivatives_aligned.mean()
    derivatives_std = derivatives_aligned.std()
    # Subtract mean is important for proper regression/loadings calculation
    derivatives_scaled = (derivatives_aligned - derivatives_mean) / derivatives_std
    
    # 3. Calculate Loadings (Beta) of each standardized derivative on the standardized PCs
    
    loadings_data = {}
    X = scores_aligned.iloc[:, :pc_count].values # Standardized PC scores
    
    # Use Linear Regression to find the standardized loading (beta) for each derivative
    for col in derivatives_scaled.columns:
        y = derivatives_scaled[col].values
        # Using intercept=False as both X (scores) and y (scaled derivative) are mean-zero
        reg = LinearRegression(fit_intercept=False) 
        reg.fit(X, y)
        loadings_data[col] = reg.coef_

    # L_D: Loadings of the full derivatives set D onto the PC space
    loadings_df = pd.DataFrame(
        loadings_data, 
        index=[f'PC{i+1}' for i in range(pc_count)]
    ).T
    
    # 4. Reconstruct the Covariance Matrix in Standardized Space
    # Sigma_Std = L_D * Lambda_p * L_D^T
    L_D = loadings_df.values
    lambda_p = eigenvalues[:pc_count]
    Sigma_Std = L_D @ np.diag(lambda_p) @ L_D.T
    
    # 5. Scale back to the original derivative data covariance matrix (Raw Space)
    # Cov_Raw = diag(sigma) * Sigma_Std * diag(sigma)
    Sigma_Raw = Sigma_Std * np.outer(derivatives_std.values, derivatives_std.values)
    
    Sigma_Raw_df = pd.DataFrame(Sigma_Raw, index=derivatives_aligned.columns, columns=derivatives_aligned.columns)
    
    return Sigma_Raw_df, derivatives_aligned

def calculate_best_and_worst_hedge_generalized(trade_label, Sigma_Raw_df):
    """
    Calculates the best/worst hedge using the generalized Raw Covariance Matrix (Sigma_Raw_df).
    """
    
    if trade_label not in Sigma_Raw_df.index:
        return None, None, None
        
    results = []
    
    # Iterate through all other derivatives as potential hedges
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label]
    
    for hedge_instrument in potential_hedges:
        
        # Terms from the reconstructed covariance matrix (Sigma)
        Var_Trade = Sigma_Raw_df.loc[trade_label, trade_label] # Var(T)
        Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument] # Var(H)
        Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument]    # Cov(T, H)
        
        # 1. Minimum Variance Hedge Ratio (k*)
        if Var_Hedge <= 1e-9: # Check for near-zero variance
            k_star = 0
        else:
            k_star = Cov_TH / Var_Hedge
            
        # 2. Residual Variance of the hedged portfolio (Var(T - k*H) = Var(T) - k*Cov(T,H))
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance) 
        
        # 3. Residual Volatility (Score) in BPS
        Residual_Volatility_BPS = np.sqrt(Residual_Variance) * 10000
        
        results.append({
            'Hedge Instrument': hedge_instrument,
            'Hedge Ratio (k*)': k_star,
            'Residual Volatility (BPS)': Residual_Volatility_BPS
        })

    if not results:
        return None, None, None
        
    results_df = pd.DataFrame(results)
    
    # Best hedge minimizes Residual Volatility
    best_hedge = results_df.sort_values(by='Residual Volatility (BPS)', ascending=True).iloc[0]
    
    # Worst hedge maximizes Residual Volatility
    worst_hedge = results_df.sort_values(by='Residual Volatility (BPS)', ascending=False).iloc[0]
    
    # Return the individual best/worst series AND the full DataFrame
    return best_hedge, worst_hedge, results_df

# --- Streamlit Application Layout (Modified Section 6) ---

st.title("SOFR Futures PCA Analyzer")

# [The remaining Streamlit setup, data loading, and processing sections 1-5 remain largely the same, 
# but are omitted here for brevity. All calculation functions above are included.]

# [ ... Streamlit setup and previous sections (1-5) ... ]

# --- Core Processing Logic ---
if not price_df_filtered.empty:
    
    # 1. Get the list of relevant contracts
    future_expiries_df = get_analysis_contracts(expiry_df, analysis_dt)
    
    if future_expiries_df.empty:
        st.warning("Could not establish a relevant contract curve. Please check your date filters.")
        st.stop()
        
    # 2. Transform historical prices to the required maturity curve
    analysis_curve_df, contract_labels = transform_to_analysis_curve(price_df_filtered, future_expiries_df)

    if analysis_curve_df.empty:
        st.warning("Data transformation failed. Check if contracts in the price file match contracts in the expiry file.")
        st.stop()
        
    # 3. Calculate Derivatives
    st.header("1. Data Derivatives Check (Contracts relevant to selected Analysis Date)")
    
    # 3M (k=1) - Used for PCA input
    spreads_3M_df = calculate_k_step_spreads(analysis_curve_df, 1)
    butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, 1)
    
    # 6M (k=2)
    spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, 2)
    butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, 2)
    
    # 12M (k=4)
    spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, 4)
    butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, 4)
    
    st.markdown("##### 3-Month Outright Spreads (k=1, e.g., Z25-H26)")
    st.dataframe(spreads_3M_df.head(5))
    
    if spreads_3M_df.empty:
        st.warning("3M Spreads could not be calculated. Need at least two contracts in the analysis curve.")
        st.stop()
        
    # 4. Perform PCA
    # 4a. PCA on 3M Spreads (Standard Method - Used for Fair Curve Reconstruction & Hedging)
    loadings_spread, explained_variance_ratio, eigenvalues, scores, spreads_3M_df_clean = perform_pca(spreads_3M_df)
    
    # 4b. PCA on Outright Prices (User Requested Independent Method - Unstandardized/Covariance)
    loadings_outright_direct, explained_variance_outright_direct = perform_pca_on_prices(analysis_curve_df)


    if loadings_spread is not None and loadings_outright_direct is not None:
        
        # [ ... Sections 2, 3, 4, 5 (Explained Variance, Loadings, Scores, Curve Snapshot) are here ... ]

        # NOTE: For the actual running code, ensure all the missing sections are present.
        # For this response, I am only showing the final, modified section 6 logic.
        
        # Reconstruct spreads (needed for fair curve, not hedging directly)
        # 1. Reconstruct 3M Spreads using only selected PCs
        data_mean = spreads_3M_df_clean.mean()
        data_std = spreads_3M_df_clean.std()
        
        # Ensure pc_count is defined (e.g., from the slider in section 2)
        # default_pc_count = min(3, len(explained_variance_ratio))
        # pc_count = st.session_state.get('pc_slider', default_pc_count)

        if 'pc_slider' not in st.session_state:
            # Re-run a minimal version of section 2 logic to get pc_count
            default_pc_count = min(3, len(explained_variance_ratio))
            # Placeholder value if slider hasn't been rendered/selected yet
            pc_count = default_pc_count 
        else:
            pc_count = st.session_state['pc_slider']


        scores_used = scores.values[:, :pc_count]
        loadings_used = loadings_spread.values[:, :pc_count]
        
        reconstructed_scaled = scores_used @ loadings_used.T
        
        reconstructed_spreads_3M = pd.DataFrame(
            reconstructed_scaled * data_std.values + data_mean.values,
            index=spreads_3M_df_clean.index, 
            columns=spreads_3M_df_clean.columns
        )

        # 2. Reconstruct Outright Prices and ALL Derivatives (3M, 6M, 12M)
        historical_outrights_df, historical_spreads_3M_df, historical_butterflies_3M_df, historical_spreads_6M_df, historical_butterflies_6M_df, historical_spreads_12M_df, historical_butterflies_12M_df = \
            reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_3M, spreads_3M_df, spreads_6M_df, butterflies_3M_df, butterflies_6M_df, spreads_12M_df, butterflies_12M_df)

        # --- HEDGING DATA PREPARATION (NEW) ---
        # 1. Combine all historical derivative time series into one DataFrame
        all_derivatives_list = [
            spreads_3M_df.rename(columns=lambda x: f"3M Spread: {x}"),
            butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}"),
            spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}"),
            butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}"),
            spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}"),
            butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}"),
        ]
        
        all_derivatives_df = pd.concat(all_derivatives_list, axis=1)

        # 2. Calculate the Generalized Covariance Matrix
        Sigma_Raw_df, all_derivatives_aligned = calculate_derivatives_covariance_generalized(
            all_derivatives_df, scores, eigenvalues, pc_count
        )
        
        
        # --------------------------- 6. PCA-Based Generalized Hedging Strategy ---------------------------
        st.header("6. PCA-Based Generalized Hedging Strategy")
        st.markdown(f"""
            This section calculates the **Minimum Variance Hedge Ratio ($k^*$ )** for *any* derivative trade, using *any* other derivative as a hedge. The calculation is based on the **full covariance matrix** of all derivatives, which is **reconstructed using the selected {pc_count} Principal Components** derived from the 3M Spreads (Level, Slope, Curvature, etc.).
            
            * **Trade:** Long 1 unit of the selected instrument.
            * **Hedge:** Short $k^*$ units of the hedging instrument.
        """)
        
        if Sigma_Raw_df.empty or Sigma_Raw_df.shape[0] < 2:
            st.warning("Not enough data to calculate generalized hedging correlations.")
        else:
            
            trade_selection = st.selectbox(
                "Select Trade Instrument (Long 1 unit):", 
                options=Sigma_Raw_df.columns.tolist(),
                index=0,
                key='trade_instrument_select'
            )
            
            # CALL TO THE GENERALIZED FUNCTION
            best_hedge_data, worst_hedge_data, all_results_df_full = calculate_best_and_worst_hedge_generalized(
                trade_selection, Sigma_Raw_df
            )
            
            if best_hedge_data is not None:
                
                col_best, col_worst = st.columns(2)
                
                with col_best:
                    st.success(f"Best Hedge for **Long 1x {trade_selection}**")
                    st.markdown(f"""
                        - **Hedge Instrument:** **{best_hedge_data['Hedge Instrument']}**
                        - **Hedge Action:** Short **{best_hedge_data['Hedge Ratio (k*)']:.4f}** units.
                        - **Residual Volatility (Score):** **{best_hedge_data['Residual Volatility (BPS)']:.2f} BPS** (Lowest Risk)
                    """)
                    
                with col_worst:
                    st.error(f"Worst Hedge for **Long 1x {trade_selection}**")
                    st.markdown(f"""
                        - **Hedge Instrument:** **{worst_hedge_data['Hedge Instrument']}**
                        - **Hedge Action:** Short **{worst_hedge_data['Hedge Ratio (k*)']:.4f}** units.
                        - **Residual Volatility (Score):** **{worst_hedge_data['Residual Volatility (BPS)']:.2f} BPS** (Highest Risk)
                    """)
                    
                st.markdown("---")
                st.markdown("###### Detailed Hedging Results (All Derivatives as Hedge Candidates)")
                
                # Use the full results DataFrame directly and sort it for display
                all_results_df_full = all_results_df_full.sort_values(by='Residual Volatility (BPS)', ascending=True)

                st.dataframe(
                    all_results_df_full.style.format({
                        'Hedge Ratio (k*)': "{:.4f}",
                        'Residual Volatility (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )
                
            else:
                st.warning("Hedging calculation failed for the selected trade. Check if enough historical data is available after filtering.")

    else:
        st.error("PCA failed. Please check your data quantity and quality.")
