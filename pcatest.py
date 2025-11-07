import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SOFR Futures PCA Analyzer (Outright Price Model)")

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
    Calculates spreads between contracts separated by 'k' steps (e.g., k=1 for 3M, k=2 for 6M).
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
    Calculates butterflies using contracts separated by 'k' steps (e.g., k=1 for 3M fly, k=2 for 6M fly).
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


# --- PRIMARY PCA FUNCTION (NOW ON OUTRIGHT PRICES - COVARIANCE MATRIX) ---
def perform_pca(price_df):
    """
    Performs PCA directly on Outright Price Levels using the COVARIANCE MATRIX 
    (unstandardized data), which results in a NON-UNIFORM PC1 (Level Factor).
    This is the new primary PCA model.
    """
    data_df_clean = price_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        # Need to return 5 values: loadings, explained_variance_ratio, eigenvalues, scores, data_df_clean
        return None, None, None, None, None 

    # Center the data, but DO NOT scale/standardize it (PCA on Covariance Matrix)
    data_centered = data_df_clean - data_df_clean.mean() 
    
    n_components = min(data_centered.shape)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_centered)
    
    # Loadings (Eigenvectors - the raw sensitivities)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    
    explained_variance_ratio = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    
    scores_df = pd.DataFrame(
        scores,
        index=data_df_clean.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings, explained_variance_ratio, eigenvalues, scores_df, data_df_clean


# --- RECONSTRUCTION LOGIC (NOW BASED ON OUTRIGHT PCA) ---

def _merge_original_and_reconstructed(original_df, reconstructed_df):
    """Helper to align and merge original and reconstructed dataframes."""
    
    # Align indices
    valid_indices = reconstructed_df.index.intersection(original_df.index)
    original_aligned = original_df.loc[valid_indices]
    reconstructed_aligned = reconstructed_df.loc[valid_indices]
    
    # Rename for merging
    original_rename = {col: col + ' (Original)' for col in original_aligned.columns}
    reconstructed_rename = {col: col + ' (PCA)' for col in reconstructed_aligned.columns}
    
    return pd.merge(
        original_aligned.rename(columns=original_rename),
        reconstructed_aligned.rename(columns=reconstructed_rename), 
        left_index=True, 
        right_index=True
    )

def reconstruct_prices_and_derivatives(analysis_curve_df_clean, loadings_outright, scores, pc_count, original_derivatives):
    """
    Reconstructs Outright Prices directly from PCA and recalculates all derivatives 
    from the reconstructed prices.
    """
    # 1. Reconstruct Outright Prices
    
    # Get Centered Data Mean (used for un-centering the reconstructed data)
    data_mean = analysis_curve_df_clean.mean() 
    
    # Use selected PC scores and loadings
    scores_used = scores.values[:, :pc_count]
    loadings_used = loadings_outright.values[:, :pc_count]
    
    # Reconstructed Centered Data: X_centered_reconstructed = Scores * Loadings^T
    reconstructed_centered = scores_used @ loadings_used.T
    
    # Reconstructed Prices: X_reconstructed = X_centered_reconstructed + Mean
    reconstructed_prices_array = reconstructed_centered + data_mean.values
    
    reconstructed_prices_df_raw = pd.DataFrame(
        reconstructed_prices_array,
        index=analysis_curve_df_clean.index,
        columns=analysis_curve_df_clean.columns
    )
    
    # Historical Outrights (Original vs. PCA Fair Price)
    historical_outrights = _merge_original_and_reconstructed(
        analysis_curve_df_clean, 
        reconstructed_prices_df_raw
    )

    # 2. Recalculate Derivatives from Reconstructed Prices
    
    # Helper to calculate a derivative from a price curve
    def calculate_k_step_spreads_reconstructed(price_df, k):
        spreads_data = {}
        for i in range(price_df.shape[1] - k):
            short_maturity = price_df.columns[i]
            long_maturity = price_df.columns[i+k]
            spread_label = f"{short_maturity}-{long_maturity}"
            spreads_data[spread_label] = price_df.iloc[:, i] - price_df.iloc[:, i+k]
        return pd.DataFrame(spreads_data)
        
    def calculate_k_step_butterflies_reconstructed(price_df, k):
        flies_data = {}
        for i in range(price_df.shape[1] - 2 * k):
            short_maturity = price_df.columns[i]
            center_maturity = price_df.columns[i+k]
            long_maturity = price_df.columns[i+2*k]
            fly_label = f"{short_maturity}-2x{center_maturity}+{long_maturity}"
            flies_data[fly_label] = price_df.iloc[:, i] - 2 * price_df.iloc[:, i+k] + price_df.iloc[:, i+2*k]
        return pd.DataFrame(flies_data)

    # Calculate PCA Fair Derivatives
    rec_spreads_3M = calculate_k_step_spreads_reconstructed(reconstructed_prices_df_raw, 1)
    rec_butterflies_3M = calculate_k_step_butterflies_reconstructed(reconstructed_prices_df_raw, 1)
    rec_spreads_6M = calculate_k_step_spreads_reconstructed(reconstructed_prices_df_raw, 2)
    rec_butterflies_6M = calculate_k_step_butterflies_reconstructed(reconstructed_prices_df_raw, 2)
    rec_spreads_12M = calculate_k_step_spreads_reconstructed(reconstructed_prices_df_raw, 4)
    rec_butterflies_12M = calculate_k_step_butterflies_reconstructed(reconstructed_prices_df_raw, 4)
    
    
    # Merge Original and PCA Fair Derivatives
    historical_spreads_3M = _merge_original_and_reconstructed(original_derivatives['spreads_3M'], rec_spreads_3M)
    historical_butterflies_3M = _merge_original_and_reconstructed(original_derivatives['butterflies_3M'], rec_butterflies_3M)
    historical_spreads_6M = _merge_original_and_reconstructed(original_derivatives['spreads_6M'], rec_spreads_6M)
    historical_butterflies_6M = _merge_original_and_reconstructed(original_derivatives['butterflies_6M'], rec_butterflies_6M)
    historical_spreads_12M = _merge_original_and_reconstructed(original_derivatives['spreads_12M'], rec_spreads_12M)
    historical_butterflies_12M = _merge_original_and_reconstructed(original_derivatives['butterflies_12M'], rec_butterflies_12M)
    
    return historical_outrights, historical_spreads_3M, historical_butterflies_3M, historical_spreads_6M, historical_butterflies_6M, historical_spreads_12M, historical_butterflies_12M


# --- OUTRIGHT HEDGING LOGIC (NEW SECTION 6) ---

def calculate_reconstructed_covariance(loadings_df, eigenvalues, pc_count):
    """
    Calculates the Raw Covariance Matrix of the Outright Prices using the 
    first 'pc_count' PCs: Sigma = L_p Lambda_p L_p^T
    """
    # 1. Select the loadings and eigenvalues for the used PCs
    L_p = loadings_df.iloc[:, :pc_count].values # Loadings (Eigenvectors on Covariance Matrix)
    lambda_p = eigenvalues[:pc_count]           # Eigenvalues (Variance of raw scores)
    
    # 2. Reconstruct the Covariance Matrix
    # Sigma = L_p * Lambda_p * L_p^T
    Sigma = L_p @ np.diag(lambda_p) @ L_p.T
    
    Sigma_df = pd.DataFrame(Sigma, index=loadings_df.index, columns=loadings_df.index)
    
    return Sigma_df

def calculate_best_and_worst_hedge_outright(trade_label, loadings_df, eigenvalues, pc_count):
    """
    Calculates the best (min residual risk) and worst (max residual risk) 
    hedge for a given Outright Contract trade using the reconstructed covariance matrix.
    (Section 6 - Outright Contracts only)
    """
    if trade_label not in loadings_df.index:
        return None, None, None
        
    # Reconstruct covariance matrix using selected PCs
    Sigma_reconstructed = calculate_reconstructed_covariance(
        loadings_df, eigenvalues, pc_count
    )
    
    trade_contract = trade_label
    
    results = []
    
    # Iterate through all other Outright Contracts as potential hedges
    potential_hedges = [col for col in Sigma_reconstructed.columns if col != trade_contract]
    
    for hedge_contract in potential_hedges:
        
        # Terms from the reconstructed covariance matrix (Sigma)
        Var_Trade = Sigma_reconstructed.loc[trade_contract, trade_contract] # Var(T)
        Var_Hedge = Sigma_reconstructed.loc[hedge_contract, hedge_contract] # Var(H)
        Cov_TH = Sigma_reconstructed.loc[trade_contract, hedge_contract]    # Cov(T, H)
        
        # 1. Minimum Variance Hedge Ratio (k*)
        if Var_Hedge <= 1e-9:
            k_star = 0
        else:
            k_star = Cov_TH / Var_Hedge
            
        # 2. Residual Variance of the hedged portfolio (Var(T - k*H) = Var(T) - k*Cov(T,H))
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance) 
        
        # 3. Residual Volatility (Score) in BPS
        Residual_Volatility_BPS = np.sqrt(Residual_Variance) * 10000
        
        results.append({
            'Hedge Contract': hedge_contract,
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


# --- GENERALIZED HEDGING LOGIC (Section 7) ---

def calculate_derivatives_covariance_generalized(all_derivatives_df, scores_df, eigenvalues, pc_count):
    """
    Calculates the Raw Covariance Matrix for ALL derivatives (Spreads, Flies) 
    by projecting their standardized time series onto the raw Outright Price PC scores.
    """
    # 1. Align and clean data - ensure all derivatives are aligned with the PC scores index
    aligned_index = all_derivatives_df.index.intersection(scores_df.index)
    derivatives_aligned = all_derivatives_df.loc[aligned_index].dropna(axis=1)
    scores_aligned = scores_df.loc[aligned_index] # Raw Outright PC Scores
    
    if derivatives_aligned.empty:
        # Return empty dataframes, but return all three expected variables
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 
        
    # 2. Standardize all derivatives
    derivatives_mean = derivatives_aligned.mean()
    derivatives_std = derivatives_aligned.std()
    # Subtract mean is important for proper regression/loadings calculation
    derivatives_scaled = (derivatives_aligned - derivatives_mean) / derivatives_std
    
    # 3. Calculate Loadings (Beta) of each standardized derivative on the RAW Outright PCs
    
    loadings_data = {}
    X = scores_aligned.iloc[:, :pc_count].values # Raw Outright PC scores (X is centered, not scaled)
    
    # The output L_D is the sensitivity (beta) of the STANDARDIZED derivative to the RAW PC score
    for col in derivatives_scaled.columns:
        y = derivatives_scaled[col].values
        # Using intercept=False as X (scores) is mean-zero and y (scaled derivative) is mean-zero
        reg = LinearRegression(fit_intercept=False) 
        reg.fit(X, y)
        loadings_data[col] = reg.coef_

    # L_D: Loadings of the full derivatives set D onto the PC space
    loadings_df = pd.DataFrame(
        loadings_data, 
        index=[f'PC{i+1}' for i in range(pc_count)]
    ).T
    
    # 4. Reconstruct the Covariance Matrix in Standardized Derivative Space
    # Sigma_Std = L_D * Lambda_p * L_D^T
    L_D = loadings_df.values
    lambda_p = eigenvalues[:pc_count] # Raw Outright PC Eigenvalues
    Sigma_Std = L_D @ np.diag(lambda_p) @ L_D.T
    
    # 5. Scale back to the original derivative data covariance matrix (Raw Space)
    # Cov_Raw = diag(sigma) * Sigma_Std * diag(sigma)
    Sigma_Raw = Sigma_Std * np.outer(derivatives_std.values, derivatives_std.values)
    
    Sigma_Raw_df = pd.DataFrame(Sigma_Raw, index=derivatives_aligned.columns, columns=derivatives_aligned.columns)
    
    return Sigma_Raw_df, derivatives_aligned, loadings_df

def calculate_best_and_worst_hedge_generalized(trade_label, Sigma_Raw_df):
    """
    Calculates the best/worst hedge using the generalized Raw Covariance Matrix (Sigma_Raw_df).
    (Section 7 - All Derivatives)
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

# --- FACTOR-BASED HEDGING LOGIC (Section 8) ---
# Note: The factor names are based on the Outright PCA result (PC1 is Non-Uniform Level/Duration)

def calculate_factor_sensitivities(loadings_df_gen, pc_count):
    """
    Calculates the Standardized Sensitivity (Beta) of every derivative to the first three 
    principal components (Level, Slope, Curvature). This uses the standardized loadings L_D 
    from the generalized regression (Section 7).
    """
    if loadings_df_gen.empty:
        return pd.DataFrame()

    # Define the factor mapping based on the first 3 PCs
    pc_map = {
        'PC1': 'Non-Uniform Level/Duration', 
        'PC2': 'Slope (Steepening/Flattening)', 
        'PC3': 'Curvature (Fly Risk)'
    }
    
    # Only use up to the number of available PCs, or 3, whichever is smaller
    available_pcs = loadings_df_gen.columns.intersection(list(pc_map.keys()))
    
    # Filter the generalized loadings L_D for the relevant PCs
    factor_sensitivities = loadings_df_gen.filter(items=available_pcs.tolist(), axis=1).copy()
    
    # Rename columns for clarity in the output
    factor_sensitivities.columns = [pc_map[col] for col in available_pcs]
    
    # The values in this table represent the standardized beta (sensitivity) 
    # to the raw PC score. We can treat these as the factor exposures.
    return factor_sensitivities

def calculate_factor_hedge_ratio(trade_label, hedge_label, factor_sensitivities_df, factor_name):
    """
    Calculates the hedge ratio k_factor to neutralize the factor exposure.
    k_factor = Trade_Exposure / Hedge_Exposure
    """
    if trade_label not in factor_sensitivities_df.index or hedge_label not in factor_sensitivities_df.index:
        return None, "One or both instruments not found in factor sensitivities."
        
    if factor_name not in factor_sensitivities_df.columns:
        return None, f"Factor '{factor_name}' not found."
        
    Trade_Exposure = factor_sensitivities_df.loc[trade_label, factor_name]
    Hedge_Exposure = factor_sensitivities_df.loc[hedge_label, factor_name]
    
    if abs(Hedge_Exposure) < 1e-9: # Avoid division by near-zero
        return None, "Hedge instrument has negligible sensitivity to this factor."
        
    # k_factor is the ratio of sensitivities
    k_factor = Trade_Exposure / Hedge_Exposure
    
    return k_factor, None

# --- Streamlit Application Layout ---

st.title("SOFR Futures PCA Analyzer (Outright Price Model)")

# --- Sidebar Inputs ---
st.sidebar.header("1. Data Uploads")
price_file = st.sidebar.file_uploader(
    "Upload Historical Price Data (e.g., 'sofr rates.csv')", 
    type=['csv'], 
    key='price_upload'
)
expiry_file = st.sidebar.file_uploader(
    "Upload Contract Expiry Dates (e.g., 'EXPIRY (2).csv')", 
    type=['csv'], 
    key='expiry_upload'
)

# Initialize dataframes
price_df = load_data(price_file)
expiry_df = load_data(expiry_file)

# Placeholder for L_D Loadings, calculated in Section 7 and used in Section 8
loadings_df_gen = pd.DataFrame()


if price_df is not None and expiry_df is not None:
    # --- Date Range Filter ---
    st.sidebar.header("2. Historical Date Range")
    min_date = price_df.index.min().date()
    max_date = price_df.index.max().date()
    
    start_date, end_date = st.sidebar.date_input(
        "Select Historical Data Range for PCA Calibration", 
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    price_df_filtered = price_df[(price_df.index.date >= start_date) & (price_df.index.date <= end_date)]
    
    # --- Analysis Date Selector (Maturity Roll) ---
    st.sidebar.header("3. Curve Analysis Date")
    
    default_analysis_date = end_date
    if default_analysis_date < min_date:
        default_analysis_date = min_date
        
    analysis_date = st.sidebar.date_input(
        "Select **Single Date** for Curve Snapshot", 
        value=default_analysis_date,
        min_value=min_date,
        max_value=max_date,
        key='analysis_date'
    )
    
    analysis_dt = datetime.combine(analysis_date, datetime.min.time())
    
else:
    st.info("Please upload both the Price Data and Expiry Data CSV files to begin the analysis.")
    st.stop()


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
    
    # 3M (k=1)
    spreads_3M_df = calculate_k_step_spreads(analysis_curve_df, 1)
    butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, 1)
    
    # 6M (k=2)
    spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, 2)
    butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, 2)
    
    # 12M (k=4)
    spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, 4)
    butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, 4)
    
    original_derivatives = {
        'spreads_3M': spreads_3M_df, 'butterflies_3M': butterflies_3M_df,
        'spreads_6M': spreads_6M_df, 'butterflies_6M': butterflies_6M_df,
        'spreads_12M': spreads_12M_df, 'butterflies_12M': butterflies_12M_df,
    }
    
    st.markdown("##### 3-Month Outright Spreads (k=1, e.g., Z25-H26)")
    st.dataframe(spreads_3M_df.head(5))
    
    if analysis_curve_df.shape[1] < 2:
        st.warning("Need at least two contracts in the analysis curve for PCA.")
        st.stop()
        
    # 4. Perform PCA (Now on Outright Prices - Unstandardized/Covariance Matrix)
    loadings_outright, explained_variance_ratio, eigenvalues, scores, analysis_curve_df_clean = perform_pca(analysis_curve_df)
    
    if loadings_outright is not None:
        
        # --- Explained Variance Visualization ---
        st.header("2. Explained Variance")
        variance_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
            'Explained Variance (%)': explained_variance_ratio * 100
        })
        variance_df['Cumulative Variance (%)'] = variance_df['Explained Variance (%)'].cumsum()
        
        col_var, col_pca_select = st.columns([1, 1])
        with col_var:
            st.dataframe(variance_df, use_container_width=True)
            
        default_pc_count = min(3, len(explained_variance_ratio))
        with col_pca_select:
            st.subheader("Fair Curve & Hedging Setup")
            pc_count = st.slider(
                "Select number of Principal Components (PCs) for Fair Curve & Hedging:",
                min_value=1,
                max_value=len(explained_variance_ratio),
                value=default_pc_count,
                key='pc_slider'
            )
            total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
            st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the **Outright Prices**. This is the risk model used.")
        
        
        # --- Component Loadings Heatmaps ---
        st.header("3. PC Loadings")
        
        # --- 3.1 Outright Loadings (New Primary Method) ---
        st.subheader("3.1 PC Loadings Heatmap (PC vs. Outright Contracts - Absolute Sensitivity)")
        st.markdown("""
            This heatmap shows the **Loadings (Eigenvectors)** of the first few PCs on each **Outright Contract**. These weights are derived from **Unstandardized PCA (Covariance Matrix)** and reflect the **absolute historical price volatility and duration** of each contract. 
            
            * **PC1 (Level/Duration) is Non-Uniform:** Longer-dated contracts have higher absolute weights.
        """)
        
        plt.style.use('default') 
        fig_outright_loading, ax_outright_loading = plt.subplots(figsize=(12, 6))
        
        loadings_outright_plot = loadings_outright.iloc[:, :default_pc_count]

        max_abs = loadings_outright_plot.abs().max().max()
        
        sns.heatmap(
            loadings_outright_plot, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".4f",
            linewidths=0.5, 
            linecolor='gray', 
            vmin=-max_abs, 
            vmax=max_abs,
            cbar_kws={'label': 'Absolute Price Sensitivity (Eigenvector Weight)'}
        )
        ax_outright_loading.set_title(f'3.1 Component Loadings for First {default_pc_count} PCs (Outright Prices)', fontsize=16)
        ax_outright_loading.set_xlabel('Principal Component')
        ax_outright_loading.set_ylabel('Outright Contract')
        st.pyplot(fig_outright_loading)
        
        
        # --- PC Scores Time Series Plot ---
        def plot_pc_scores(scores_df, explained_variance_ratio):
            """Plots the time series of the first 3 PC scores."""
            
            pc_labels = ['Level/Duration (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
            num_pcs = min(3, scores_df.shape[1])
            if num_pcs == 0: return None

            fig, axes = plt.subplots(nrows=num_pcs, ncols=1, figsize=(15, 4 * num_pcs), sharex=True)
            if num_pcs == 1: axes = [axes] 

            plt.suptitle("Time Series of Principal Component Scores (Risk Factors)", fontsize=16, y=1.02)

            for i in range(num_pcs):
                ax = axes[i]
                pc_label = pc_labels[i]
                variance_pct = explained_variance_ratio[i] * 100
                
                ax.plot(scores_df.index, scores_df.iloc[:, i], label=f'{pc_label} ({variance_pct:.2f}% Var.)', linewidth=1.5, color=plt.cm.tab10(i))
                
                ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
                
                ax.set_title(f'{pc_label} Factor Score (Explaining {variance_pct:.2f}% of Price Variance)', fontsize=14)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_ylabel('Score Value')
                ax.legend(loc='upper left')
                
            plt.xlabel('Date')
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            return fig

        st.header("4. PC Factor Scores Time Series")
        st.markdown("This plot shows the historical movement of the **latent risk factors** (Level, Slope, and Curvature) over the chosen historical range. The scores are derived from the **Outright Price PCA (3.1)**.")
        fig_scores = plot_pc_scores(scores, explained_variance_ratio)
        if fig_scores:
            st.pyplot(fig_scores)
            
        
        # --- Historical Reconstruction (Based on Outright Price PCA) ---
        
        historical_outrights_df, historical_spreads_3M_df, historical_butterflies_3M_df, historical_spreads_6M_df, historical_butterflies_6M_df, historical_spreads_12M_df, historical_butterflies_12M_df = \
            reconstruct_prices_and_derivatives(analysis_curve_df_clean, loadings_outright, scores, pc_count, original_derivatives)


        
        # --- HELPER FUNCTION FOR PLOTTING SNAPSHOTS (defined here to use local variables) ---
        def plot_snapshot(historical_df, derivative_type, analysis_dt, pc_count):
            """Plots and displays the table for a single derivative type snapshot."""
            
            if historical_df.empty:
                 st.info(f"Not enough contracts to calculate and plot {derivative_type} snapshot.")
                 return
                 
            try:
                # 1. Select the single day's data
                snapshot_original = historical_df.filter(regex='\(Original\)$').loc[[analysis_dt]].T
                snapshot_pca = historical_df.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
                
                # 2. Rename column (which is the datetime key) and clean the index labels
                snapshot_original.columns = ['Original']
                snapshot_original.index = snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)

                snapshot_pca.columns = ['PCA Fair']
                snapshot_pca.index = snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)

                # 3. Concatenate and drop NaNs (if any value is missing for a contract)
                comparison = pd.concat([snapshot_original, snapshot_pca], axis=1).dropna()
                
                if comparison.empty:
                    st.warning(f"No complete {derivative_type} data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')} after combining Original and PCA Fair values.")
                    return

                # --- Plot the Derivative ---
                fig, ax = plt.subplots(figsize=(15, 7))
                
                ax.plot(comparison.index, comparison['Original'], 
                              label=f'Original Market {derivative_type}', marker='o', linestyle='-', linewidth=2.5, color='blue')
                
                ax.plot(comparison.index, comparison['PCA Fair'], 
                              label=f'PCA Fair {derivative_type} ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='red')
                
                mispricing = comparison['Original'] - comparison['PCA Fair']
                ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7) 
                
                # Annotate the derivative with the largest absolute mispricing
                max_abs_mispricing = mispricing.abs().max()
                if max_abs_mispricing > 0:
                    mispricing_contract = mispricing.abs().idxmax()
                    mispricing_value = mispricing.loc[mispricing_contract] * 10000 # Convert to BPS
                    
                    ax.annotate(
                        f"Mispricing: {mispricing_value:.2f} BPS",
                        (mispricing_contract, comparison.loc[mispricing_contract]['Original']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                    )
                
                ax.set_title(f'Market {derivative_type} vs. PCA Fair {derivative_type}', fontsize=16)
                ax.set_xlabel(f'{derivative_type} Contract')
                ax.set_ylabel(f'{derivative_type} Value (Price Difference)')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle=':', alpha=0.6)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # --- Detailed Table ---
                st.markdown(f"###### {derivative_type} Mispricing")
                detailed_comparison = comparison.copy()
                detailed_comparison.index.name = f'{derivative_type} Contract'
                detailed_comparison['Mispricing (BPS)'] = mispricing * 10000
                detailed_comparison = detailed_comparison.rename(
                    columns={'Original': f'Original {derivative_type}', 'PCA Fair': f'PCA Fair {derivative_type}'}
                )
                
                st.dataframe(
                    detailed_comparison.style.format({
                        f'Original {derivative_type}': "{:.4f}",
                        f'PCA Fair {derivative_type}': "{:.4f}",
                        'Mispricing (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )

            except KeyError:
                st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for {derivative_type}. Please choose a different date within the historical range.")
        # --- END HELPER FUNCTION ---


        # --- 5. Curve Snapshot Analysis ---
        st.header("5. Curve Snapshot Analysis: " + analysis_date.strftime('%Y-%m-%d'))

        # --- 5.1 Outright Price Snapshot ---
        st.subheader("5.1 Outright Price Curve")
        
        try:
            # Note: historical_outrights_df already has the Original and PCA Fair columns
            curve_comparison = historical_outrights_df.loc[[analysis_dt]].T
            curve_comparison.columns = ['Snapshot'] # Use a temporary column name
            
            # Extract Original and PCA Fair data
            curve_snapshot_original = curve_comparison.filter(regex='\(Original\)$')
            curve_snapshot_pca = curve_comparison.filter(regex='\(PCA\)$')
            
            # Clean up the index names
            original_index_clean = curve_snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)
            pca_index_clean = curve_snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)
            
            # Recreate DataFrame
            curve_comparison = pd.DataFrame({
                'Original': curve_snapshot_original['Snapshot'].values,
                'PCA Fair': curve_snapshot_pca['Snapshot'].values
            }, index=original_index_clean).dropna()
            
            if curve_comparison.empty:
                st.warning(f"No complete Outright Price data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')} after combining Original and PCA Fair values.")
            else:
                # --- Plot the Curve ---
                fig_curve, ax_curve = plt.subplots(figsize=(15, 7))
                
                ax_curve.plot(curve_comparison.index, curve_comparison['Original'], 
                              label='Original Market Curve', marker='o', linestyle='-', linewidth=2.5, color='blue')
                
                ax_curve.plot(curve_comparison.index, curve_comparison['PCA Fair'], 
                              label=f'PCA Fair Curve ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='red')
                
                mispricing = curve_comparison['Original'] - curve_comparison['PCA Fair']
                
                max_abs_mispricing = mispricing.abs().max()
                if max_abs_mispricing > 0:
                    mispricing_contract = mispricing.abs().idxmax()
                    mispricing_value = mispricing.loc[mispricing_contract] * 10000 
                    
                    ax_curve.annotate(
                        f"Mispricing: {mispricing_value:.2f} BPS",
                        (mispricing_contract, curve_comparison.loc[mispricing_contract]['Original']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                    )
                
                ax_curve.set_title(f'Market Price Curve vs. PCA Fair Price Curve', fontsize=16)
                ax_curve.set_xlabel('Contract Maturity')
                ax_curve.set_ylabel('Price (100 - Rate)')
                ax_curve.legend(loc='upper right')
                ax_curve.grid(True, linestyle=':', alpha=0.6)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_curve)
                
                # --- Detailed Contract Price/Rate Table (Outright) ---
                st.markdown("###### Outright Price and Rate Mispricing")
                
                detailed_comparison = curve_comparison.copy()
                detailed_comparison.index.name = 'Contract'
                
                detailed_comparison['Original Rate (%)'] = 100.0 - detailed_comparison['Original']
                detailed_comparison['PCA Fair Rate (%)'] = 100.0 - detailed_comparison['PCA Fair']
                detailed_comparison['Mispricing (BPS)'] = (detailed_comparison['Original'] - detailed_comparison['PCA Fair']) * 10000

                detailed_comparison = detailed_comparison.rename(
                    columns={'Original': 'Original Price', 'PCA Fair': 'PCA Fair Price'}
                )
                
                detailed_comparison = detailed_comparison[[
                    'Original Price', 
                    'Original Rate (%)', 
                    'PCA Fair Price', 
                    'PCA Fair Rate (%)', 
                    'Mispricing (BPS)'
                ]]
                
                st.dataframe(
                    detailed_comparison.style.format({
                        'Original Price': "{:.4f}",
                        'PCA Fair Price': "{:.4f}",
                        'Original Rate (%)': "{:.4f}",
                        'PCA Fair Rate (%)': "{:.4f}",
                        'Mispricing (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )
                
        except KeyError:
            st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for Outright Prices. Please choose a different date within the historical range.")

        
        # --- 5.2 Spread Snapshot (3M) ---
        st.subheader("5.2 3M Spread Snapshot (k=1, e.g., Z25-H26)")
        plot_snapshot(historical_spreads_3M_df, "3M Spread", analysis_dt, pc_count)


        # --- 5.3 Butterfly (Fly) Snapshot (3M) ---
        if not historical_butterflies_3M_df.empty:
            st.subheader("5.3 3M Butterfly (Fly) Snapshot (k=1, e.g., Z25-2xH26+M26)")
            plot_snapshot(historical_butterflies_3M_df, "3M Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 3 or more) to calculate and plot 3M butterfly snapshot.")
            
        # --------------------------- 6-Month (k=2) Derivatives ---------------------------
        
        # --- 5.4 Spread Snapshot (6M) ---
        st.subheader("5.4 6M Spread Snapshot (k=2, e.g., Z25-M26)")
        plot_snapshot(historical_spreads_6M_df, "6M Spread", analysis_dt, pc_count)

        # --- 5.5 Butterfly (Fly) Snapshot (6M) ---
        if not historical_butterflies_6M_df.empty:
            st.subheader("5.5 6M Butterfly (Fly) Snapshot (k=2, e.g., Z25-2xM26+Z26)")
            plot_snapshot(historical_butterflies_6M_df, "6M Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 5 or more) to calculate and plot 6M butterfly snapshot.")

        # --------------------------- 12-Month (k=4) Derivatives ---------------------------
            
        # --- 5.6 Spread Snapshot (12M) ---
        st.subheader("5.6 12M Spread Snapshot (k=4, e.g., Z25-Z26)")
        plot_snapshot(historical_spreads_12M_df, "12M Spread", analysis_dt, pc_count)

        # --- 5.7 Butterfly (Fly) Snapshot (12M) ---
        if not historical_butterflies_12M_df.empty:
            st.subheader("5.7 12M Butterfly (Fly) Snapshot (k=4, e.g., Z25-2xZ26+Z27)")
            plot_snapshot(historical_butterflies_12M_df, "12M Butterfly", analysis_dt, pc_count)
        else:
            st.info("Not enough contracts (need 9 or more) to calculate and plot 12M butterfly snapshot.")


        # --------------------------- 6. PCA-Based Hedging Strategy (Outright Contracts ONLY - NEW SECTION) ---------------------------
        st.header("6. PCA-Based Hedging Strategy (Outright Contracts ONLY)")
        st.markdown(f"""
            This section calculates the **Minimum Variance Hedge Ratio ($k^*$ )** for a chosen **Outright Contract** trade, using *another Outright Contract* as the hedge. The calculation uses the **Covariance Matrix** of the **Outright Prices**, which is **reconstructed using the selected {pc_count} Principal Components**.
            
            * **Trade:** Long 1 unit of the selected contract.
            * **Hedge:** Short $k^*$ units of the hedging contract.
        """)
        
        if analysis_curve_df_clean.shape[1] < 2:
            st.warning("Need at least two outright contracts to analyze hedging.")
        else:
            
            default_trade = contract_labels[0]
                
            trade_selection_outright = st.selectbox(
                "Select Trade Outright Contract (Long 1 unit):", 
                options=analysis_curve_df_clean.columns.tolist(),
                index=0,
                key='trade_contract_select_outright'
            )
            
            # CALL TO THE OUTRIGHT HEDGE FUNCTION
            best_hedge_data_outright, worst_hedge_data_outright, all_results_df_full_outright = calculate_best_and_worst_hedge_outright(
                trade_selection_outright, loadings_outright, eigenvalues, pc_count
            )
            
            if best_hedge_data_outright is not None:
                
                col_best, col_worst = st.columns(2)
                
                with col_best:
                    st.success(f"Best Hedge for **Long 1x {trade_selection_outright}**")
                    st.markdown(f"""
                        - **Hedge Contract:** **{best_hedge_data_outright['Hedge Contract']}**
                        - **Hedge Action:** Short **{best_hedge_data_outright['Hedge Ratio (k*)']:.4f}** units.
                        - **Residual Volatility (Score):** **{best_hedge_data_outright['Residual Volatility (BPS)']:.2f} BPS** (Lowest Risk)
                    """)
                    
                with col_worst:
                    st.error(f"Worst Hedge for **Long 1x {trade_selection_outright}**")
                    st.markdown(f"""
                        - **Hedge Contract:** **{worst_hedge_data_outright['Hedge Contract']}**
                        - **Hedge Action:** Short **{worst_hedge_data_outright['Hedge Ratio (k*)']:.4f}** units.
                        - **Residual Volatility (Score):** **{worst_hedge_data_outright['Residual Volatility (BPS)']:.2f} BPS** (Highest Risk)
                    """)
                    
                st.markdown("---")
                st.markdown("###### Detailed Hedging Results (All Outright Contracts as Hedge Candidates)")
                
                # Use the full results DataFrame directly and sort it for display
                all_results_df_full_outright = all_results_df_full_outright.sort_values(by='Residual Volatility (BPS)', ascending=True)

                st.dataframe(
                    all_results_df_full_outright.style.format({
                        'Hedge Ratio (k*)': "{:.4f}",
                        'Residual Volatility (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )
                
            else:
                st.warning("Outright Hedging calculation failed. Check if enough historical data is available after filtering.")


        # --------------------------- 7. PCA-Based Generalized Hedging Strategy (Minimum Variance) ---------------------------
        st.header("7. PCA-Based Generalized Hedging Strategy (Minimum Variance)")
        st.markdown(f"""
            This section calculates the **Minimum Variance Hedge Ratio ($k^*$ )** for *any* derivative trade, using *any* other derivative as a hedge. The calculation is based on the **full covariance matrix** of all derivatives, which is **reconstructed using the selected {pc_count} Principal Components** derived from the **Outright Price PCA**.
            
            * **Trade:** Long 1 unit of the selected instrument.
            * **Hedge:** Short $k^*$ units of the hedging instrument.
        """)
        
        # --- HEDGING DATA PREPARATION ---
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

        # 2. Calculate the Generalized Covariance Matrix AND Loadings L_D
        Sigma_Raw_df, all_derivatives_aligned, loadings_df_gen = calculate_derivatives_covariance_generalized(
            all_derivatives_df, scores, eigenvalues, pc_count
        )
        
        if Sigma_Raw_df.empty or Sigma_Raw_df.shape[0] < 2:
            st.warning("Not enough data to calculate generalized hedging correlations.")
        else:
            
            trade_selection_gen = st.selectbox(
                "Select Trade Instrument (Long 1 unit):", 
                options=Sigma_Raw_df.columns.tolist(),
                index=0,
                key='trade_instrument_select_gen'
            )
            
            # CALL TO THE GENERALIZED FUNCTION
            best_hedge_data_gen, worst_hedge_data_gen, all_results_df_full_gen = calculate_best_and_worst_hedge_generalized(
                trade_selection_gen, Sigma_Raw_df
            )
            
            if best_hedge_data_gen is not None:
                
                col_best_gen, col_worst_gen = st.columns(2)
                
                with col_best_gen:
                    st.success(f"Best Hedge for **Long 1x {trade_selection_gen}**")
                    st.markdown(f"""
                        - **Hedge Instrument:** **{best_hedge_data_gen['Hedge Instrument']}**
                        - **Hedge Action:** Short **{best_hedge_data_gen['Hedge Ratio (k*)']:.4f}** units.
                        - **Residual Volatility (Score):** **{best_hedge_data_gen['Residual Volatility (BPS)']:.2f} BPS** (Lowest Risk)
                    """)
                    
                with col_worst_gen:
                    st.error(f"Worst Hedge for **Long 1x {trade_selection_gen}**")
                    st.markdown(f"""
                        - **Hedge Instrument:** **{worst_hedge_data_gen['Hedge Instrument']}**
                        - **Hedge Action:** Short **{worst_hedge_data_gen['Hedge Ratio (k*)']:.4f}** units.
                        - **Residual Volatility (Score):** **{worst_hedge_data_gen['Residual Volatility (BPS)']:.2f} BPS** (Highest Risk)
                    """)
                    
                st.markdown("---")
                st.markdown("###### Detailed Hedging Results (All Derivatives as Hedge Candidates)")
                
                # Use the full results DataFrame directly and sort it for display
                all_results_df_full_gen = all_results_df_full_gen.sort_values(by='Residual Volatility (BPS)', ascending=True)

                st.dataframe(
                    all_results_df_full_gen.style.format({
                        'Hedge Ratio (k*)': "{:.4f}",
                        'Residual Volatility (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )
                
            else:
                st.warning("Generalized Hedging calculation failed for the selected trade. Check if enough historical data is available after filtering.")


        # --------------------------- 8. PCA-Based Factor Hedging Strategy (Sensitivity Hedging) ---------------------------
        st.header("8. PCA-Based Factor Hedging Strategy (Sensitivity Hedging)")
        st.markdown(f"""
            This section calculates the hedge ratio ($k_{{factor}}$) required to **completely neutralize** the exposure of a chosen trade to a specific **macro risk factor** (Non-Uniform Level/Duration, Slope, or Curvature) derived from the **Outright Price PCA**. This is a purely factor-driven hedge.
            
            * **Factor View:** Hedge is used to remove sensitivity to the selected factor.
            * **Trade:** Long 1 unit of the selected instrument.
            * **Hedge:** Short $k_{{factor}}$ units of the hedging instrument.
            * **Formula:** $k_{{factor}} = \\frac{{\\text{{Sensitivity}}(\\text{{Trade}}, \\text{{Factor}})}}{{\\text{{Sensitivity}}(\\text{{Hedge}}, \\text{{Factor}})}}$
        """)
        
        if loadings_df_gen.empty or loadings_df_gen.shape[0] < 2:
             st.warning("Factor sensitivity data is unavailable. Please ensure Section 7 successfully calculated the loadings.")
        else:
            
            # 1. Calculate the sensitivities from the generalized loadings L_D
            factor_sensitivities_df = calculate_factor_sensitivities(loadings_df_gen, pc_count)
            
            if factor_sensitivities_df.empty:
                 st.error("Factor sensitivity calculation failed.")
                 st.stop()
                 
            # 2. Setup the selection boxes
            col_trade_sel, col_hedge_sel, col_factor_sel = st.columns(3)
            
            instrument_options = factor_sensitivities_df.index.tolist()
            factor_options = factor_sensitivities_df.columns.tolist()

            with col_trade_sel:
                trade_selection_factor = st.selectbox(
                    "1. Select Trade Instrument (Long 1 unit):", 
                    options=instrument_options,
                    index=0,
                    key='trade_instrument_factor'
                )
            with col_hedge_sel:
                # Exclude the trade instrument itself from the hedge candidates
                hedge_options = [c for c in instrument_options if c != trade_selection_factor]
                hedge_selection_factor = st.selectbox(
                    "2. Select Hedge Instrument:", 
                    options=hedge_options,
                    index=0 if hedge_options else None,
                    key='hedge_instrument_factor'
                )
            with col_factor_sel:
                factor_selection = st.selectbox(
                    "3. Select Factor to Neutralize:", 
                    options=factor_options,
                    index=0,
                    key='factor_select'
                )

            st.markdown("---")
            
            # 4. Calculate Hedge Ratio
            if trade_selection_factor and hedge_selection_factor and factor_selection:
                k_factor, error_msg = calculate_factor_hedge_ratio(
                    trade_selection_factor, 
                    hedge_selection_factor, 
                    factor_sensitivities_df, 
                    factor_selection
                )

                if k_factor is not None:
                    st.success(f"**Factor Hedge Result**")
                    st.markdown(f"""
                        To neutralize the **{factor_selection}** risk (i.e., making the portfolio $\\beta_{{\\text{{Factor}}}} = 0$):
                        
                        * **Trade:** Long 1 unit of **{trade_selection_factor}**
                        * **Hedge Action:** Short **{k_factor:.4f}** units of **{hedge_selection_factor}**
                    """)
                    
                    # Display the sensitivities that led to the ratio
                    col_trade_sens, col_hedge_sens = st.columns(2)
                    with col_trade_sens:
                        st.info(f"Trade Sensitivity to {factor_selection}: **{factor_sensitivities_df.loc[trade_selection_factor, factor_selection]:.4f}**")
                    with col_hedge_sens:
                        st.info(f"Hedge Sensitivity to {factor_selection}: **{factor_sensitivities_df.loc[hedge_selection_factor, factor_selection]:.4f}**")
                        
                    st.markdown("---")
                    
                    # 5. Display Full Sensitivities Table
                    st.subheader(f"Factor Sensitivities (Standardized Beta) Table")
                    st.markdown("The values below are the standardized exposures of each instrument to the three main risk factors derived from the **Outright Price PCA**.")
                    
                    st.dataframe(
                        factor_sensitivities_df.style.format("{:.4f}"),
                        use_container_width=True
                    )

                else:
                    st.error(f"Factor hedge calculation failed: {error_msg}")
            else:
                 st.info("Please ensure both trade and hedge instruments are selected.")


    else:
        st.error("PCA failed. Please check your data quantity and quality.")
