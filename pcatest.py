import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression # Added for Generalized Hedging
from scipy.linalg import solve # ADDED: Required for PC1/PC2 Neutrality
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

# --- RECONSTRUCTION LOGIC ---

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


# --- MINIMUM VARIANCE HEDGING LOGIC (Section 6) ---

def calculate_reconstructed_covariance(loadings_df, eigenvalues, spread_std_dev, pc_count):
    """
    Calculates the covariance matrix of the STANDARDIZED spreads 
    reconstructed using the first 'pc_count' PCs: Sigma_scaled = L_p Lambda_p L_p^T
    Then scales back to original spread space: Sigma = (diag(sigma)) * Sigma_scaled * (diag(sigma))
    """
    # 1. Select the loadings and eigenvalues for the used PCs
    L_p = loadings_df.iloc[:, :pc_count].values # Loadings (Eigenvectors on Correlation Matrix)
    lambda_p = eigenvalues[:pc_count]           # Eigenvalues (Variance of standardized scores)
    
    # 2. Reconstruct the Covariance Matrix of the Standardized Data
    # Sigma_scaled = L_p * Lambda_p * L_p^T
    Sigma_scaled = L_p @ np.diag(lambda_p) @ L_p.T
    
    # 3. Scale back to the original spread data covariance matrix
    # Cov(X) = diag(sigma) * Cov(Z) * diag(sigma)
    Sigma = Sigma_scaled * np.outer(spread_std_dev.values, spread_std_dev.values)
    
    Sigma_df = pd.DataFrame(Sigma, index=loadings_df.index, columns=loadings_df.index)
    
    return Sigma_df

def calculate_best_and_worst_hedge_3M(trade_label, loadings_df, eigenvalues, pc_count, spreads_3M_df_clean):
    """
    Calculates the best (min residual risk) and worst (max residual risk) 
    hedge for a given 3M spread trade using the reconstructed covariance matrix, 
    and returns the full results DataFrame as well. (Section 6 - 3M Spreads only)
    """
    if trade_label not in loadings_df.index:
        return None, None, None
        
    spread_std_dev = spreads_3M_df_clean.std()
    
    # Reconstruct covariance matrix using selected PCs
    Sigma_reconstructed = calculate_reconstructed_covariance(
        loadings_df, eigenvalues, spread_std_dev, pc_count
    )
    
    trade_spread = trade_label
    
    results = []
    
    # Iterate through all other 3M spreads as potential hedges
    potential_hedges = [col for col in Sigma_reconstructed.columns if col != trade_spread]
    
    for hedge_spread in potential_hedges:
        
        # Terms from the reconstructed covariance matrix (Sigma)
        Var_Trade = Sigma_reconstructed.loc[trade_spread, trade_spread] # Var(T)
        Var_Hedge = Sigma_reconstructed.loc[hedge_spread, hedge_spread] # Var(H)
        Cov_TH = Sigma_reconstructed.loc[trade_spread, hedge_spread]    # Cov(T, H)
        
        # 1. Minimum Variance Hedge Ratio (k*)
        if Var_Hedge == 0:
            k_star = 0
        else:
            k_star = Cov_TH / Var_Hedge
            
        # 2. Residual Variance of the hedged portfolio (Var(T - k*H) = Var(T) - k*Cov(T,H))
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance) 
        
        # 3. Residual Volatility (Score) in BPS
        Residual_Volatility_BPS = np.sqrt(Residual_Variance) * 10000
        
        results.append({
            'Hedge Spread': hedge_spread,
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


# --- GENERALIZED HEDGING LOGIC (Section 7 and 8) ---

def calculate_derivatives_covariance_generalized(all_derivatives_df, scores_df, eigenvalues, pc_count):
    """
    Calculates the Raw Covariance Matrix and Loadings (Betas) for ALL derivatives 
    by projecting their standardized time series onto the standardized 3M Spread PC scores.
    
    MODIFIED to also return the loadings (generalized_loadings) for Section 8.
    """
    # 1. Align and clean data - ensure all derivatives are aligned with the PC scores index
    aligned_index = all_derivatives_df.index.intersection(scores_df.index)
    derivatives_aligned = all_derivatives_df.loc[aligned_index].dropna(axis=1)
    scores_aligned = scores_df.loc[aligned_index]
    
    if derivatives_aligned.empty:
        # MODIFIED RETURN: Now returns a third, empty DataFrame for loadings
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
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

    loadings_df = pd.DataFrame(
        loadings_data, 
        index=[f'PC{i+1}' for i in range(pc_count)]
    ).T # Transpose to get Instruments x PCs
    
    # 4. Reconstruct the Raw Covariance Matrix
    L_p = loadings_df.values # Loadings of derivatives on standardized PCs
    lambda_p = eigenvalues[:pc_count] # Variance of the standardized 3M Spread PCs
    
    # Covariance Matrix of Standardized Derivatives: Sigma_Z = L_p * Lambda_p * L_p^T
    Sigma_Z = L_p @ np.diag(lambda_p) @ L_p.T
    
    # Scale back to the original derivative space: Sigma_Raw = diag(sigma) * Sigma_Z * diag(sigma)
    Sigma_Raw = Sigma_Z * np.outer(derivatives_std.values, derivatives_std.values)
    
    Sigma_Raw_df = pd.DataFrame(Sigma_Raw, index=derivatives_aligned.columns, columns=derivatives_aligned.columns)
    
    return Sigma_Raw_df, derivatives_aligned, loadings_df # MODIFIED: Return loadings_df

def calculate_best_and_worst_hedge_generalized(trade_label, Sigma_Raw_df):
    """
    Calculates the best/worst hedge using the generalized Raw Covariance Matrix (Sigma_Raw_df). (Section 7 - All Derivatives)
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
        Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument] # Cov(T, H)
        
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
    
    return best_hedge, worst_hedge, results_df

def solve_pc1_pc2_neutral_hedge(trade_instrument, hedge1, hedge2, generalized_loadings_df):
    """
    Solves for the hedge weights h1 and h2 required to neutralize a trade
    to PC1 and PC2 factors using the generalized loadings.
    (B_h1 * h1) + (B_h2 * h2) = B_t  (Matrix form: A * H = B)
    """
    try:
        # Check if PC1 and PC2 are available
        if 'PC1' not in generalized_loadings_df.columns or 'PC2' not in generalized_loadings_df.columns:
             return None, None, 0.0 # Indicate failure if PCs < 2

        # Get Betas for PC1 and PC2 for all three instruments
        # Loadings here are the Betas of the instrument's standardized P&L on the standardized PC scores
        beta_t = generalized_loadings_df.loc[trade_instrument, ['PC1', 'PC2']].values
        beta_h1 = generalized_loadings_df.loc[hedge1, ['PC1', 'PC2']].values
        beta_h2 = generalized_loadings_df.loc[hedge2, ['PC1', 'PC2']].values

        # A is the matrix of hedge betas (2x2)
        # Note: We solve for T = h1*H1 + h2*H2
        A = np.array([beta_h1, beta_h2]).T 
        
        # B is the vector of trade betas (2x1) - [Beta_T_PC1, Beta_T_PC2]
        B = beta_t

        # Solve for H (h1, h2)
        H = solve(A, B)
        h1, h2 = H[0], H[1]
        
        # Calculate the residual PC3 exposure (if PC3 exists and was calculated)
        residual_pc3 = 0.0
        if 'PC3' in generalized_loadings_df.columns:
            beta_t_pc3 = generalized_loadings_df.loc[trade_instrument, 'PC3']
            beta_h1_pc3 = generalized_loadings_df.loc[hedge1, 'PC3']
            beta_h2_pc3 = generalized_loadings_df.loc[hedge2, 'PC3']
            # Residual is the total PC3 exposure left in the portfolio: T - h1*H1 - h2*H2
            residual_pc3 = beta_t_pc3 - h1 * beta_h1_pc3 - h2 * beta_h2_pc3
            
        return h1, h2, residual_pc3
        
    except np.linalg.LinAlgError:
        # Singular matrix (hedges are too correlated/identical)
        return None, None, None 
    except KeyError as e:
        # Instrument not in loadings or PC is missing
        return None, None, None

# --- STREAMLIT APP LOGIC START ---

st.title("SOFR Futures PCA Analyzer")

st.sidebar.header("1. Upload Data")
price_file = st.sidebar.file_uploader("Upload Price Data CSV", type=['csv'], key="price_upload")
expiry_file = st.sidebar.file_uploader("Upload Expiry Data CSV", type=['csv'], key="expiry_upload")

price_df = load_data(price_file)
expiry_df = load_data(expiry_file)

if price_df is None or expiry_df is None:
    st.info("Please upload both the Price Data and Expiry Data CSV files in the sidebar to begin the analysis.")
    st.stop()

# --- SIDEBAR CONTROLS ---

st.sidebar.header("2. Analysis Parameters")

# Filter for the earliest date common to both dataframes
if not price_df.index.empty:
    min_date = price_df.index.min().to_pydatetime().date()
else:
    min_date = date(2020, 1, 1)

# Default to the most recent date available in the price data
if not price_df.index.empty:
    default_analysis_date = price_df.index.max().to_pydatetime().date()
else:
    default_analysis_date = date.today()

analysis_dt = st.sidebar.date_input(
    "Analysis Date (for snapshot)",
    value=default_analysis_date,
    min_value=min_date,
    max_value=default_analysis_date
)
analysis_date = datetime.combine(analysis_dt, datetime.min.time())

# Determine relevant contracts based on the analysis date
future_expiries_df = get_analysis_contracts(expiry_df, analysis_date)
analysis_curve_df, contract_list = transform_to_analysis_curve(price_df, future_expiries_df)

if analysis_curve_df.empty:
    st.error("No contracts found with prices on or after the selected analysis date.")
    st.stop()

# Derivative steps selection
st.sidebar.subheader("Derivative Tenors")
k_spread = st.sidebar.number_input("Spread Step (K=1 for 3M, K=2 for 6M, etc.)", min_value=1, max_value=4, value=1, key='k_spread')
k_fly = st.sidebar.number_input("Butterfly Step (K=1 for 3M fly, K=2 for 6M fly, etc.)", min_value=1, max_value=4, value=1, key='k_fly')

# PCA parameters
st.sidebar.subheader("PCA Parameters")
max_pcs = analysis_curve_df.shape[1] - k_spread if analysis_curve_df.shape[1] > k_spread else 1
pc_count = st.sidebar.number_input("Number of Principal Components (K) to use", min_value=1, max_value=max_pcs, value=min(3, max_pcs))

# --- PCA Execution ---

# 1. Calculate 3M Spreads (Base for Fair Curve PCA)
spreads_3M_df = calculate_k_step_spreads(analysis_curve_df, k=1)
if spreads_3M_df.empty:
    st.error("Not enough contracts (need 2 or more) to calculate 3M spreads.")
    st.stop()
    
# 2. Perform PCA on 3M spreads (Correlations Matrix PCA)
loadings_corr, explained_variance_ratio, eigenvalues, scores, spreads_3M_df_clean = perform_pca(spreads_3M_df)

# Check if PCA was successful
if loadings_corr is None:
    st.error("PCA failed to converge. Please ensure your data has sufficient historical depth and non-zero variance.")
    st.stop()
    
# Calculate other derivatives for analysis and generalized hedging
spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, k=2)
spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, k=4)
butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, k=1)
butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, k=2)
butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, k=4)


# --- 3. Explained Variance ---
st.header("3. Explained Variance & Principal Components (3M Spread Basis)")
st.markdown(f"PCA performed on **{spreads_3M_df_clean.shape[0]}** daily observations of **{spreads_3M_df_clean.shape[1]}** 3-month spreads.")

# Variance Plot
fig, ax = plt.subplots(figsize=(10, 4))
cumulative_variance = np.cumsum(explained_variance_ratio)
ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.6, color='skyblue', label='Individual')
ax.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance * 100, marker='o', color='navy', label='Cumulative')
ax.axhline(y=90, color='r', linestyle='--', linewidth=1, label='90% Threshold')
ax.set_title("Explained Variance by Principal Component")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance (%)")
ax.legend(loc='center right')
ax.grid(axis='y', linestyle='--')
st.pyplot(fig)

# Variance Table
variance_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
    'Explained Variance (%)': explained_variance_ratio * 100,
    'Cumulative Variance (%)': cumulative_variance * 100
})
st.dataframe(variance_df.style.format("{:.2f}"), use_container_width=True)

# --- 4. PCA Loadings (Eigenvectors) ---
st.header("4. PCA Loadings (3M Spread Basis)")
st.markdown("Loadings show the sensitivity of each 3M spread to the risk factors (PCs).")

# Select the Loadings for the selected number of PCs
loadings_display = loadings_corr.iloc[:, :pc_count]

fig_load, ax_load = plt.subplots(figsize=(12, 6))
loadings_display.plot(kind='bar', ax=ax_load)
ax_load.set_title(f"Loadings (Sensitivities) for PC1 to PC{pc_count}")
ax_load.set_xlabel("3M Spread Contract")
ax_load.set_ylabel("Loading Value")
ax_load.legend(title='Principal Component', loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_load)

# Loadings Table
st.markdown("###### Loadings Table")
st.dataframe(loadings_display.style.format("{:.4f}"), use_container_width=True)

# --- 5. Fair Curve & Mispricing Snapshot ---

st.header("5. Fair Curve & Mispricing Snapshot")

# Reconstruct all derivatives using the selected PC count
num_pcs_used = pc_count
reconstructed_scores = scores.iloc[:, :num_pcs_used]

# Reconstruct the standardized 3M spreads Z_PCA = S_PCA / sigma
reconstructed_spreads_scaled = reconstructed_scores.dot(loadings_corr.iloc[:, :num_pcs_used].T)

# Unscale back to basis points: S_PCA = Z_PCA * sigma
spreads_std = spreads_3M_df_clean.std()
reconstructed_spreads_3M_df = reconstructed_spreads_scaled * spreads_std

# Reconstruct all instruments from the reconstructed 3M spreads
(
    historical_outrights_df, 
    historical_spreads_3M_df, 
    historical_butterflies_3M_df, 
    historical_spreads_6M_df, 
    historical_butterflies_6M_df,
    historical_spreads_12M_df,
    historical_butterflies_12M_df
) = reconstruct_prices_and_derivatives(
    analysis_curve_df, 
    reconstructed_spreads_3M_df, 
    spreads_3M_df, 
    spreads_6M_df, 
    butterflies_3M_df, 
    butterflies_6M_df,
    spreads_12M_df,
    butterflies_12M_df
)

def plot_snapshot(historical_df, derivative_type, analysis_date, pc_count):
    """Helper function to plot and display the mispricing table."""
    try:
        # Filter for the specific analysis date
        snapshot = historical_df.loc[analysis_date]
        
        original_cols = [col for col in snapshot.index if '(Original)' in col]
        pca_cols = [col for col in snapshot.index if '(PCA)' in col]
        
        original_values = snapshot[original_cols].rename(index=lambda x: x.replace(' (Original)', ''))
        pca_values = snapshot[pca_cols].rename(index=lambda x: x.replace(' (PCA)', ''))
        
        comparison = pd.concat([original_values, pca_values], axis=1)
        comparison.columns = ['Original', 'PCA Fair']
        
        mispricing = comparison['Original'] - comparison['PCA Fair']
        
        # Plotting
        fig_snap, ax_snap = plt.subplots(figsize=(12, 5))
        comparison['Original'].plot(ax=ax_snap, kind='bar', position=1, width=0.4, color='darkblue', label='Original')
        comparison['PCA Fair'].plot(ax=ax_snap, kind='bar', position=0, width=0.4, color='lightblue', label=f'PCA Fair (K={pc_count})')
        
        ax_snap.set_title(f"{derivative_type} Snapshot and PCA Fair Value on {analysis_date.strftime('%Y-%m-%d')}")
        ax_snap.set_ylabel(f"{derivative_type} Value (Basis Points)")
        ax_snap.set_xlabel(f"{derivative_type} Instrument")
        ax_snap.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_snap)
        
        # --- Detailed Table ---
        st.markdown(f"###### {derivative_type} Mispricing")
        detailed_comparison = comparison.copy()
        detailed_comparison.index.name = f'{derivative_type} Contract'
        detailed_comparison['Mispricing (BPS)'] = mispricing * 10000
        
        st.dataframe(
            detailed_comparison.style.format({
                'Original': "{:.4f}",
                'PCA Fair': "{:.4f}",
                'Mispricing (BPS)': "{:.2f}"
            }),
            use_container_width=True
        )

    except KeyError:
        st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for {derivative_type}. Please choose a different date within the historical range.")


# --- 5.1 Outright Price Snapshot ---
st.subheader("5.1 Outright Price Snapshot (P)")
try:
    # Filter for the specific analysis date
    outright_snapshot = historical_outrights_df.loc[analysis_date]
    
    original_cols = [col for col in outright_snapshot.index if '(Original)' in col]
    pca_cols = [col for col in outright_snapshot.index if '(PCA)' in col]
    
    original_prices = outright_snapshot[original_cols].rename(index=lambda x: x.replace(' (Original)', ''))
    pca_prices = outright_snapshot[pca_cols].rename(index=lambda x: x.replace(' (PCA)', ''))
    
    comparison = pd.concat([original_prices, pca_prices], axis=1)
    comparison.columns = ['Original Price', 'PCA Fair Price']
    
    # Calculate Rate (%) and Mispricing (BPS)
    comparison['Original Rate (%)'] = (100 - comparison['Original Price'])
    comparison['PCA Fair Rate (%)'] = (100 - comparison['PCA Fair Price'])
    mispricing = comparison['Original Price'] - comparison['PCA Fair Price']
    comparison['Mispricing (BPS)'] = mispricing * 10000

    # Plotting
    fig_price, ax_price = plt.subplots(figsize=(12, 5))
    comparison['Original Price'].plot(ax=ax_price, kind='line', marker='o', color='darkblue', label='Original Price')
    comparison['PCA Fair Price'].plot(ax=ax_price, kind='line', marker='x', color='lightblue', label=f'PCA Fair Price (K={pc_count})')
    
    ax_price.set_title(f"Outright Price Snapshot and PCA Fair Value on {analysis_date.strftime('%Y-%m-%d')}")
    ax_price.set_ylabel("Futures Price")
    ax_price.set_xlabel("Contract")
    ax_price.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_price)
    
    # --- Detailed Table ---
    st.markdown("###### Outright Price Mispricing")
    detailed_comparison = comparison.copy()
    detailed_comparison.index.name = 'Contract'
    detailed_comparison = detailed_comparison[[
        'Original Price', 
        'PCA Fair Price', 
        'Original Rate (%)', 
        'PCA Fair Rate (%)', 
        'Mispricing (BPS)'
    ]]
    
    # Display the table, formatted for financial data
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


# --- 5.2 Spread Snapshot (k=1, 3M) ---
st.subheader(f"5.2 {k_spread}M Spread Snapshot (C1-C2)")
plot_snapshot(historical_spreads_3M_df, "Spread (3M)", analysis_date, pc_count)


# --- 5.3 Butterfly (Fly) Snapshot (k=1, 3M) ---
if not historical_butterflies_3M_df.empty:
    st.subheader(f"5.3 {k_fly}M Butterfly (Fly) Snapshot (C1-2xC2+C3)")
    plot_snapshot(historical_butterflies_3M_df, "Butterfly (3M)", analysis_date, pc_count)
else:
    st.info("Not enough contracts (need 3 or more) to calculate and plot 3M butterfly snapshot.")


# --- 6. Minimum Variance Hedge (MVH) - 3M Spreads Only ---

st.header("6. Minimum Variance Hedge (MVH) - 3M Spreads")
st.markdown("Calculates the hedge ratio (k*) that minimizes the residual volatility between two **3M spreads** using the reconstructed covariance matrix from the selected K PCs.")

trade_selection_mvh = st.selectbox("Select Trade Spread (T)", spreads_3M_df_clean.columns.tolist(), key='trade_mvh')

if trade_selection_mvh:
    best_hedge_data, worst_hedge_data, all_results_df_full = calculate_best_and_worst_hedge_3M(
        trade_selection_mvh, 
        loadings_corr, 
        eigenvalues, 
        pc_count, 
        spreads_3M_df_clean
    )
    
    if best_hedge_data is not None:
        
        # Determine the hedge action (opposite of k* for Long trade T - k*H)
        best_hedge_action = "Short" if best_hedge_data['Hedge Ratio (k*)'] >= 0 else "Long"
        worst_hedge_action = "Short" if worst_hedge_data['Hedge Ratio (k*)'] >= 0 else "Long"

        col_best, col_worst = st.columns(2)
        
        with col_best:
            st.success(f"Best Hedge for **Long 1x {trade_selection_mvh}**")
            st.markdown(f"""
                - **Hedge Spread:** **{best_hedge_data['Hedge Spread']}**
                - **Hedge Action:** {best_hedge_action} **{abs(best_hedge_data['Hedge Ratio (k*)']):.4f}** units.
                - **Residual Volatility (Score):** **{best_hedge_data['Residual Volatility (BPS)']:.2f} BPS** (Lowest Risk)
            """)
            
        with col_worst:
            st.error(f"Worst Hedge for **Long 1x {trade_selection_mvh}**")
            st.markdown(f"""
                - **Hedge Spread:** **{worst_hedge_data['Hedge Spread']}**
                - **Hedge Action:** {worst_hedge_action} **{abs(worst_hedge_data['Hedge Ratio (k*)']):.4f}** units.
                - **Residual Volatility (Score):** **{worst_hedge_data['Residual Volatility (BPS)']:.2f} BPS** (Highest Risk)
            """)
            
        st.markdown("---")
        st.markdown("###### Detailed Hedging Results (3M Spreads Only)")
        
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
        st.warning("MVH calculation failed for the selected spread. Check if enough historical data is available after filtering.")


# --- 7. Generalized Minimum Variance Hedge (MVH) - All Derivatives ---

st.header("7. Generalized Minimum Variance Hedge (MVH) - All Derivatives")
st.markdown("Calculates the MVH using all derivative types (Spreads & Butterflies) as potential trade and hedge instruments.")

# Combine all derivative dataframes
all_derivatives_list = [spreads_3M_df, spreads_6M_df, spreads_12M_df, butterflies_3M_df, butterflies_6M_df, butterflies_12M_df]
all_derivatives_df = pd.concat(all_derivatives_list, axis=1).dropna(axis=1, how='all')

# Execute the generalized covariance and loading calculation (MODIFIED function)
Sigma_Raw_df, all_derivatives_df_clean, generalized_loadings = calculate_derivatives_covariance_generalized(
    all_derivatives_df, 
    scores, 
    eigenvalues, 
    pc_count
)

if not Sigma_Raw_df.empty:
    
    all_instruments = Sigma_Raw_df.index.tolist()
    trade_selection_gen = st.selectbox("Select Trade Instrument (T)", all_instruments, key='trade_gen')

    if trade_selection_gen:
        best_hedge_data_gen, worst_hedge_data_gen, all_results_df_full_gen = calculate_best_and_worst_hedge_generalized(
            trade_selection_gen, 
            Sigma_Raw_df
        )
        
        if best_hedge_data_gen is not None:
            
            # Determine the hedge action (opposite of k* for Long trade T - k*H)
            best_hedge_action_gen = "Short" if best_hedge_data_gen['Hedge Ratio (k*)'] >= 0 else "Long"
            worst_hedge_action_gen = "Short" if worst_hedge_data_gen['Hedge Ratio (k*)'] >= 0 else "Long"

            col_best_gen, col_worst_gen = st.columns(2)
            
            with col_best_gen:
                st.success(f"Best Hedge for **Long 1x {trade_selection_gen}**")
                st.markdown(f"""
                    - **Hedge Instrument:** **{best_hedge_data_gen['Hedge Instrument']}**
                    - **Hedge Action:** {best_hedge_action_gen} **{abs(best_hedge_data_gen['Hedge Ratio (k*)']):.4f}** units.
                    - **Residual Volatility (Score):** **{best_hedge_data_gen['Residual Volatility (BPS)']:.2f} BPS** (Lowest Risk)
                """)
                
            with col_worst_gen:
                st.error(f"Worst Hedge for **Long 1x {trade_selection_gen}**")
                st.markdown(f"""
                    - **Hedge Instrument:** **{worst_hedge_data_gen['Hedge Instrument']}**
                    - **Hedge Action:** {worst_hedge_action_gen} **{abs(worst_hedge_data_gen['Hedge Ratio (k*)']):.4f}** units.
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

else:
    st.error("Generalized PCA failed. Please check your data quantity and quality.")

# --- NEW SECTION 8: ADVANCED SCENARIO HEDGING (PC1/PC2 NEUTRALITY) ---

if 'generalized_loadings' in locals() and not generalized_loadings.empty and pc_count >= 2:

    st.header("8. Advanced Scenario Hedging: PC1/PC2 Neutrality")
    st.markdown("Use two hedge instruments to perfectly neutralize the trade's exposure to **PC1 (Level)** and **PC2 (Slope)**, based on the **Generalized Loadings** (Factor Sensitivities).")
    
    # Re-use all derivatives for trade/hedge selection
    all_instruments_for_hedge = generalized_loadings.index.tolist()
    
    # 1. Trade Selection
    trade_selection_scenario = st.selectbox("1. Select Trade Instrument (T)", all_instruments_for_hedge, key='trade_scenario')
    
    # 2. Trade Direction
    trade_direction_scenario = st.radio("2. Trade Direction of T", ['Long 1 unit', 'Short 1 unit'], key='direction_scenario')
    is_long = (trade_direction_scenario == 'Long 1 unit')

    
    # 3. Hedge Candidate Selection (Must exclude the trade instrument)
    hedge_candidates_2x2 = [i for i in all_instruments_for_hedge if i != trade_selection_scenario]
    
    if len(hedge_candidates_2x2) < 2:
        st.warning("Not enough distinct instruments to form a 2-factor hedge (need at least two hedge candidates different from the trade).")
    else:
        col_h1, col_h2 = st.columns(2)
        
        # --- Default Selection Logic (Simplified) ---
        default_h1 = hedge_candidates_2x2[0]
        default_h2 = hedge_candidates_2x2[1]
        
        # Try to select the two nearest contracts to the trade in the list as better defaults
        try:
             default_index_trade = all_instruments_for_hedge.index(trade_selection_scenario)
             default_h1_index = (default_index_trade + 1)
             default_h2_index = (default_index_trade + 2)
             
             # Only use if they are in the candidates list
             if default_h1_index < len(all_instruments_for_hedge) and all_instruments_for_hedge[default_h1_index] in hedge_candidates_2x2:
                default_h1 = all_instruments_for_hedge[default_h1_index]
             
             if default_h2_index < len(all_instruments_for_hedge) and all_instruments_for_hedge[default_h2_index] in hedge_candidates_2x2 and all_instruments_for_hedge[default_h2_index] != default_h1:
                default_h2 = all_instruments_for_hedge[default_h2_index]
             elif default_h2 == default_h1 and len(hedge_candidates_2x2) > 1:
                # Fallback to the second candidate if defaults overlap
                default_h2 = [h for h in hedge_candidates_2x2 if h != default_h1][0] 
        except ValueError:
            # Trade instrument might not be found in the full list if filtering happened, fall back to simple list
            pass
        # --- End Default Selection Logic ---

        with col_h1:
            hedge1 = st.selectbox("3. Select Hedge Candidate 1 (H1)", hedge_candidates_2x2, 
                                index=hedge_candidates_2x2.index(default_h1) if default_h1 in hedge_candidates_2x2 else 0, 
                                key='hedge1_scenario')
        
        # Candidates for H2 must exclude H1
        candidates_h2 = [h for h in hedge_candidates_2x2 if h != hedge1]
        
        with col_h2:
            if candidates_h2:
                # Ensure default_h2 is still valid
                if default_h2 not in candidates_h2:
                    default_h2 = candidates_h2[0]
                    
                hedge2 = st.selectbox("4. Select Hedge Candidate 2 (H2)", candidates_h2, 
                                index=candidates_h2.index(default_h2) if default_h2 in candidates_h2 else 0, 
                                key='hedge2_scenario')
            else:
                st.warning("Not enough unique candidates left for Hedge 2.")
                st.stop()
        
        # --- Solve and Display Results ---
        h1, h2, residual_pc3_beta = solve_pc1_pc2_neutral_hedge(trade_selection_scenario, hedge1, hedge2, generalized_loadings)
        
        if h1 is not None:
            
            # Determine the required action based on trade direction and calculated ratio 
            # (h is for T - h*H = 0)
            
            if is_long:
                # If T is Long, we need to Short the calculated hedge ratios
                action_h1 = "Short" if h1 >= 0 else "Long"
                action_h2 = "Short" if h2 >= 0 else "Long"
                scenario_explanation = f"Since your trade is **Long 1 unit of {trade_selection_scenario}**, the required hedge weights (*h* = {h1:.4f}, {h2:.4f}) must be applied in the **opposite direction** to perfectly offset the PC1 and PC2 exposures. This creates a residual-PC-risk-only trade."
            else:
                # If T is Short, we need to Long the calculated hedge ratios 
                action_h1 = "Long" if h1 >= 0 else "Short"
                action_h2 = "Long" if h2 >= 0 else "Short"
                scenario_explanation = f"Since your trade is **Short 1 unit of {trade_selection_scenario}**, the required hedge weights (*h* = {h1:.4f}, {h2:.4f}) must be applied in the **same direction** to perfectly offset the negative PC1 and PC2 exposures. This creates a residual-PC-risk-only trade."


            st.success("#### PC1/PC2 Neutral Hedge Recommendation")
            st.markdown(f"##### Required Hedge Weights for **{trade_direction_scenario}**")
            
            st.dataframe(
                pd.DataFrame({
                    'Hedge Instrument': [hedge1, hedge2],
                    'Required Units': [f"{abs(h1):.4f}", f"{abs(h2):.4f}"],
                    'Action': [action_h1, action_h2]
                }, index=['H1', 'H2']),
                use_container_width=True
            )
            
            with st.expander("Explanation & Residual Risk"):
                st.markdown(f"**Scenario Hedge Explanation:**")
                st.markdown(scenario_explanation)
                
                st.markdown("This two-instrument hedge is designed to have a net sensitivity of zero to **PC1 (Level)** and **PC2 (Slope)**.")
                
                # Check if PC3 was included in the generalized loadings calculation (i.e., if pc_count >= 3)
                if 'PC3' in generalized_loadings.columns and residual_pc3_beta is not None:
                    st.markdown(f"**Residual Risk:** The resulting portfolio has residual exposure of **{residual_pc3_beta:.4f}** to PC3 (Curve/Butterfly) risk. This residual is the unhedged sensitivity to the standardized PC3 factor.")
                else:
                    st.info("Residual PC3 risk could not be calculated because the number of selected PCs (K) is less than 3.")

        
        else:
            st.error("Could not solve the 2-factor hedge system. This usually means the two selected hedge candidates are linearly dependent (too similar or identical). Please select two instruments with more distinct factor exposures.")

else:
    st.info("Requires at least 2 Principal Components (K >= 2) to calculate Level/Slope Neutrality. Please increase 'Number of Principal Components' in the sidebar.")
