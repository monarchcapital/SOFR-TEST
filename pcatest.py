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


# --- ORIGINAL HEDGING LOGIC (Section 6) ---

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


# --- NEW GENERALIZED HEDGING LOGIC (Section 7 and 8) ---

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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Modified return
        
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
        # Get Betas for PC1 and PC2 for all three instruments
        # Loadings here are the Betas of the instrument's standardized P&L on the standardized PC scores
        beta_t = generalized_loadings_df.loc[trade_instrument, ['PC1', 'PC2']].values
        beta_h1 = generalized_loadings_df.loc[hedge1, ['PC1', 'PC2']].values
        beta_h2 = generalized_loadings_df.loc[hedge2, ['PC1', 'PC2']].values

        # A is the matrix of hedge betas (2x2)
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
            # Residual is the total PC3 exposure left in the portfolio
            residual_pc3 = beta_t_pc3 - h1 * beta_h1_pc3 - h2 * beta_h2_pc3
            
        # The result is: T - h1*H1 - h2*H2 = Residual

        return h1, h2, residual_pc3
    except np.linalg.LinAlgError:
        # Singular matrix (hedges are too correlated/identical)
        return None, None, None 
    except KeyError as e:
        # PC1 or PC2 not available (pc_count < 2) or trade/hedge not in loadings
        print(f"KeyError during PC1/PC2 hedge: {e}")
        return None, None, None


# --- Streamlit Application Layout ---

# [ ... Sidebar and UI code from pcatest (10).py ... ]
if 'price_df' not in locals():
    # Placeholder for loading the data, assume it's loaded higher up
    st.title("SOFR Futures PCA Analyzer")
    st.info("Please upload both the Price Data and Expiry Data CSV files to begin the analysis.")
    st.stop()


# NOTE: Assuming the rest of the application's top-level code (including data loading and PCA execution) 
# remains the same as pcatest (10).py until the hedging sections.
# The code below should be placed AFTER the entire Section 7.

if 'Sigma_Raw_df' not in locals():
    # If the previous parts of the application failed, stop here
    # This is a safety check for a complete application structure
    # st.error("Core PCA and Generalized Hedging setup failed.")
    pass # Assume variables are set up if the user runs the whole script


# --- NEW SECTION 8: ADVANCED SCENARIO HEDGING (PC1/PC2 NEUTRALITY) ---

st.header("8. Advanced Scenario Hedging: PC1/PC2 Neutrality")
st.markdown("Use two hedge instruments to perfectly neutralize the trade's exposure to **PC1 (Level)** and **PC2 (Slope)**, based on the **Generalized Loadings** (Factor Sensitivities).")

# The variable 'generalized_loadings' is now available because calculate_derivatives_covariance_generalized was modified.
if 'generalized_loadings' in locals() and not generalized_loadings.empty and pc_count >= 2:
    
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
        if len(hedge_candidates_2x2) > 2:
             # Try to select the two nearest contracts to the trade in the list as better defaults
             try:
                 default_index_trade = all_instruments_for_hedge.index(trade_selection_scenario)
                 default_h1_index = (default_index_trade + 1)
                 default_h2_index = (default_index_trade + 2)
                 
                 # Only use if they are in the candidates list
                 if default_h1_index < len(all_instruments_for_hedge) and all_instruments_for_hedge[default_h1_index] in hedge_candidates_2x2:
                    default_h1 = all_instruments_for_hedge[default_h1_index]
                 
                 if default_h2_index < len(all_instruments_for_hedge) and all_instruments_for_hedge[default_h2_index] in hedge_candidates_2x2:
                    default_h2 = all_instruments_for_hedge[default_h2_index]
                 
                 if default_h1 == default_h2:
                    # Fallback if both defaults point to the same item (shouldn't happen with +1, +2 but safe check)
                    default_h2 = hedge_candidates_2x2[1]

             except ValueError:
                 # Trade instrument might not be found in the full list if filtering happened, fall back to simple list
                 pass
        # --- End Default Selection Logic ---

        with col_h1:
            hedge1 = st.selectbox("3. Select Hedge Candidate 1 (H1)", hedge_candidates_2x2, index=hedge_candidates_2x2.index(default_h1) if default_h1 in hedge_candidates_2x2 else 0, key='hedge1_scenario')
        
        # Candidates for H2 must exclude H1
        candidates_h2 = [h for h in hedge_candidates_2x2 if h != hedge1]
        
        with col_h2:
            if candidates_h2:
                # Ensure default_h2 is still valid
                if default_h2 not in candidates_h2:
                    default_h2 = candidates_h2[0]
                    
                hedge2 = st.selectbox("4. Select Hedge Candidate 2 (H2)", candidates_h2, index=candidates_h2.index(default_h2) if default_h2 in candidates_h2 else 0, key='hedge2_scenario')
            else:
                st.warning("Not enough unique candidates left for Hedge 2.")
                st.stop()
        
        # --- Solve and Display Results ---
        h1, h2, residual_pc3_beta = solve_pc1_pc2_neutral_hedge(trade_selection_scenario, hedge1, hedge2, generalized_loadings)
        
        if h1 is not None:
            
            # Determine the required action based on trade direction and calculated ratio (h is for T - h*H = 0)
            if is_long:
                action_h1 = "Short" if h1 > 0 else "Long"
                action_h2 = "Short" if h2 > 0 else "Long"
                scenario_explanation = f"Since your trade is **Long 1 unit of {trade_selection_scenario}**, the calculated hedge weights (*h* = {h1:.4f}, {h2:.4f}) must be applied in the **opposite direction** to perfectly offset the PC1 and PC2 exposures. This is a **PC-Neutral Basis Trade**."
            else:
                action_h1 = "Long" if h1 > 0 else "Short"
                action_h2 = "Long" if h2 > 0 else "Short"
                scenario_explanation = f"Since your trade is **Short 1 unit of {trade_selection_scenario}**, the calculated hedge weights (*h* = {h1:.4f}, {h2:.4f}) must be applied in the **same direction** to perfectly offset the negative PC1 and PC2 exposures. This is a **PC-Neutral Basis Trade**."


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
                if 'PC3' in generalized_loadings.columns:
                    st.markdown(f"**Residual Risk:** The resulting portfolio has residual exposure of **{residual_pc3_beta:.4f}** to PC3 (Curve/Butterfly) risk. This residual is the unhedged sensitivity to the standardized PC3 factor.")
                else:
                    st.info("Residual PC3 risk could not be calculated because the number of selected PCs (K) is less than 3.")

        
        else:
            st.error("Could not solve the 2-factor hedge system. This usually means the two selected hedge candidates are linearly dependent (too similar or identical). Please select two instruments with more distinct factor exposures.")

else:
    st.info("Requires at least 2 Principal Components (K >= 2) to calculate Level/Slope Neutrality. Please increase 'Number of Principal Components' in the sidebar.")
