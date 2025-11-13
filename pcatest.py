import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression 
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

# --- NEW: Double Butterfly Calculation Function ---
@st.cache_data
def calculate_k_step_double_butterflies(analysis_curve_df, k):
    """
    Calculates double butterflies using contracts separated by 'k' steps (e.g., k=1 for 3M DBF).
    Formula: C_i - 3 * C_{i+k} + 3 * C_{i+2k} - C_{i+3k}
    Label Format: C_i-3xC_{i+k}+3xC_{i+2k}-C_{i+3k}
    """
    # Need 4 contracts: C_i, C_{i+k}, C_{i+2k}, C_{i+3k}
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < 3 * k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    dbflies_data = {}

    for i in range(num_contracts - 3 * k):
        c1_maturity = analysis_curve_df.columns[i]          # C_i
        c2_maturity = analysis_curve_df.columns[i+k]        # C_{i+k}
        c3_maturity = analysis_curve_df.columns[i+2*k]      # C_{i+2k}
        c4_maturity = analysis_curve_df.columns[i+3*k]      # C_{i+3k}

        # DBF = C_i - 3*C_{i+k} + 3*C_{i+2k} - C_{i+3k}
        dbfly_label = f"{c1_maturity}-3x{c2_maturity}+3x{c3_maturity}-{c4_maturity}"

        dbflies_data[dbfly_label] = (
            analysis_curve_df.iloc[:, i] 
            - 3 * analysis_curve_df.iloc[:, i+k] 
            + 3 * analysis_curve_df.iloc[:, i+2*k] 
            - analysis_curve_df.iloc[:, i+3*k]
        )

    return pd.DataFrame(dbflies_data)

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

# --- MODIFIED: _reconstruct_derivative now handles 'dbfly' ---
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
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                c1, c_long = core_label.split('-')
                
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - reconstructed_prices_aligned[c_long + ' (PCA)']
                )
            
            elif derivative_type == 'fly':
                # Fly: C_i - 2 * C_{i+k} + C_{i+2k}. Label format: C_i-2xC_{i+k}+C_{i+2k}
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                parts = core_label.split('-', 1) 
                c1 = parts[0] 
                sub_parts = parts[1].split('+')
                c2_label = sub_parts[0].split('x')[1] 
                c3_label = sub_parts[1] 
                
                # Reconstruct the derivative
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - 
                    2 * reconstructed_prices_aligned[c2_label + ' (PCA)'] + 
                    reconstructed_prices_aligned[c3_label + ' (PCA)']
                )
            
            elif derivative_type == 'dbfly':
                # Double Fly: C_i - 3 * C_{i+k} + 3 * C_{i+2k} - C_{i+3k}. Label format: C_i-3xC_{i+k}+3xC_{i+2k}-C_{i+3k}
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                parts = core_label.split('-', 1) 
                c1 = parts[0] # C_i
                
                sub_parts_1 = parts[1].split('+')
                
                c2_label = sub_parts_1[0].split('x')[1] # C_{i+k} from '3xC_{i+k}'
                
                sub_parts_2 = sub_parts_1[1].split('-')
                
                c3_label = sub_parts_2[0].split('x')[1] # C_{i+2k} from '3xC_{i+2k}'
                c4_label = sub_parts_2[1] # C_{i+3k}
                
                # Reconstruct the derivative
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - 
                    3 * reconstructed_prices_aligned[c2_label + ' (PCA)'] + 
                    3 * reconstructed_prices_aligned[c3_label + ' (PCA)'] -
                    reconstructed_prices_aligned[c4_label + ' (PCA)']
                )
            
        except Exception as e:
             # Skip if reconstruction fails due to malformed label or missing price
             continue 
    
    reconstructed_df = pd.DataFrame(reconstructed_data, index=reconstructed_prices_aligned.index)
    
    original_rename = {col: col + ' (Original)' for col in original_df_aligned.columns}
    original_df_renamed = original_df_aligned.rename(columns=original_rename)
    
    return pd.merge(original_df_renamed, reconstructed_df, left_index=True, right_index=True)


# --- MODIFIED: reconstruct_prices_and_derivatives now handles Double Butterflies ---
def reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_3M_df, spreads_3M_df, spreads_6M_df, butterflies_3M_df, butterflies_6M_df, spreads_12M_df, butterflies_12M_df, double_butterflies_3M_df, double_butterflies_6M_df, double_butterflies_12M_df):
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
    
    spreads_3M_df_no_prefix = spreads_3M_df.copy()

    # Reconstruct all subsequent contracts using the reconstructed 3M spreads (k=1)
    for i in range(1, len(analysis_curve_df_aligned.columns)):
        prev_maturity = analysis_curve_df_aligned.columns[i-1]
        current_maturity = analysis_curve_df_aligned.columns[i]
        spread_label_no_prefix = f"{prev_maturity}-{current_maturity}" # This is always the 3M spread label

        spread_label_reconstructed = f"3M Spread: {spread_label_no_prefix}"
        
        if spread_label_no_prefix in reconstructed_spreads_3M_df.columns:
            # P_i = P_i-1 (PCA) - S_i-1,i (PCA)
            reconstructed_prices_df[current_maturity + ' (PCA)'] = (
                reconstructed_prices_df[prev_maturity + ' (PCA)'] - reconstructed_spreads_3M_df[spread_label_no_prefix]
            )
        else:
            # Fallback if the 3M spread is missing for that contract roll
             reconstructed_prices_df[current_maturity + ' (PCA)'] = reconstructed_prices_df[prev_maturity + ' (PCA)']

    original_price_rename = {col: col + ' (Original)' for col in analysis_curve_df_aligned.columns}
    original_prices_df = analysis_curve_df_aligned.rename(columns=original_price_rename)
    historical_outrights = pd.merge(original_prices_df, reconstructed_prices_df, left_index=True, right_index=True)


    # --- 2. Reconstruct Derivatives from Reconstructed Prices ---
    
    # Prepare derivative DFs with prefixes for _reconstruct_derivative to correctly rename columns
    spreads_3M_df_prefixed = spreads_3M_df_no_prefix.rename(columns=lambda x: f"3M Spread: {x}")
    butterflies_3M_df_prefixed = butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}")
    spreads_6M_df_prefixed = spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}")
    butterflies_6M_df_prefixed = butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}")
    spreads_12M_df_prefixed = spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}")
    butterflies_12M_df_prefixed = butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}")
    
    # New Double Butterfly DFs
    double_butterflies_3M_df_prefixed = double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}")
    double_butterflies_6M_df_prefixed = double_butterflies_6M_df.rename(columns=lambda x: f"6M Double Fly: {x}")
    double_butterflies_12M_df_prefixed = double_butterflies_12M_df.rename(columns=lambda x: f"12M Double Fly: {x}")

    historical_spreads_3M = _reconstruct_derivative(spreads_3M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_3M = _reconstruct_derivative(butterflies_3M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_6M = _reconstruct_derivative(spreads_6M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_6M = _reconstruct_derivative(butterflies_6M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_12M = _reconstruct_derivative(spreads_12M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_12M = _reconstruct_derivative(butterflies_12M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    # New Double Butterfly reconstructions
    historical_double_butterflies_3M = _reconstruct_derivative(double_butterflies_3M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')
    historical_double_butterflies_6M = _reconstruct_derivative(double_butterflies_6M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')
    historical_double_butterflies_12M = _reconstruct_derivative(double_butterflies_12M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')

    # MODIFIED: Return the new historical double butterfly DFs
    return historical_outrights, historical_spreads_3M, historical_butterflies_3M, historical_spreads_6M, historical_butterflies_6M, historical_spreads_12M, historical_butterflies_12M, historical_double_butterflies_3M, historical_double_butterflies_6M, historical_double_butterflies_12M, spreads_3M_df_no_prefix


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


# --- GENERALIZED HEDGING LOGIC (Section 7) ---

def calculate_derivatives_covariance_generalized(all_derivatives_df, scores_df, eigenvalues, pc_count):
    """
    Calculates the Raw Covariance Matrix for ALL derivatives (Spreads, Flies, Double Flies) 
    by projecting their standardized time series onto the standardized 3M Spread PC scores.
    Returns the Raw Covariance Matrix, the aligned derivatives data, and the standardized loadings (L_D).
    """
    # 1. Align and clean data - ensure all derivatives are aligned with the PC scores index
    aligned_index = all_derivatives_df.index.intersection(scores_df.index)
    derivatives_aligned = all_derivatives_df.loc[aligned_index].dropna(axis=1)
    scores_aligned = scores_df.loc[aligned_index]
    
    if derivatives_aligned.empty:
        # Return empty dataframes, but return all three expected variables
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

def calculate_factor_sensitivities(loadings_df_gen, pc_count):
    """
    Calculates the Standardized Sensitivity (Beta) of every derivative to the first three 
    principal components (Level, Slope, Curvature).
    """
    if loadings_df_gen.empty:
        return pd.DataFrame()

    # Define the factor mapping based on the first 3 PCs
    pc_map = {
        'PC1': 'Level (Whole Curve Shift)', 
        'PC2': 'Slope (Steepening/Flattening)', 
        'PC3': 'Curvature (Fly Risk)'
    }
    
    # Only use up to the number of available PCs, or 3, whichever is smaller
    available_pcs = loadings_df_gen.columns.intersection(list(pc_map.keys()))
    
    # Filter the generalized loadings L_D for the relevant PCs
    factor_sensitivities = loadings_df_gen.filter(items=available_pcs.tolist(), axis=1).copy()
    
    # Rename columns for clarity in the output
    factor_sensitivities.columns = [pc_map[col] for col in available_pcs]
    
    return factor_sensitivities

def calculate_all_factor_hedges(trade_label, factor_name, factor_sensitivities_df, Sigma_Raw_df):
    """ 
    Calculates the Factor Hedge Ratio and the resulting Residual Volatility for all potential hedge instruments, for a specified factor. 
    NOTE: 'Trade Sensitivity' is included in the resulting DataFrame.
    """
    if trade_label not in factor_sensitivities_df.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not found in sensitivities."
    if factor_name not in factor_sensitivities_df.columns:
        return pd.DataFrame(), f"Factor '{factor_name}' not found."
    if trade_label not in Sigma_Raw_df.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not found in covariance matrix."

    results = []
    Trade_Exposure = factor_sensitivities_df.loc[trade_label, factor_name]

    Var_Trade = Sigma_Raw_df.loc[trade_label, trade_label] # Var(T)

    # Iterate through all other derivatives as potential hedges
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label]

    for hedge_instrument in potential_hedges:
        try:
            Hedge_Exposure = factor_sensitivities_df.loc[hedge_instrument, factor_name]
            Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument] # Var(H)
            Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument] # Cov(T, H)

            # 1. Calculate Factor Hedge Ratio (k_factor)
            if abs(Hedge_Exposure) < 1e-9:
                k_factor = 0.0
                Residual_Volatility_BPS = np.nan # Cannot neutralize factor with zero-exposure hedge
            else:
                # k_factor is the ratio of sensitivities: k = Beta_T / Beta_H
                k_factor = Trade_Exposure / Hedge_Exposure

                # 2. Calculate Residual Variance of the hedged portfolio (Var(T - k*H))
                # Var(P) = Var(T) + k^2 Var(H) - 2k Cov(T, H)
                Residual_Variance = Var_Trade + (k_factor**2 * Var_Hedge) - (2 * k_factor * Cov_TH)
                Residual_Variance = max(0, Residual_Variance)

                # 3. Residual Volatility (Score) in BPS
                Residual_Volatility_BPS = np.sqrt(Residual_Variance) * 10000

            results.append({
                'Hedge Instrument': hedge_instrument,
                'Trade Sensitivity': Trade_Exposure, # Included the Trade Sensitivity
                'Hedge Sensitivity': Hedge_Exposure,
                f'Factor Hedge Ratio (k_factor)': k_factor,
                'Residual Volatility (BPS)': Residual_Volatility_BPS
            })
        except Exception as e:
            continue

    if not results:
        return pd.DataFrame(), "No valid hedge candidates found."

    results_df = pd.DataFrame(results)
    # Sort by Residual Volatility (BPS) to show the most effective hedges first
    results_df = results_df.sort_values(by='Residual Volatility (BPS)', ascending=True, na_position='last')

    return results_df, None
    
# --- NEW HELPER FUNCTION TO CALCULATE MISPRICING FOR SECTION 8 DISPLAY ---
def calculate_single_mispricing(instrument_label, historical_dfs_dict, analysis_date):
    """
    Calculates the mispricing (Original - PCA) for a single instrument on the analysis date.
    Instrument label includes prefix, e.g., '3M Spread: Z25-M26'.
    Returns mispricing in BPS.
    """
    try:
        # Determine derivative type (prefix)
        parts = instrument_label.split(': ', 1)
        if len(parts) != 2:
            return np.nan # Not a standard derivative label

        derivative_type = parts[0]

        if derivative_type not in historical_dfs_dict:
            return np.nan # Derivative type not found (e.g., Outright is not included here)

        history_df = historical_dfs_dict[derivative_type]

        # Construct the column names in the history DataFrame
        original_col = instrument_label + ' (Original)'
        pca_col = instrument_label + ' (PCA)'

        if original_col not in history_df.columns or pca_col not in history_df.columns:
            return np.nan # Columns not found

        # Align the analysis_date (which is a date object) with the DateTimeIndex
        target_dates = history_df.index[history_df.index.date == analysis_date]
        if target_dates.empty:
            return np.nan # Date not found

        # Use the latest timestamp on that date
        latest_date = target_dates.max() 
        
        original_val = history_df.loc[latest_date, original_col]
        pca_val = history_df.loc[latest_date, pca_col]
        
        mispricing = (original_val - pca_val) * 10000 # In BPS

        return mispricing
        
    except Exception as e:
        return np.nan


# --- Streamlit Application Layout ---
st.title("SOFR Futures PCA Analyzer")

# --- Sidebar Inputs ---
st.sidebar.header("1. Data Uploads")
price_file = st.sidebar.file_uploader(
    "Upload Historical Price Data (e.g., 'sofr rates.csv')", type=['csv'], key='price_upload'
)
expiry_file = st.sidebar.file_uploader(
    "Upload Contract Expiry Dates (e.g., 'EXPIRY (2).csv')", type=['csv'], key='expiry_upload'
)

# Initialize dataframes
price_df = load_data(price_file)
expiry_df = load_data(expiry_file)

# Placeholder variables (needed for the rest of the script to run)
loadings_df_gen = pd.DataFrame()
Sigma_Raw_df = pd.DataFrame()
spreads_3M_df_no_prefix = pd.DataFrame()

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
        max_value=max_date
    )
    
    # --- Data Processing and Filtering ---
    future_expiries_df = get_analysis_contracts(expiry_df, analysis_date)
    analysis_curve_df, valid_contracts = transform_to_analysis_curve(price_df_filtered, future_expiries_df)

    if analysis_curve_df.empty:
        st.warning("Curve could not be constructed for the selected date and range.")
    else:
        # --- 4. Derivative Calculations ---
        spreads_3M_df = calculate_k_step_spreads(analysis_curve_df, k=1)
        butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, k=1)
        spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, k=2)
        butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, k=2)
        spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, k=4)
        butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, k=4)
        
        double_butterflies_3M_df = calculate_k_step_double_butterflies(analysis_curve_df, k=1)
        double_butterflies_6M_df = calculate_k_step_double_butterflies(analysis_curve_df, k=2)
        double_butterflies_12M_df = calculate_k_step_double_butterflies(analysis_curve_df, k=4)

        all_derivatives_df = pd.concat([
            spreads_3M_df.rename(columns=lambda x: f"3M Spread: {x}"),
            butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}"),
            spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}"),
            butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}"),
            spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}"),
            butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}"),
            double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}"),
            double_butterflies_6M_df.rename(columns=lambda x: f"6M Double Fly: {x}"),
            double_butterflies_12M_df.rename(columns=lambda x: f"12M Double Fly: {x}"),
        ], axis=1)

        # --- 5. PCA on 3M Spreads (Fair Curve) ---
        loadings_3M, explained_variance_3M, eigenvalues_3M, scores_3M, spreads_3M_df_clean = perform_pca(spreads_3M_df)
        
        if loadings_3M is not None:
            # --- 6. Curve Reconstruction ---
            # Reconstruct the 3M spreads using all PCs (n_components)
            n_components_3M = loadings_3M.shape[1]
            reconstructed_spreads_3M_df = (
                scores_3M @ loadings_3M.T * spreads_3M_df_clean.std() + spreads_3M_df_clean.mean()
            )
            reconstructed_spreads_3M_df.columns = spreads_3M_df_clean.columns

            historical_outrights, historical_spreads_3M, historical_butterflies_3M, historical_spreads_6M, historical_butterflies_6M, historical_spreads_12M, historical_butterflies_12M, historical_double_butterflies_3M, historical_double_butterflies_6M, historical_double_butterflies_12M, spreads_3M_df_no_prefix = reconstruct_prices_and_derivatives(
                analysis_curve_df, reconstructed_spreads_3M_df, spreads_3M_df, spreads_6M_df, butterflies_3M_df, butterflies_6M_df, spreads_12M_df, butterflies_12M_df, double_butterflies_3M_df, double_butterflies_6M_df, double_butterflies_12M_df
            )
            
            # --- MISPRICING SETUP (Dictionary for Mispricing calculation) ---
            historical_dfs_dict = {
                'Outright': historical_outrights,
                '3M Spread': historical_spreads_3M,
                '3M Fly': historical_butterflies_3M,
                '6M Spread': historical_spreads_6M,
                '6M Fly': historical_butterflies_6M,
                '12M Spread': historical_spreads_12M,
                '12M Fly': historical_butterflies_12M,
                '3M Double Fly': historical_double_butterflies_3M,
                '6M Double Fly': historical_double_butterflies_6M,
                '12M Double Fly': historical_double_butterflies_12M
            }

            # --- 7. Generalized PCA (Loadings L_D and Raw Covariance Sigma_Raw) ---
            if scores_3M is not None and not all_derivatives_df.empty and eigenvalues_3M is not None:
                # Use enough PCs to capture most risk (e.g., first 5 or n_components)
                pc_count_gen = min(scores_3M.shape[1], 5) 
                
                Sigma_Raw_df, derivatives_aligned, loadings_df_gen = calculate_derivatives_covariance_generalized(
                    all_derivatives_df, scores_3M, eigenvalues_3M, pc_count_gen
                )
            
                # --- 8. Factor-Based Hedging (Trade Sensitivity & Mispricing Table) ---
                st.markdown("## 8. Factor-Based Hedging (Level, Slope, Curvature)")

                if loadings_df_gen.empty:
                    st.error("Generalized PCA failed. Cannot calculate factor hedges.")
                else:
                    factor_sensitivities_df = calculate_factor_sensitivities(loadings_df_gen, pc_count_gen)
                    available_factors = factor_sensitivities_df.columns.tolist()
                    available_trades = factor_sensitivities_df.index.tolist()
                    
                    st.sidebar.markdown("---")
                    st.sidebar.header("8. Factor Hedge Inputs")
                    factor_name = st.sidebar.selectbox(
                        "Select Factor to Neutralize", available_factors, key='hedge_factor'
                    )
                    trade_label = st.sidebar.selectbox(
                        "Select Trade Instrument", available_trades, key='trade_instrument_factor'
                    )
                    
                    if st.button("Calculate Factor Hedges"):
                        if not factor_name or not trade_label:
                            st.warning("Please select a factor and a trade instrument.")
                            st.stop()
                            
                        # Calculate the hedges
                        factor_results_df, error = calculate_all_factor_hedges(
                            trade_label, factor_name, factor_sensitivities_df, Sigma_Raw_df
                        )
                        
                        st.subheader(f"Factor-Neutral Hedge for Trade: {trade_label} (Neutralizing {factor_name})")
                        
                        if not factor_results_df.empty and error is None:
                            
                            # NEW STEP: Calculate Mispricing for each Hedge Instrument
                            mispricing_list = []
                            for index, row in factor_results_df.iterrows():
                                hedge_instrument_label = row['Hedge Instrument']
                                mispricing = calculate_single_mispricing(hedge_instrument_label, historical_dfs_dict, analysis_date)
                                mispricing_list.append(mispricing)
                                
                            factor_results_df['Hedge Mispricing (BPS)'] = mispricing_list
                            
                            # --- MODIFIED: Prepare table for display to include 'Trade Sensitivity' and 'Hedge Mispricing (BPS)'
                            display_df = factor_results_df.rename(columns={
                                f'Factor Hedge Ratio (k_factor)': 'Hedge Ratio (k_factor)',
                                'Residual Volatility (BPS)': 'Residual Volatility (BPS)'
                            })[[
                                'Hedge Instrument', 
                                'Trade Sensitivity',          # <-- NEW: Included Trade Sensitivity
                                'Hedge Mispricing (BPS)',     # <-- NEW: Included Mispricing
                                'Hedge Ratio (k_factor)', 
                                'Residual Volatility (BPS)', 
                                'Hedge Sensitivity'
                            ]]
                            
                            # Format and display
                            st.dataframe(
                                display_df.style.format({
                                    'Trade Sensitivity': "{:.4f}",       
                                    'Hedge Mispricing (BPS)': "{:.2f}",  
                                    'Hedge Ratio (k_factor)': "{:.4f}",
                                    'Residual Volatility (BPS)': "{:.2f}",
                                    'Hedge Sensitivity': "{:.4f}",
                                }),
                                use_container_width=True
                            )
                            
                            st.info(f"Mispricing is calculated as (Original - PCA) for the Hedge Instrument on {analysis_date.strftime('%Y-%m-%d')} (in BPS).")
                            
                        else:
                            st.info("No hedge candidates could be successfully processed. Check if any instrument has non-zero sensitivity to the selected factor.")
                            
                        # Display full sensitivities table as before for reference
                        st.markdown("---")
                        st.subheader(f"Factor Sensitivities (Standardized Beta) Table for Reference")
                        st.markdown("This shows the raw input exposures used for the ratio calculation.")
                        
                        st.dataframe(
                            factor_sensitivities_df.style.format("{:.4f}"),
                            use_container_width=True
                        )
        else:
            st.error("PCA failed. Please check your data quantity and quality.")
