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


def reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_3M_df, spreads_3M_df, spreads_6M_df, butterflies_3M_df, butterflies_6M_df, spreads_12M_df, butterflies_12M_df, double_butterflies_3M_df, double_butterflies_6M_df, double_butterflies_12M_df):
    """
    Reconstructs Outright Prices and all derivative types based on the 
    reconstructed 3M spreads (PCA result) and the original nearest contract price anchor.
    """
    # Filter the analysis_curve_df to match the index of the reconstructed 3M spreads
    reconstructed_spreads_3M_df_no_prefix = reconstructed_spreads_3M_df.rename(columns=lambda x: x.replace("3M Spread: ", ""))
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

        if spread_label_no_prefix in reconstructed_spreads_3M_df_no_prefix.columns:
            # P_i = P_i-1 (PCA) - S_i-1,i (PCA)
            reconstructed_prices_df[current_maturity + ' (PCA)'] = (
                reconstructed_prices_df[prev_maturity + ' (PCA)'] - reconstructed_spreads_3M_df_no_prefix[spread_label_no_prefix]
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

    # Return the new historical double butterfly DFs
    return historical_outrights, historical_spreads_3M, historical_butterflies_3M, historical_spreads_6M, historical_butterflies_6M, historical_spreads_12M, historical_butterflies_12M, historical_double_butterflies_3M, historical_double_butterflies_6M, historical_double_butterflies_12M, spreads_3M_df_no_prefix


# --- ORIGINAL HEDGING LOGIC (Section 7) ---

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
    and returns the full results DataFrame as well. (Section 7 - 3M Spreads only)
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


# --- GENERALIZED HEDGING LOGIC (Section 8) ---

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
    (Section 8 - All Derivatives)
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

# --- FACTOR-BASED HEDGING LOGIC (Section 11) ---

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
    Calculates the Factor Hedge Ratio and the resulting Residual Volatility for all potential 
    hedge instruments, for a specified factor.
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
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label and col in factor_sensitivities_df.index]

    for hedge_instrument in potential_hedges:
        try:
            Hedge_Exposure = factor_sensitivities_df.loc[hedge_instrument, factor_name]
            Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument] # Var(H)
            Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument]        # Cov(T, H)

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
                'Trade Sensitivity': Trade_Exposure,
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

# --- NEW UTILITY FUNCTION FOR HEDGING WITH NEW INSTRUMENTS (Outrights) ---

def calculate_loadings_for_new_instruments(new_instrument_df, scores_df, pc_count):
    """
    Calculates the Standardized Loadings (Beta) of a set of new instruments 
    (e.g., outright prices) onto the existing standardized PC scores.
    """
    # 1. Align and clean data
    aligned_index = new_instrument_df.index.intersection(scores_df.index)
    instruments_aligned = new_instrument_df.loc[aligned_index].dropna(axis=1)
    scores_aligned = scores_df.loc[aligned_index]
    
    if instruments_aligned.empty:
        return pd.DataFrame(), pd.DataFrame() 
        
    # 2. Standardize instruments
    instruments_std = instruments_aligned.std()
    instruments_scaled = (instruments_aligned - instruments_aligned.mean()) / instruments_std
    
    # 3. Calculate Loadings (Beta)
    loadings_data = {}
    X = scores_aligned.iloc[:, :pc_count].values # Standardized PC scores
    
    for col in instruments_scaled.columns:
        y = instruments_scaled[col].values
        reg = LinearRegression(fit_intercept=False) 
        reg.fit(X, y)
        loadings_data[col] = reg.coef_

    # L_new: Loadings of the new instruments onto the PC space
    loadings_df = pd.DataFrame(
        loadings_data, 
        index=[f'PC{i+1}' for i in range(pc_count)]
    ).T
    
    return loadings_df, instruments_aligned.std()


def calculate_combined_sensitivities(loadings_df_gen, outright_loadings_df):
    """
    Combines standardized loadings (sensitivities) of all derivatives and all outrights.
    Used for factor hedging between any two instruments.
    """
    if loadings_df_gen.empty and outright_loadings_df.empty:
        return pd.DataFrame()
        
    # If one is empty, return the other. Pad with NaN for missing PCs if needed.
    if loadings_df_gen.empty:
        return outright_loadings_df
    if outright_loadings_df.empty:
        return loadings_df_gen

    # Ensure consistent set of PCs (columns)
    all_pcs = sorted(list(set(loadings_df_gen.columns).union(set(outright_loadings_df.columns))))
    
    # Reindex/realign before concatenation
    loadings_df_gen_aligned = loadings_df_gen.reindex(columns=all_pcs, fill_value=0)
    outright_loadings_df_aligned = outright_loadings_df.reindex(columns=all_pcs, fill_value=0)

    combined_sensitivities = pd.concat([
        loadings_df_gen_aligned, 
        outright_loadings_df_aligned
    ], axis=0)
    
    return combined_sensitivities


# --- NEW FUNCTION 1: Historical Tracking (For Section 5) ---

def plot_historical_tracking(historical_df, derivative_label):
    """
    Plots the historical Original, PCA Fair, and Mispricing for a selected derivative.
    """
    if historical_df.empty:
        st.warning("Historical data is empty. Cannot plot.")
        return

    original_col = derivative_label + ' (Original)'
    pca_col = derivative_label + ' (PCA)'
    
    if original_col not in historical_df.columns or pca_col not in historical_df.columns:
        st.warning(f"Columns for '{derivative_label}' not found in historical data.")
        return

    plot_df = historical_df[[original_col, pca_col]].copy()
    plot_df['Mispricing (Original - PCA Fair)'] = plot_df[original_col] - plot_df[pca_col]
    
    # Convert to BPS for better visualization
    plot_df = plot_df * 10000 
    
    # Rename columns for chart legend
    plot_df.columns = ['Market Price (BPS)', 'PCA Fair Value (BPS)', 'Mispricing (BPS)']
    
    st.subheader(f"Historical Mispricing for: {derivative_label}")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### Price vs. Fair Value (BPS)")
        st.line_chart(plot_df[['Market Price (BPS)', 'PCA Fair Value (BPS)']], use_container_width=True)
    
    with col2:
        st.markdown("##### Mispricing Time Series (BPS)")
        st.line_chart(plot_df['Mispricing (BPS)'], use_container_width=True)


# --- NEW FUNCTION 2: Variance Contribution (For Section 9) ---

def calculate_variance_attribution(trade_label, loadings_df_gen, eigenvalues, combined_std_dev, pc_count):
    """
    Calculates the percentage contribution of each PC to the trade's total explained variance.
    Var(T)_explained = sum_{i=1}^{N} (beta_{T,i}^2 * lambda_i) * sigma_T^2
    """
    # Use the full loadings_df_gen (derivatives + outrights) for trade look-up
    if trade_label not in loadings_df_gen.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not found in sensitivities."
        
    # Standard deviation used to scale standardized variance back to raw variance
    try:
        trade_std = combined_std_dev.loc[trade_label] # sigma_T
    except KeyError:
        return pd.DataFrame(), "Internal Error: Trade standard deviation not found."


    trade_loadings = loadings_df_gen.loc[trade_label].iloc[:pc_count] # beta_T,i
    
    # Variance of the selected PCs (lambda_i)
    lambda_p = pd.Series(eigenvalues[:pc_count], index=[f'PC{i+1}' for i in range(pc_count)])
    
    # 1. Calculate contribution to the STANDARDIZED variance (beta^2 * lambda)
    standardized_contribution = (trade_loadings ** 2) * lambda_p
    
    # 2. Scale up to the RAW explained variance (by multiplying by sigma_T^2)
    raw_explained_variance = standardized_contribution * (trade_std ** 2)
    
    # 3. Calculate the total explained variance (sum of raw_explained_variance)
    total_explained_variance = raw_explained_variance.sum()
    
    if total_explained_variance < 1e-12: # Check for near zero
        return pd.DataFrame(), "Total explained variance is near zero or missing data."
    
    # 4. Calculate percentage contribution
    percentage_contribution = (raw_explained_variance / total_explained_variance) * 100
    
    attribution_df = pd.DataFrame({
        'Component': percentage_contribution.index,
        'Raw Var. Contribution': raw_explained_variance.values,
        'Percentage Contribution (%)': percentage_contribution.values
    }).set_index('Component')
    
    return attribution_df.sort_values(by='Percentage Contribution (%)', ascending=False), None


# --- Streamlit Application Layout ---

st.title("SOFR Futures PCA Analyzer")

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

# Placeholder for L_D Loadings and Sigma_Raw_df, calculated in Section 8 and used in Section 9+
loadings_df_gen = pd.DataFrame()
Sigma_Raw_df = pd.DataFrame()
spreads_3M_df_no_prefix = pd.DataFrame() 
all_derivatives_aligned = pd.DataFrame() 
outright_std_dev = pd.DataFrame()
combined_sensitivities = pd.DataFrame()
combined_std_dev = pd.Series(dtype=float)

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
    
    # 3M (k=1) - Used for PCA input
    spreads_3M_df_raw = calculate_k_step_spreads(analysis_curve_df, 1) # No prefix here
    butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, 1)
    double_butterflies_3M_df = calculate_k_step_double_butterflies(analysis_curve_df, 1) 
    
    # 6M (k=2)
    spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, 2)
    butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, 2)
    double_butterflies_6M_df = calculate_k_step_double_butterflies(analysis_curve_df, 2) 
    
    # 12M (k=4)
    spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, 4)
    butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, 4)
    double_butterflies_12M_df = calculate_k_step_double_butterflies(analysis_curve_df, 4) 
    
    st.markdown("##### 3-Month Outright Spreads (k=1, e.g., Z25-H26)")
    st.dataframe(spreads_3M_df_raw.head(5))
    st.markdown("##### 3-Month Double Butterfly (k=1, e.g., $Z25-3 \cdot H26+3 \cdot M26-Z26$)")
    st.dataframe(double_butterflies_3M_df.head(5))
    
    if spreads_3M_df_raw.empty:
        st.warning("3M Spreads could not be calculated. Need at least two contracts in the analysis curve.")
        st.stop()
        
    # 4. Perform PCA
    # 4a. PCA on 3M Spreads (Standard Method - Used for Fair Curve Reconstruction & Hedging)
    loadings_spread, explained_variance_ratio, eigenvalues, scores, spreads_3M_df_clean = perform_pca(spreads_3M_df_raw)
    
    # 4b. PCA on Outright Prices (User Requested Independent Method - Unstandardized/Covariance)
    loadings_outright_direct, explained_variance_outright_direct = perform_pca_on_prices(analysis_curve_df)
    
    if loadings_spread is not None and loadings_outright_direct is not None:
        
        # --- Explained Variance Visualization (Section 2) ---
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
            st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the spreads. This is the risk model used.")
            
        # --- Component Loadings Heatmaps (Section 3) ---
        st.header("3. PC Loadings")
        
        # --- 3.1 Spread Loadings (Standard Method) ---
        st.subheader("3.1 PC Loadings Heatmap (PC vs. 3M Spreads)")
        st.markdown(""" 
        This heatmap shows the **Loadings (Eigenvectors)** of the first few PCs on each **3-Month Spread**. 
        These weights are derived from **Standardized PCA** and represent how each spread contributes to the overall risk factors (Level, Slope, Curvature).
        """)
        plt.style.use('default')
        fig_spread_loading, ax_spread_loading = plt.subplots(figsize=(12, 6))
        loadings_spread_plot = loadings_spread.iloc[:, :default_pc_count]
        sns.heatmap(
            loadings_spread_plot, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            center=0, 
            ax=ax_spread_loading
        )
        ax_spread_loading.set_title(f'Loadings of {loadings_spread_plot.shape[1]} PCs on 3M Spreads (Standardized PCA)')
        ax_spread_loading.set_ylabel('3M Spreads')
        st.pyplot(fig_spread_loading)
        
        # --- 3.2 Outright Price Loadings (Covariance Method) ---
        st.subheader("3.2 PC Loadings Heatmap (PC vs. Outright Prices)")
        st.markdown(""" 
        This heatmap shows the **Loadings** of the first few PCs on each **Outright Contract Price**. 
        These weights are derived from **Covariance PCA (unstandardized)** and show the absolute duration-scaled sensitivity. 
        Note the non-uniform nature of PC1 (duration factor).
        """)
        fig_price_loading, ax_price_loading = plt.subplots(figsize=(12, 6))
        loadings_price_plot = loadings_outright_direct.iloc[:, :default_pc_count]
        sns.heatmap(
            loadings_price_plot, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            center=0, 
            ax=ax_price_loading
        )
        ax_price_loading.set_title(f'Loadings of {loadings_price_plot.shape[1]} PCs on Outright Prices (Covariance PCA)')
        ax_price_loading.set_ylabel('Outright Contracts')
        st.pyplot(fig_price_loading)
        
        # --- 4. Fair Curve Reconstruction ---
        st.header("4. Fair Curve Reconstruction")
        
        # Reconstructed 3M Spreads
        reconstructed_spreads_3M_raw = pd.DataFrame(
            scores.iloc[:, :pc_count] @ loadings_spread.iloc[:, :pc_count].T,
            index=scores.index,
            columns=loadings_spread.index
        )
        
        (
            historical_outrights, 
            historical_spreads_3M, 
            historical_butterflies_3M, 
            historical_spreads_6M, 
            historical_butterflies_6M, 
            historical_spreads_12M, 
            historical_butterflies_12M,
            historical_double_butterflies_3M, 
            historical_double_butterflies_6M, 
            historical_double_butterflies_12M,
            spreads_3M_df_no_prefix
        ) = reconstruct_prices_and_derivatives(
            analysis_curve_df, 
            reconstructed_spreads_3M_raw, 
            spreads_3M_df_raw, spreads_6M_df, butterflies_3M_df, butterflies_6M_df, 
            spreads_12M_df, butterflies_12M_df, double_butterflies_3M_df, double_butterflies_6M_df, double_butterflies_12M_df
        )
        
        # Combine all historical derivative DFs for sections 5, 6, 9, 10, 11
        all_historical_derivatives = pd.concat([
            historical_spreads_3M, 
            historical_butterflies_3M, 
            historical_double_butterflies_3M,
            historical_spreads_6M, 
            historical_butterflies_6M, 
            historical_double_butterflies_6M,
            historical_spreads_12M, 
            historical_butterflies_12M,
            historical_double_butterflies_12M
        ], axis=1)

        # Get the latest row for the curve snapshot
        curve_snapshot_dt = analysis_dt if analysis_dt in historical_outrights.index else historical_outrights.index[-1]
        curve_snapshot_prices = historical_outrights.loc[curve_snapshot_dt]
        curve_snapshot_derivatives = all_historical_derivatives.loc[curve_snapshot_dt].to_frame().T
        
        # Clean up labels for plotting/display
        contract_maturities = [c for c in curve_snapshot_prices.index if 'Original' in c]
        outright_prices_df = pd.DataFrame({
            'Contract': [c.replace(' (Original)', '') for c in contract_maturities],
            'Market Price': curve_snapshot_prices.loc[contract_maturities].values,
            'PCA Fair Price': curve_snapshot_prices.loc[[c.replace('(Original)', '(PCA)') for c in contract_maturities]].values,
        }).set_index('Contract')
        
        outright_prices_df['Mispricing (BPS)'] = (outright_prices_df['Market Price'] - outright_prices_df['PCA Fair Price']) * 10000
        
        st.subheader(f"4.1 Outright Curve Snapshot ({curve_snapshot_dt.strftime('%Y-%m-%d')})")
        
        # Plotting the curve snapshot
        fig_curve, ax_curve = plt.subplots(figsize=(10, 5))
        ax_curve.plot(outright_prices_df.index, outright_prices_df['Market Price'], marker='o', label='Market Price')
        ax_curve.plot(outright_prices_df.index, outright_prices_df['PCA Fair Price'], marker='x', linestyle='--', label='PCA Fair Price')
        ax_curve.set_title('Outright Market Curve vs. PCA Fair Curve')
        ax_curve.set_ylabel('Price Level')
        ax_curve.set_xlabel('Contract Maturity')
        ax_curve.legend()
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_curve)
        
        st.dataframe(outright_prices_df.style.format({
            'Market Price': "{:.4f}", 
            'PCA Fair Price': "{:.4f}",
            'Mispricing (BPS)': "{:.2f}"
        }), use_container_width=True)
        
        # --- NEW SECTION 5: Historical Mispricing Analysis ---
        st.header("5. Historical Mispricing Analysis")
        st.markdown("Select a single derivative to visualize its market vs. fair value over the historical period.")
        
        derivative_labels = sorted([col.replace(' (Original)', '') for col in all_historical_derivatives.columns if 'Original' in col])
        if derivative_labels:
            # Check if previous selection is still valid
            try:
                selected_derivative = st.selectbox("Select Derivative for Historical Tracking:", derivative_labels, key='historical_mispricing_selector')
            except StreamlitAPIException: # Handle case where selection might be out of sync after code update
                selected_derivative = st.selectbox("Select Derivative for Historical Tracking:", derivative_labels, index=0, key='historical_mispricing_selector_resync')
                
            plot_historical_tracking(all_historical_derivatives, selected_derivative)
        else:
            st.info("No derivatives found for historical analysis.")

        # --- Renumbered Section 6: Comprehensive Derivative Mispricing Snapshot ---
        st.header("6. Comprehensive Derivative Mispricing Snapshot")
        st.subheader(f"All Derivative Mispricing ({curve_snapshot_dt.strftime('%Y-%m-%d')})")
        
        # Restore the comprehensive table
        mispricing_data = []
        for col in curve_snapshot_derivatives.columns:
            if 'Original' in col:
                derivative = col.replace(' (Original)', '')
                market_val = curve_snapshot_derivatives.loc[curve_snapshot_dt, col]
                pca_val = curve_snapshot_derivatives.loc[curve_snapshot_dt, derivative + ' (PCA)']
                
                mispricing_data.append({
                    'Derivative': derivative,
                    'Market Value': market_val,
                    'PCA Fair Value': pca_val,
                    'Mispricing (BPS)': (market_val - pca_val) * 10000
                })
        
        mispricing_df_all = pd.DataFrame(mispricing_data).set_index('Derivative')
        
        # Identify most mispriced
        mispricing_df_all['Abs Mispricing (BPS)'] = mispricing_df_all['Mispricing (BPS)'].abs()
        mispricing_df_all = mispricing_df_all.sort_values(by='Abs Mispricing (BPS)', ascending=False).drop(columns=['Abs Mispricing (BPS)'])
        
        st.markdown("All Derivatives (3M, 6M, 12M Spreads, Flies, and Double Flies) sorted by absolute mispricing (Mispricing = Market Value - PCA Fair Value):")
        st.dataframe(mispricing_df_all.style.format({
            'Market Value': "{:.4f}", 
            'PCA Fair Value': "{:.4f}",
            'Mispricing (BPS)': "{:.2f}"
        }), use_container_width=True)


        # --- Renumbered Section 7: Minimum Variance Hedge (3M Spread Only) ---
        st.header("7. Minimum Variance Hedge (3M Spread Only)")
        st.markdown("Calculates the optimal hedge ratio (k*) using the PCA-reconstructed covariance matrix for **3M spreads** only.")
        
        trade_labels_3M = spreads_3M_df_no_prefix.columns.tolist()
        if trade_labels_3M:
            try:
                selected_trade_3M = st.selectbox(
                    "Select Trade (3M Spread) to Hedge:", 
                    trade_labels_3M, 
                    key='trade_select_3M'
                )
            except StreamlitAPIException:
                selected_trade_3M = st.selectbox(
                    "Select Trade (3M Spread) to Hedge:", 
                    trade_labels_3M, 
                    index=0,
                    key='trade_select_3M_resync'
                )
            
            best_hedge_3M, worst_hedge_3M, results_df_3M = calculate_best_and_worst_hedge_3M(
                selected_trade_3M, 
                loadings_spread, 
                eigenvalues, 
                pc_count, 
                spreads_3M_df_clean
            )
            
            if best_hedge_3M is not None:
                st.subheader(f"Hedging {selected_trade_3M} (Trade size: 1 lot)")
                col_best, col_worst = st.columns(2)
                
                with col_best:
                    st.success(f"Best Hedge: {best_hedge_3M['Hedge Spread']}")
                    st.metric(label="Hedge Ratio (k*)", value=f"{best_hedge_3M['Hedge Ratio (k*)']:.4f}")
                    st.metric(label="Residual Volatility (BPS)", value=f"{best_hedge_3M['Residual Volatility (BPS)']:.2f}", help="Lower is better. Represents the remaining market risk in BPS.")
                    st.markdown(f"**Action:** Sell/Buy **{abs(best_hedge_3M['Hedge Ratio (k*)']):.4f}** lots of **{best_hedge_3M['Hedge Spread']}**")

                with col_worst:
                    st.error(f"Worst Hedge: {worst_hedge_3M['Hedge Spread']}")
                    st.metric(label="Hedge Ratio (k*)", value=f"{worst_hedge_3M['Hedge Ratio (k*)']:.4f}")
                    st.metric(label="Residual Volatility (BPS)", value=f"{worst_hedge_3M['Residual Volatility (BPS)']:.2f}", help="Higher is worse.")

                with st.expander("View Full 3M Hedge Results Table"):
                    st.dataframe(results_df_3M.style.format({
                        'Hedge Ratio (k*)': "{:.4f}",
                        'Residual Volatility (BPS)': "{:.2f}"
                    }), use_container_width=True)
            else:
                st.warning("Cannot calculate 3M spread hedges. Check if selected PC count is valid.")


        # --- Renumbered Section 8: Generalized Minimum Variance Hedge (All Instruments) ---
        st.header("8. Generalized Minimum Variance Hedge (All Instruments)")
        
        all_derivatives_df_raw = pd.concat([
            spreads_3M_df_raw.rename(columns=lambda x: f"3M Spread: {x}"),
            butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}"),
            double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}"),
            spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}"),
            butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}"),
            double_butterflies_6M_df.rename(columns=lambda x: f"6M Double Fly: {x}"),
            spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}"),
            butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}"),
            double_butterflies_12M_df.rename(columns=lambda x: f"12M Double Fly: {x}"),
        ], axis=1)
        
        # Calculate generalized Covariance and Loadings (L_D)
        Sigma_Raw_df, all_derivatives_aligned, loadings_df_gen = calculate_derivatives_covariance_generalized(
            all_derivatives_df_raw, scores, eigenvalues, pc_count
        )
        
        all_derivative_labels = sorted(all_derivatives_aligned.columns.tolist())
        
        if all_derivative_labels:
            try:
                selected_trade_gen = st.selectbox(
                    "Select Trade (Any Derivative) to Hedge:", 
                    all_derivative_labels, 
                    key='trade_select_gen'
                )
            except StreamlitAPIException:
                selected_trade_gen = st.selectbox(
                    "Select Trade (Any Derivative) to Hedge:", 
                    all_derivative_labels, 
                    index=0,
                    key='trade_select_gen_resync'
                )

            best_hedge_gen, worst_hedge_gen, results_df_gen = calculate_best_and_worst_hedge_generalized(
                selected_trade_gen, 
                Sigma_Raw_df
            )
            
            if best_hedge_gen is not None:
                st.subheader(f"Hedging {selected_trade_gen} (Trade size: 1 lot)")
                col_best_gen, col_worst_gen = st.columns(2)
                
                with col_best_gen:
                    st.success(f"Best Hedge: {best_hedge_gen['Hedge Instrument']}")
                    st.metric(label="Hedge Ratio (k*)", value=f"{best_hedge_gen['Hedge Ratio (k*)']:.4f}")
                    st.metric(label="Residual Volatility (BPS)", value=f"{best_hedge_gen['Residual Volatility (BPS)']:.2f}", help="Lower is better.")
                    
                with col_worst_gen:
                    st.error(f"Worst Hedge: {worst_hedge_gen['Hedge Instrument']}")
                    st.metric(label="Hedge Ratio (k*)", value=f"{worst_hedge_gen['Hedge Ratio (k*)']:.4f}")
                    st.metric(label="Residual Volatility (BPS)", value=f"{worst_hedge_gen['Residual Volatility (BPS)']:.2f}", help="Higher is worse.")
                    
                with st.expander("View Full Generalized Hedge Results Table"):
                    st.dataframe(results_df_gen.style.format({
                        'Hedge Ratio (k*)': "{:.4f}",
                        'Residual Volatility (BPS)': "{:.2f}"
                    }), use_container_width=True)
            else:
                st.warning("Cannot calculate generalized minimum variance hedges. Check if data is sufficient.")
        else:
            st.info("No derivatives available for generalized hedging analysis.")

        # --- NEW SECTION 9: Variance Attribution ---
        st.header("9. Factor Contribution to Trade Variance")
        st.markdown("Identifies the percentage of a trade's historical volatility explained by each Principal Component.")

        # 1. Calculate Standardized Loadings and Std Dev for Outright Prices
        outright_loadings_df, outright_std_dev = calculate_loadings_for_new_instruments(
            analysis_curve_df, scores, len(eigenvalues) 
        )
        
        # 2. Combine all sensitivities (Derivatives + Outright Prices)
        combined_sensitivities = calculate_combined_sensitivities(loadings_df_gen, outright_loadings_df)
        
        # 3. Combine all standard deviations
        combined_std_dev = pd.concat([all_derivatives_aligned.std(), outright_std_dev])
        
        all_instrument_labels = sorted(combined_sensitivities.index.tolist())

        if not all_instrument_labels:
            st.info("No instruments available for variance attribution.")
        else:
            try:
                selected_trade_attr = st.selectbox(
                    "Select Instrument for Variance Attribution (Derivative or Outright):", 
                    all_instrument_labels, 
                    key='trade_select_attr'
                )
            except StreamlitAPIException:
                selected_trade_attr = st.selectbox(
                    "Select Instrument for Variance Attribution (Derivative or Outright):", 
                    all_instrument_labels, 
                    index=0,
                    key='trade_select_attr_resync'
                )
            
            attribution_df, error = calculate_variance_attribution(
                selected_trade_attr, 
                combined_sensitivities, # Use combined sensitivities here
                eigenvalues, 
                combined_std_dev, # Use combined standard deviation here
                pc_count
            )

            if error:
                st.warning(error)
            elif not attribution_df.empty:
                st.subheader(f"Variance Attribution for {selected_trade_attr}")
                
                # Use factor labels if available
                factor_map = {f'PC{i+1}': factor for i, factor in enumerate(['Level', 'Slope', 'Curvature'])}
                attribution_df['Factor Label'] = [factor_map.get(idx, idx) for idx in attribution_df.index]
                
                # Plot
                fig_attr, ax_attr = plt.subplots(figsize=(8, 5))
                ax_attr.bar(attribution_df['Factor Label'], attribution_df['Percentage Contribution (%)'])
                ax_attr.set_title(f"PCA Factor Contribution to {selected_trade_attr} Volatility")
                ax_attr.set_ylabel('Percentage Contribution (%)')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_attr)
                
                st.dataframe(attribution_df.drop(columns=['Factor Label']).style.format({
                    'Raw Var. Contribution': "{:.8f}",
                    'Percentage Contribution (%)': "{:.2f}"
                }), use_container_width=True)
                
                
        # --- NEW SECTION 10: Outright Factor Hedging (Duration Hedge) ---
        st.header("10. Outright Factor Hedging (Duration Hedge)")
        st.markdown("""
        Neutralize the risk exposure of a derivative trade to a single factor (e.g., PC1/Level) 
        by hedging with a single outright contract.
        """)
        
        # Filter combined sensitivities for PC1 (Level)
        factor_name_pc1 = 'PC1'
        if factor_name_pc1 in combined_sensitivities.columns:
            
            pc1_sensitivities = combined_sensitivities[factor_name_pc1].to_frame().rename(
                columns={factor_name_pc1: 'Level (PC1) Sensitivity'}
            )

            # Available trades are derivatives
            try:
                selected_trade_outright_hedge = st.selectbox(
                    "Select Derivative Trade to Hedge (T):", 
                    all_derivative_labels, 
                    key='trade_select_outright_hedge'
                )
            except StreamlitAPIException:
                 selected_trade_outright_hedge = st.selectbox(
                    "Select Derivative Trade to Hedge (T):", 
                    all_derivative_labels, 
                    index=0,
                    key='trade_select_outright_hedge_resync'
                )
            
            # Available hedges are outright contracts
            outright_hedge_labels = outright_loadings_df.index.tolist()
            try:
                selected_hedge_outright = st.selectbox(
                    "Select Outright Contract to Use as Hedge (H):", 
                    outright_hedge_labels, 
                    key='hedge_select_outright'
                )
            except StreamlitAPIException:
                selected_hedge_outright = st.selectbox(
                    "Select Outright Contract to Use as Hedge (H):", 
                    outright_hedge_labels, 
                    index=0,
                    key='hedge_select_outright_resync'
                )
            
            Trade_Exposure = pc1_sensitivities.loc[selected_trade_outright_hedge, 'Level (PC1) Sensitivity']
            Hedge_Exposure = pc1_sensitivities.loc[selected_hedge_outright, 'Level (PC1) Sensitivity']

            if abs(Hedge_Exposure) < 1e-9:
                st.warning(f"The hedge instrument {selected_hedge_outright} has zero sensitivity to the Level (PC1) factor.")
            else:
                # Calculate Factor Hedge Ratio
                k_factor = Trade_Exposure / Hedge_Exposure
                
                # --- Residual Volatility Calculation using Factor Model Approximation ---
                
                # 1. Get Cov(T, H) using the combined factor model:
                # Cov(T, H) = sum_{i=1}^{N} (beta_{T,i} * beta_{H,i}) * lambda_i * sigma_T * sigma_H
                
                trade_loadings = combined_sensitivities.loc[selected_trade_outright_hedge].iloc[:pc_count]
                hedge_loadings = combined_sensitivities.loc[selected_hedge_outright].iloc[:pc_count]
                lambda_p = pd.Series(eigenvalues[:pc_count])
                
                sigma_T = combined_std_dev.loc[selected_trade_outright_hedge]
                sigma_H = combined_std_dev.loc[selected_hedge_outright]
                
                # Raw Cov(T, H) approximation
                cov_th = (trade_loadings * hedge_loadings * lambda_p).sum() * sigma_T * sigma_H
                
                # Raw Var(T) is already in Sigma_Raw_df for the derivative
                Var_Trade = Sigma_Raw_df.loc[selected_trade_outright_hedge, selected_trade_outright_hedge]
                
                # Raw Var(H) approximation
                Var_Hedge = (hedge_loadings**2 * lambda_p).sum() * (sigma_H**2)
                
                Residual_Variance = Var_Trade + (k_factor**2 * Var_Hedge) - (2 * k_factor * cov_th)
                Residual_Volatility_BPS = np.sqrt(max(0, Residual_Variance)) * 10000
                
                st.subheader(f"Factor Hedge for {selected_trade_outright_hedge} vs. {selected_hedge_outright} (Target: PC1 Neutral)")
                
                col_factor_hedge, col_info = st.columns(2)
                
                with col_factor_hedge:
                    st.metric(label="Factor Hedge Ratio (k_factor)", value=f"{k_factor:.4f}", help="Quantity of the hedge instrument needed to zero out PC1 risk.")
                    st.metric(label="Residual Volatility (BPS)", value=f"{Residual_Volatility_BPS:.2f}", help="The remaining risk in the portfolio after hedging PC1.")
                    st.markdown(f"**Action:** Sell/Buy **{abs(k_factor):.4f}** lots of **{selected_hedge_outright}**")

                with col_info:
                    st.metric(label="Trade PC1 Exposure", value=f"{Trade_Exposure:.4f}")
                    st.metric(label="Hedge PC1 Exposure", value=f"{Hedge_Exposure:.4f}")

        else:
            st.info("Level (PC1) factor not available in combined sensitivities for outright hedging.")


        # --- Renumbered Section 11: Factor-Based Hedging (Derivative vs Derivative) ---
        st.header("11. Factor-Based Hedging (Derivative vs Derivative)")
        st.markdown("Neutralize exposure to a specific risk factor (e.g., Slope) by hedging with another derivative.")
        
        factor_sensitivities_df = calculate_factor_sensitivities(loadings_df_gen, pc_count)
        
        factor_options = factor_sensitivities_df.columns.tolist()
        
        if factor_options and all_derivative_labels:
            col_trade_f, col_factor_f = st.columns(2)
            
            with col_trade_f:
                try:
                    selected_trade_factor = st.selectbox(
                        "Select Trade (Derivative) to Hedge:", 
                        all_derivative_labels, 
                        key='trade_select_factor'
                    )
                except StreamlitAPIException:
                    selected_trade_factor = st.selectbox(
                        "Select Trade (Derivative) to Hedge:", 
                        all_derivative_labels, 
                        index=0,
                        key='trade_select_factor_resync'
                    )
            
            with col_factor_f:
                try:
                    selected_factor = st.selectbox(
                        "Select Factor to Neutralize:", 
                        factor_options, 
                        key='factor_select'
                    )
                except StreamlitAPIException:
                    selected_factor = st.selectbox(
                        "Select Factor to Neutralize:", 
                        factor_options, 
                        index=0,
                        key='factor_select_resync'
                    )
                
            factor_results_df, error_f = calculate_all_factor_hedges(
                selected_trade_factor, selected_factor, factor_sensitivities_df, Sigma_Raw_df
            )
            
            if error_f:
                st.warning(error_f)
            elif not factor_results_df.empty:
                st.subheader(f"Best Hedges for {selected_trade_factor} to Neutralize **{selected_factor}**")
                st.markdown(f"The table below is sorted by the lowest **Residual Volatility**, showing the most effective factor hedges.")
                
                # Prepare table for display
                display_df = factor_results_df.rename(columns={
                    f'Factor Hedge Ratio (k_factor)': 'Hedge Ratio (k_factor)',
                    'Residual Volatility (BPS)': 'Residual Volatility (BPS)'
                })[['Hedge Instrument', 'Hedge Ratio (k_factor)', 'Residual Volatility (BPS)', 'Hedge Sensitivity']]
                
                # Format and display
                st.dataframe(
                    display_df.style.format({
                        'Hedge Ratio (k_factor)': "{:.4f}",
                        'Residual Volatility (BPS)': "{:.2f}",
                        'Hedge Sensitivity': "{:.4f}",
                    }),
                    use_container_width=True
                )
                
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
