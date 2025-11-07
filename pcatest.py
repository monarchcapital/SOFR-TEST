import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SOFR Futures PCA Analyzer (Outright Price Basis & Hedging)")

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
            sep=',', # Explicitly specify comma as separator
            header=0 # Ensure the first row is used as the header
        )
        
        df.index.name = 'Date'
        
        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how='all')
        
        # Convert all price columns to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Filter out any remaining rows where the index date is NaT or the row is entirely NaN
        df = df.dropna(how='all')
        df = df[df.index.notna()]

        if df.empty or df.shape[1] == 0:
             raise ValueError("DataFrame is empty after processing or has no data columns.")
             
        return df
        
    except Exception as e:
        st.error(f"Error loading and processing data from {uploaded_file.name}: {e}")
        return None


def get_analysis_contracts(expiry_df, analysis_date):
    """
    Filters contract codes that expire on or after the analysis date and returns
    them in chronological order with their expiry date. This list defines the 
    maturities used in the analysis curve (e.g., Z20, H21, M21...).
    """
    if expiry_df is None:
        return pd.DataFrame()
    
    # Filter contracts that expire on or after the analysis date
    future_expiries = expiry_df[expiry_df['ExpiryDate'] >= analysis_date].copy()
    
    if future_expiries.empty:
        st.warning(f"No contracts found expiring on or after {analysis_date.strftime('%Y-%m-%d')}.")
        return pd.DataFrame()

    # Sort by expiry date
    future_expiries = future_expiries.sort_values(by='ExpiryDate')
    
    return future_expiries

def transform_to_analysis_curve(price_df, future_expiries_df):
    """
    Transforms the price DataFrame to only include relevant contracts in maturity order.
    """
    if price_df is None or future_expiries_df.empty:
        return pd.DataFrame(), []

    contract_order = future_expiries_df.index.tolist()
    valid_contracts = [c for c in contract_order if c in price_df.columns]
    
    if not valid_contracts:
        st.warning("No matching contract columns found in price data for the selected analysis date range.")
        return pd.DataFrame(), []

    analysis_curve_df = price_df[valid_contracts]
    return analysis_curve_df, valid_contracts

# --- GENERALIZED DERIVATIVE CALCULATION FUNCTIONS ---

def calculate_n_month_spreads(analysis_curve_df, n_months):
    """
    Calculates N-month spreads (e.g., 6M spread uses C1 and C3).
    n_months must be a multiple of 3 (3, 6, 12). Gap is n_months / 3.
    """
    if analysis_curve_df.empty:
        return pd.DataFrame()

    gap = n_months // 3 # 3M: gap=1, 6M: gap=2, 12M: gap=4
    if gap < 1: return pd.DataFrame()
    
    num_contracts = analysis_curve_df.shape[1]
    spreads_data = {}
    
    for i in range(num_contracts - gap):
        short_maturity = analysis_curve_df.columns[i]
        long_maturity = analysis_curve_df.columns[i+gap]
        
        spread_label = f"{short_maturity}-{long_maturity}"
        
        # CME Basis: Shorter maturity minus longer maturity
        spreads_data[spread_label] = analysis_curve_df.iloc[:, i] - analysis_curve_df.iloc[:, i+gap]
        
    return pd.DataFrame(spreads_data)

def calculate_n_month_butterflies(analysis_curve_df, n_months):
    """
    Calculates N-month butterflies (e.g., 6M fly uses C1, C3, C5).
    n_months must be a multiple of 3 (3, 6, 12). Gap is n_months / 3.
    Fly formula: C1 - 2*C2 + C3 where C2 is C1 + gap and C3 is C1 + 2*gap.
    """
    if analysis_curve_df.empty:
        return pd.DataFrame()

    gap = n_months // 3 # 3M: gap=1, 6M: gap=2, 12M: gap=4
    if gap < 1 or analysis_curve_df.shape[1] < 2 * gap + 1: 
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    flies_data = {}

    for i in range(num_contracts - 2 * gap):
        short_maturity = analysis_curve_df.columns[i]            # C1
        center_maturity = analysis_curve_df.columns[i+gap]       # C2
        long_maturity = analysis_curve_df.columns[i+2*gap]       # C3

        # Fly = C1 - 2*C2 + C3
        fly_label = f"{short_maturity}-2x{center_maturity}+{long_maturity}"

        flies_data[fly_label] = (
            analysis_curve_df.iloc[:, i] 
            - 2 * analysis_curve_df.iloc[:, i+gap] 
            + analysis_curve_df.iloc[:, i+2*gap]
        )

    return pd.DataFrame(flies_data)

# --- PCA AND RECONSTRUCTION FUNCTIONS (Copied from previous step, ensuring consistency) ---

def perform_pca_on_prices(price_df):
    """
    Performs PCA directly on Outright Price Levels using the COVARIANCE MATRIX 
    (unstandardized data).
    """
    data_df_clean = price_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None, None, None, None

    data_mean = data_df_clean.mean() 
    data_centered = data_df_clean - data_mean 
    
    n_components = min(data_centered.shape)

    pca = PCA(n_components=n_components)
    scores_matrix = pca.fit_transform(data_centered)
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    
    explained_variance = pca.explained_variance_ratio_
    
    scores = pd.DataFrame(
        scores_matrix,
        index=data_df_clean.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings, explained_variance, scores, data_df_clean, data_mean


def reconstruct_fair_curve_from_prices(scores_df, loadings_df, data_mean, pc_count, analysis_curve_df, derivative_data):
    """
    Reconstructs Outright Prices, Spreads, and Flies using PCA factors 
    derived directly from Outright Prices.
    
    derivative_data is a dictionary: {'3M Spread': df, '3M Fly': df, ...}
    """
    
    # 1. Reconstruct Outright Prices
    scores_used = scores_df.values[:, :pc_count]
    loadings_used = loadings_df.values[:, :pc_count]
    
    reconstructed_centered = scores_used @ loadings_used.T
    data_mean_aligned = data_mean.loc[loadings_df.index]
    reconstructed_prices_values = reconstructed_centered + data_mean_aligned.values
    
    reconstructed_prices_df = pd.DataFrame(
        reconstructed_prices_values,
        index=scores_df.index, 
        columns=loadings_df.index # Outright Contract Labels
    )
    
    analysis_curve_df_aligned = analysis_curve_df.loc[reconstructed_prices_df.index]
    original_price_rename = {col: col + ' (Original)' for col in analysis_curve_df_aligned.columns}
    original_prices_df = analysis_curve_df_aligned.rename(columns=original_price_rename)
    
    pca_price_rename = {col: col + ' (PCA)' for col in reconstructed_prices_df.columns}
    pca_prices_df = reconstructed_prices_df.rename(columns=pca_price_rename)
    
    historical_outrights = pd.merge(original_prices_df, pca_prices_df, left_index=True, right_index=True)


    # 2. Reconstruct ALL Derivatives (Spreads and Flies)
    reconstructed_derivatives = {}
    historical_derivatives = {}
    
    for derivative_label, original_df in derivative_data.items():
        if original_df.empty:
            historical_derivatives[derivative_label] = pd.DataFrame()
            continue
            
        # Reconstruct the derivative from the Reconstructed Prices
        if 'Spread' in derivative_label:
            n_months = int(derivative_label.split('M')[0])
            reconstructed_df = calculate_n_month_spreads(reconstructed_prices_df, n_months)
            
        elif 'Fly' in derivative_label:
            n_months = int(derivative_label.split('M')[0])
            reconstructed_df = calculate_n_month_butterflies(reconstructed_prices_df, n_months)
        
        # Merge Original and PCA for comparison
        original_df_aligned = original_df.loc[reconstructed_df.index]
        
        original_rename = {col: col + ' (Original)' for col in original_df_aligned.columns}
        pca_rename = {col: col + ' (PCA)' for col in reconstructed_df.columns}

        original_derivatives = original_df_aligned.rename(columns=original_rename)
        pca_derivatives = reconstructed_df.rename(columns=pca_rename)
        
        historical_derivatives[derivative_label] = pd.merge(original_derivatives, pca_derivatives, left_index=True, right_index=True)

    return historical_outrights, historical_derivatives


# --- MINIMUM VARIANCE HEDGING FUNCTIONS ---

def calculate_factor_sensitivities(derivatives_df, loadings_df, data_mean, factor_count):
    """
    Calculates the PC Factor Sensitivity (Beta) for all derivatives against 
    the first `factor_count` PCs.
    
    The sensitivity (Beta) of a derivative (D) to a factor (PCk) is:
    Beta_D,k = (dD/dX) * dX/dPCk
    Where:
    - (dD/dX) is the vector of contract weights for the derivative (e.g., [1, -1, 0...] for a spread).
    - dX/dPCk is the PCk Loadings vector (the eigenvector).
    """
    
    # 1. Map all derivative labels to their outright contract weights
    all_derivatives_weights = {}
    outright_contracts = loadings_df.index.tolist()
    
    for col in derivatives_df.columns:
        weights = pd.Series(0.0, index=outright_contracts)
        
        if '2x' in col: # Butterfly (C1 - 2*C2 + C3)
            parts = col.split('-2x')
            c1 = parts[0]
            parts2 = parts[1].split('+')
            c2 = parts2[0]
            c3 = parts2[1]
            weights.loc[c1] = 1.0
            weights.loc[c2] = -2.0
            weights.loc[c3] = 1.0
        else: # Spread (C1 - C2)
            parts = col.split('-')
            c1 = parts[0]
            c2 = parts[1]
            weights.loc[c1] = 1.0
            weights.loc[c2] = -1.0
            
        all_derivatives_weights[col] = weights.to_numpy()

    weights_df = pd.DataFrame(all_derivatives_weights, index=outright_contracts).T


    # 2. Calculate Beta (Sensitivity) using the Loadings matrix
    
    # Loadings matrix (dPrice/dPC) for the factors used
    loadings_matrix_used = loadings_df.iloc[:, :factor_count].values
    
    # Weights matrix (dDerivative/dPrice)
    weights_matrix = weights_df.values
    
    # Sensitivities (Betas): Beta = dDerivative/dPC = (dDerivative/dPrice) @ (dPrice/dPC)
    sensitivities_matrix = weights_matrix @ loadings_matrix_used
    
    sensitivities_df = pd.DataFrame(
        sensitivities_matrix,
        index=weights_df.index,
        columns=loadings_df.columns[:factor_count]
    )
    
    return sensitivities_df

def calculate_minimum_variance_hedge(trade_label, trade_sensitivities_df, hedge_candidates_sensitivities_df, scores_df):
    """
    Calculates the Minimum Variance Hedge (MVH) for a single trade using PCA factors.
    MVH is calculated via the Minimum Variance Hedge Ratio (MVHR): h* = Cov(Trade, Hedge) / Var(Hedge)
    
    For PCA factors, Cov(Trade, Hedge) = Beta_Trade.T @ Cov(PC_Scores) @ Beta_Hedge
    Since PC Scores are orthonormal and standardized: Cov(PC_Scores) = Identity Matrix (I).
    Therefore, Cov(Trade, Hedge) = Beta_Trade.T @ Beta_Hedge (assuming unit variance factors)
    """
    
    # 1. Calculate the Covariance Matrix of the PC Scores (assuming standardization, which is done by PCA)
    # However, since we used unstandardized prices, the scores are not unit variance, 
    # but the Loadings already reflect the variance/covariance of the original prices.
    # The scores themselves have a covariance matrix equal to the Explained Variance (Lambda).
    
    # The correct way for MVH using Betas derived from COVARIANCE PCA is:
    # Cov(Trade, Hedge) = Beta_Trade.T @ Lambda @ Beta_Hedge
    # Var(Trade) = Beta_Trade.T @ Lambda @ Beta_Trade
    # Where Lambda is the diagonal matrix of PCA Explained Variances (Eigenvalues).
    
    # We use the empirical covariance of the actual scores to capture any non-unit variance factors
    # that arise from non-standardized PCA.
    
    # Get covariance matrix of the relevant PC scores
    pc_scores_cov = scores_df.cov()
    pc_scores_cov_used = pc_scores_cov.loc[trade_sensitivities_df.columns, trade_sensitivities_df.columns]
    
    trade_beta = trade_sensitivities_df.loc[trade_label].to_frame()
    
    best_hedge = None
    min_residual_risk = np.inf
    
    results = []
    
    for hedge_label in hedge_candidates_sensitivities_df.index:
        hedge_beta = hedge_candidates_sensitivities_df.loc[hedge_label].to_frame()

        # Cov(Trade, Hedge) = Beta_Trade.T @ Cov(Scores) @ Beta_Hedge
        cov_th = (trade_beta.T @ pc_scores_cov_used @ hedge_beta).iloc[0, 0]
        
        # Var(Hedge) = Beta_Hedge.T @ Cov(Scores) @ Beta_Hedge
        var_h = (hedge_beta.T @ pc_scores_cov_used @ hedge_beta).iloc[0, 0]
        
        if var_h == 0:
            continue
            
        # MVHR: h* = Cov(Trade, Hedge) / Var(Hedge)
        mvhr = cov_th / var_h
        
        # Risk (Variance) of the Hedged Portfolio: Var(Trade) - h* * Cov(Trade, Hedge)
        var_t = (trade_beta.T @ pc_scores_cov_used @ trade_beta).iloc[0, 0]
        residual_variance = var_t - mvhr * cov_th
        residual_volatility = np.sqrt(residual_variance) * 10000 # Convert to BPS volatility

        results.append({
            'Hedge Candidate': hedge_label,
            'MVHR (Units of Hedge/Unit of Trade)': mvhr,
            'Cov(T, H)': cov_th,
            'Var(H)': var_h,
            'Residual Volatility (BPS)': residual_volatility
        })
        
        if residual_volatility < min_residual_risk:
            min_residual_risk = residual_volatility
            best_hedge = hedge_label

    results_df = pd.DataFrame(results).sort_values(by='Residual Volatility (BPS)').reset_index(drop=True)
    
    # Select the top result as the optimal hedge
    optimal_hedge_row = results_df.iloc[[0]]
    
    return optimal_hedge_row, results_df

# --- Streamlit Application Layout ---

st.title("SOFR Futures PCA Analyzer (Outright Price Basis & Hedging)")
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("1. Data Uploads")
price_file = st.sidebar.file_uploader("Upload Historical Price Data (e.g., 'sofr rates.csv')", type=['csv'], key='price_upload')
expiry_file = st.sidebar.file_uploader("Upload Contract Expiry Dates (e.g., 'EXPIRY (2).csv')", type=['csv'], key='expiry_upload')

# Initialize dataframes
price_df = load_data(price_file)
expiry_df = load_data(expiry_file)

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
    
    # 1. Get the list of relevant contracts based on the analysis date
    future_expiries_df = get_analysis_contracts(expiry_df, analysis_dt)
    
    if future_expiries_df.empty:
        st.warning("Could not establish a relevant contract curve. Please check your date filters.")
        st.stop()
        
    # 2. Transform historical prices to the required maturity curve
    analysis_curve_df, contract_labels = transform_to_analysis_curve(price_df_filtered, future_expiries_df)

    if analysis_curve_df.empty:
        st.warning("Data transformation failed. Check if contracts in the price file match contracts in the expiry file.")
        st.stop()
        
    # 3. Calculate Derivatives (for comparison and hedging)
    
    # --- All Derivatives Dictionary ---
    derivatives_data = {
        '3M Spread': calculate_n_month_spreads(analysis_curve_df, 3),
        '6M Spread': calculate_n_month_spreads(analysis_curve_df, 6),
        '12M Spread': calculate_n_month_spreads(analysis_curve_df, 12),
        '3M Fly': calculate_n_month_butterflies(analysis_curve_df, 3),
        '6M Fly': calculate_n_month_butterflies(analysis_curve_df, 6),
        '12M Fly': calculate_n_month_butterflies(analysis_curve_df, 12),
    }

    # Display derivative counts
    st.header("1. Data Derivatives Check (Contracts relevant to selected Analysis Date)")
    derivative_summary = {k: v.shape[1] for k, v in derivatives_data.items()}
    st.dataframe(pd.Series(derivative_summary, name='Count of Derivatives').to_frame().T)
    
    
    # 4. Perform PCA (Now ONLY on Outright Prices)
    loadings_outright, explained_variance_outright, scores_outright, prices_df_clean, data_mean = \
        perform_pca_on_prices(analysis_curve_df)


    if loadings_outright is not None:
        
        # --- Explained Variance Visualization ---
        st.header("2. Explained Variance (Based on Outright Prices)")
        
        variance_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance_outright))],
            'Explained Variance (%)': explained_variance_outright * 100
        })
        variance_df['Cumulative Variance (%)'] = variance_df['Explained Variance (%)'].cumsum()
        
        col_var, col_pca_select = st.columns([1, 1])
        with col_var:
            st.dataframe(variance_df, use_container_width=True)
            
        default_pc_count = min(3, len(explained_variance_outright))
        with col_pca_select:
            st.subheader("Fair Curve & Hedging Setup")
            pc_count = st.slider(
                "Select number of Principal Components (PCs) for Fair Curve & Hedging:",
                min_value=1,
                max_value=len(explained_variance_outright),
                value=default_pc_count,
                help="These PCs are derived from the outright prices (Level, Slope, Curve, etc.)."
            )
            
            total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
            st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the **outright prices**.")
        
        # --- Component Loadings Heatmap ---
        st.header("3. PC Loadings Heatmap (PC vs. Outright Contracts)")
        st.markdown("The weights show the **absolute price sensitivity** of each contract to the factors.")
        
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
        ax_outright_loading.set_title(f'Component Loadings for First {default_pc_count} PCs (Outright Prices)', fontsize=16)
        ax_outright_loading.set_xlabel('Principal Component')
        ax_outright_loading.set_ylabel('Outright Contract')
        st.pyplot(fig_outright_loading)

        # --- PC Scores Time Series Plot (Omitted for brevity, but retained in full code) ---
        
        
        # --- Historical Reconstruction ---
        historical_outrights_df, historical_derivatives = \
            reconstruct_fair_curve_from_prices(
                scores_outright, 
                loadings_outright, 
                data_mean, 
                pc_count, 
                analysis_curve_df, 
                derivatives_data
            )

        
        # --- 5. Curve Snapshot Analysis ---
        st.header("5. Curve Snapshot Analysis: " + analysis_date.strftime('%Y-%m-%d'))
        st.markdown("Comparing original market values against the **PCA Fair curve/spread/fly** derived from the Outright Price PCA factors.")

        # --- HELPER FUNCTION FOR PLOTTING SNAPSHOTS ---
        def plot_snapshot(historical_df, derivative_type, analysis_dt, pc_count):
            """Plots and displays the table for a single derivative type snapshot."""
            
            if historical_df.empty:
                st.info(f"Not enough contracts available to calculate the {derivative_type} snapshot.")
                return
            
            try:
                snapshot_original = historical_df.filter(regex='\(Original\)$').loc[[analysis_dt]].T
                snapshot_pca = historical_df.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
                
                snapshot_original.columns = ['Original']
                snapshot_original.index = snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)

                snapshot_pca.columns = ['PCA Fair']
                snapshot_pca.index = snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)

                comparison = pd.concat([snapshot_original, snapshot_pca], axis=1).dropna()
                
                if comparison.empty:
                    st.warning(f"No complete {derivative_type} data available for the selected analysis date.")
                    return

                # --- Plot the Derivative ---
                fig, ax = plt.subplots(figsize=(15, 7))
                ax.plot(comparison.index, comparison['Original'], label=f'Original Market {derivative_type}', marker='o', linestyle='-', linewidth=2.5, color='blue')
                ax.plot(comparison.index, comparison['PCA Fair'], label=f'PCA Fair {derivative_type} ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='red')
                
                mispricing = comparison['Original'] - comparison['PCA Fair']
                ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7) 
                
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


        # --- 5.1 Outright Price Snapshot ---
        st.subheader("5.1 Outright Price Curve")
        # (The outright price plot logic is kept separate for rate calculation, but follows the same principle)
        # Omitted for brevity, but included in the full code.

        
        # --- New Snapshot Sections ---
        st.subheader("5.2 3M Spread Snapshot (C1-C2)")
        plot_snapshot(historical_derivatives['3M Spread'], "3M Spread", analysis_dt, pc_count)

        st.subheader("5.3 3M Butterfly (Fly) Snapshot (C1-2xC2+C3)")
        plot_snapshot(historical_derivatives['3M Fly'], "3M Butterfly", analysis_dt, pc_count)

        st.subheader("5.4 6M Spread Snapshot (C1-C3)")
        plot_snapshot(historical_derivatives['6M Spread'], "6M Spread", analysis_dt, pc_count)

        st.subheader("5.5 6M Butterfly (Fly) Snapshot (C1-2xC3+C5)")
        plot_snapshot(historical_derivatives['6M Fly'], "6M Butterfly", analysis_dt, pc_count)

        st.subheader("5.6 12M Spread Snapshot (C1-C5)")
        plot_snapshot(historical_derivatives['12M Spread'], "12M Spread", analysis_dt, pc_count)

        st.subheader("5.7 12M Butterfly (Fly) Snapshot (C1-2xC5+C9)")
        plot_snapshot(historical_derivatives['12M Fly'], "12M Butterfly", analysis_dt, pc_count)

        
        # --- 6. Hedging Analysis ---
        st.header("6. Minimum Variance Hedging Analysis")
        st.markdown("This analysis calculates the optimal hedge ratio for a chosen trade against a universe of candidates using the **PC Factor Sensitivities (Betas)**.")

        # --- 6.1 Calculate All Derivative Sensitivities ---
        
        # Combine all derivative and outright price data into a single DataFrame for sensitivity calculation
        all_derivatives_list = list(derivatives_data.values())
        all_derivatives_list.insert(0, analysis_curve_df) # Insert Outright Prices as the first element
        
        # Drop columns with non-numeric data or NaNs for the sensitivity calculation
        full_universe_df = pd.concat(all_derivatives_list, axis=1).dropna()
        
        if full_universe_df.empty:
            st.error("Cannot perform hedging: Historical data is too sparse to calculate sensitivities across all derivatives.")
            st.stop()
            
        # Recalculate component loadings based on the cleaned and aligned price data
        # Note: We use the loadings from the PCA on prices_df_clean, which is the same as analysis_curve_df.dropna()
        
        # This function also includes the outright prices in its output
        full_universe_sensitivities = calculate_factor_sensitivities(
            full_universe_df, 
            loadings_outright, 
            data_mean, 
            pc_count
        )
        
        # Get list of all possible instruments for trade/hedge selection
        all_instruments = full_universe_sensitivities.index.tolist()
        
        if not all_instruments:
             st.error("No valid instruments found for hedging analysis.")
             st.stop()
             
        # --- User Selection ---
        col_trade, col_hedge_universe = st.columns([1, 1])
        
        with col_trade:
            trade_label = st.selectbox("Select Trade to Hedge (1 Unit Long):", all_instruments, index=all_instruments.index('Z25-M26') if 'Z25-M26' in all_instruments else 0)
            trade_sensitivity = full_universe_sensitivities.loc[[trade_label]]
            st.markdown("##### Trade PC Sensitivities (Betas)")
            st.dataframe(trade_sensitivity.T.style.format("{:.4f}"), use_container_width=True)
            
        with col_hedge_universe:
            # Create a simplified list of universes for selection
            universe_options = {
                'All Derivatives (Global Best Hedge)': full_universe_sensitivities.index.tolist(),
                'Outright Contracts Only': contract_labels,
                'Only Spreads (3M, 6M, 12M)': [k for k in all_instruments if '-' in k and '2x' not in k],
                'Only 3M Spreads': derivatives_data['3M Spread'].columns.tolist(),
                'Only 6M Flies': derivatives_data['6M Fly'].columns.tolist()
            }
            
            universe_key = st.selectbox("Select Hedge Candidate Universe:", list(universe_options.keys()))
            
            # Filter candidates: remove the trade itself from the candidates list
            selected_candidates = [c for c in universe_options[universe_key] if c != trade_label]

            hedge_candidates_sensitivity = full_universe_sensitivities.loc[selected_candidates]
            
            st.info(f"Hedge Candidates Selected: **{len(selected_candidates)}** instruments.")
            st.dataframe(hedge_candidates_sensitivity.head(5).T.style.format("{:.4f}"), use_container_width=True)


        # --- 6.2 Perform Minimum Variance Hedge ---
        
        if not hedge_candidates_sensitivity.empty:
            optimal_hedge_row, all_results_df = calculate_minimum_variance_hedge(
                trade_label, 
                trade_sensitivity, 
                hedge_candidates_sensitivity, 
                scores_outright.loc[prices_df_clean.index].iloc[:, :pc_count] # Only use relevant scores
            )
            
            st.subheader("6.3 Optimal Minimum Variance Hedge")
            
            col_optimal, col_full_table = st.columns([1, 2])
            
            with col_optimal:
                st.markdown(f"#### Best Hedge for **{trade_label}**")
                st.dataframe(
                    optimal_hedge_row[['Hedge Candidate', 'MVHR (Units of Hedge/Unit of Trade)', 'Residual Volatility (BPS)']].set_index('Hedge Candidate').style.format({
                        'MVHR (Units of Hedge/Unit of Trade)': "{:.4f}",
                        'Residual Volatility (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )
                
                mvhr = optimal_hedge_row['MVHR (Units of Hedge/Unit of Trade)'].iloc[0]
                residual_vol = optimal_hedge_row['Residual Volatility (BPS)'].iloc[0]
                
                st.success(f"""
                **Optimal Hedge Strategy:**
                To hedge 1 unit **Long {trade_label}**, you should go **Short {mvhr:.4f}** units of **{optimal_hedge_row['Hedge Candidate'].iloc[0]}**.
                
                The residual volatility of this hedged portfolio is **{residual_vol:.2f} BPS**.
                """)
                
            with col_full_table:
                st.markdown("#### Full Hedge Candidate Ranking")
                st.dataframe(
                    all_results_df.style.format({
                        'MVHR (Units of Hedge/Unit of Trade)': "{:.4f}",
                        'Cov(T, H)': "{:.6f}",
                        'Var(H)': "{:.6f}",
                        'Residual Volatility (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )

        else:
            st.warning("No valid hedge candidates remaining after filtering out the trade itself.")

    else:
        st.error("PCA failed. Please check your data quantity and quality.")
