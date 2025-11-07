import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
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
        return None, None, None, None

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
    explained_variance = pca.explained_variance_ratio_
    scores = pd.DataFrame(
        pca.transform(data_scaled),
        index=data_df_clean.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings, explained_variance, scores, data_df_clean

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
    
    NOTE: FIX applied here to correctly parse the generalized k-step butterfly label
    (e.g., Z25-2xM26+Z26) which caused the IndexError in the previous attempt.
    """
    if original_df.empty:
        return pd.DataFrame()

    # Align the original data index with the reconstructed prices index
    original_df_aligned = original_df.loc[reconstructed_prices.index]
    reconstructed_data = {}
    
    for label in original_df_aligned.columns:
        
        if derivative_type == 'spread':
            # Spread: C_i - C_{i+k}. Label is C_i-C_{i+k} (e.g., Z25-M26)
            c1, c_long = label.split('-')
            reconstructed_data[label + ' (PCA)'] = (
                reconstructed_prices[c1 + ' (PCA)'] - reconstructed_prices[c_long + ' (PCA)']
            )
        
        elif derivative_type == 'fly':
            # Fly: C_i - 2 * C_{i+k} + C_{i+2k}. Label format: C_i-2xC_{i+k}+C_{i+2k}
            # Example: Z25-2xM26+Z26
            
            try:
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
                    reconstructed_prices[c1 + ' (PCA)'] - 
                    2 * reconstructed_prices[c2_label + ' (PCA)'] + 
                    reconstructed_prices[c3_label + ' (PCA)']
                )
            except Exception as e:
                 # Skip if reconstruction fails due to malformed label or missing price
                 print(f"Skipping fly reconstruction for {label}: Error {e}")
                 continue 
    
    reconstructed_df = pd.DataFrame(reconstructed_data, index=reconstructed_prices.index)
    
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
    # The reconstructed prices DF can now be used to reconstruct all k-step derivatives, 
    # as they are simply functions of the outright prices.
    
    historical_spreads_3M = _reconstruct_derivative(spreads_3M_df, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_3M = _reconstruct_derivative(butterflies_3M_df, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_6M = _reconstruct_derivative(spreads_6M_df, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_6M = _reconstruct_derivative(butterflies_6M_df, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_12M = _reconstruct_derivative(spreads_12M_df, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_12M = _reconstruct_derivative(butterflies_12M_df, reconstructed_prices_df, derivative_type='fly')
    
    return historical_outrights, historical_spreads_3M, historical_butterflies_3M, historical_spreads_6M, historical_butterflies_6M, historical_spreads_12M, historical_butterflies_12M


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
    spreads_3M_df = calculate_k_step_spreads(analysis_curve_df, 1)
    butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, 1)
    
    # 6M (k=2) - Requested sections 5.4, 5.5
    spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, 2)
    butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, 2)
    
    # 12M (k=4) - Requested sections 5.6, 5.7
    spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, 4)
    butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, 4)
    
    st.markdown("##### 3-Month Outright Spreads (k=1, e.g., Z25-H26)")
    st.dataframe(spreads_3M_df.head(5))
    
    if spreads_3M_df.empty:
        st.warning("3M Spreads could not be calculated. Need at least two contracts in the analysis curve.")
        st.stop()
        
    # 4. Perform PCA
    # 4a. PCA on 3M Spreads (Standard Method - Used for Fair Curve Reconstruction)
    loadings_spread, explained_variance, scores, spreads_3M_df_clean = perform_pca(spreads_3M_df)
    
    # 4b. PCA on Outright Prices (User Requested Independent Method - Unstandardized/Covariance)
    loadings_outright_direct, explained_variance_outright_direct = perform_pca_on_prices(analysis_curve_df)


    if loadings_spread is not None and loadings_outright_direct is not None:
        
        # --- Explained Variance Visualization ---
        st.header("2. Explained Variance")
        variance_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Explained Variance (%)': explained_variance * 100
        })
        variance_df['Cumulative Variance (%)'] = variance_df['Explained Variance (%)'].cumsum()
        
        col_var, col_pca_select = st.columns([1, 1])
        with col_var:
            st.dataframe(variance_df, use_container_width=True)
            
        default_pc_count = min(3, len(explained_variance))
        with col_pca_select:
            st.subheader("Fair Curve Setup")
            pc_count = st.slider(
                "Select number of Principal Components (PCs) for Fair Curve:",
                min_value=1,
                max_value=len(explained_variance),
                value=default_pc_count,
                key='pc_slider'
            )
            total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
            st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the spreads.")
        
        
        # --- Component Loadings Heatmaps ---
        st.header("3. PC Loadings")
        
        # --- 3.1 Spread Loadings (Standard Method) ---
        st.subheader("3.1 PC Loadings Heatmap (PC vs. 3M Spreads)")
        st.markdown("""
            This heatmap shows the **Loadings (Eigenvectors)** of the first few PCs on each **3-Month Spread**. These weights are derived from **Standardized PCA** and represent how each spread contributes to the overall risk factors (Level, Slope, Curvature).
            
            * **Interpretation of Loadings (Weights):** The value of the loading (weight) indicates the **sensitivity** of that specific spread to the respective Principal Component. A high absolute value means the spread has historically been highly correlated with the movement of that PC factor.
            * **Example: PC1 (Level/Parallel Shift):** Loadings are typically all **positive and uniform** across most spreads. This means that when the PC1 score increases, **all spreads tend to move in the same direction** (e.g., the entire forward curve shifts up or down in a parallel manner).
            * **Example: PC2 (Slope/Twist):** Loadings are typically **negative for short-end spreads** and **positive for long-end spreads**. This means that when the PC2 score increases, the **short-end flattens** while the **long-end steepens**, causing the overall slope of the curve to change.
            * **Example: PC3 (Curvature/Bow):** Loadings show a **pattern of positive/negative/positive** or similar inversions, representing a 'bowing' or 'humping' of the curve (e.g., short and long ends move in one direction while the belly moves in the opposite).
        """)
        
        plt.style.use('default') 
        fig_spread_loading, ax_spread_loading = plt.subplots(figsize=(12, 6))
        
        loadings_spread_plot = loadings_spread.iloc[:, :default_pc_count]

        sns.heatmap(
            loadings_spread_plot, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            linewidths=0.5, 
            linecolor='gray', 
            cbar_kws={'label': 'Loading Weight'}
        )
        ax_spread_loading.set_title(f'3.1 Component Loadings for First {default_pc_count} Principal Components (on Spreads)', fontsize=16)
        ax_spread_loading.set_xlabel('Principal Component')
        ax_spread_loading.set_ylabel('Spread Contract')
        st.pyplot(fig_spread_loading)
        
        
        # --- 3.2 Outright Loadings (User Requested Independent Method) ---
        st.subheader("3.2 PC Loadings Heatmap (PC vs. Outright Contracts - Absolute Sensitivity)")
        
        pc1_outright_variance = explained_variance_outright_direct[0] * 100
        st.markdown(f"""
            This heatmap shows the **independent sensitivity** of each **outright contract price** to the principal components. This result is based on **Unstandardized PCA (Covariance Matrix)**, meaning the weights reflect the **absolute historical price volatility and duration** of each contract.
            
            * **Interpretation of Loadings (Weights):** The value here represents the raw eigenvector weight, indicating the **absolute magnitude** of historical price movement for that contract that is captured by the PC.
            * **Example: PC1 (Non-Uniform Level):** The loadings are highly **non-uniform**, increasing significantly with maturity. The weight on the shortest contract is small, while the weight on the longest contract is largest. This happens because **long-dated contracts have historically experienced larger absolute price moves** (higher duration/volatility) than short-dated contracts, and unstandardized PCA captures this absolute movement.
            
            **PC1 Explained Variance (Absolute Price):** **{pc1_outright_variance:.2f}%**
        """)
        
        fig_outright_loading, ax_outright_loading = plt.subplots(figsize=(12, 6))
        
        loadings_outright_plot = loadings_outright_direct.iloc[:, :default_pc_count]

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
        ax_outright_loading.set_title(f'3.2 Component Loadings for First {default_pc_count} PCs (Unstandardized Outright Prices)', fontsize=16)
        ax_outright_loading.set_xlabel('Principal Component')
        ax_outright_loading.set_ylabel('Outright Contract')
        st.pyplot(fig_outright_loading)
        
        
        # --- PC Scores Time Series Plot ---
        def plot_pc_scores(scores_df, explained_variance):
            """Plots the time series of the first 3 PC scores."""
            
            pc_labels = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
            num_pcs = min(3, scores_df.shape[1])
            if num_pcs == 0: return None

            fig, axes = plt.subplots(nrows=num_pcs, ncols=1, figsize=(15, 4 * num_pcs), sharex=True)
            if num_pcs == 1: axes = [axes] 

            plt.suptitle("Time Series of Principal Component Scores (Risk Factors)", fontsize=16, y=1.02)

            for i in range(num_pcs):
                ax = axes[i]
                pc_label = pc_labels[i]
                variance_pct = explained_variance[i] * 100
                
                ax.plot(scores_df.index, scores_df.iloc[:, i], label=f'{pc_label} ({variance_pct:.2f}% Var.)', linewidth=1.5, color=plt.cm.tab10(i))
                
                ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
                
                ax.set_title(f'{pc_label} Factor Score (Explaining {variance_pct:.2f}% of Spread Variance)', fontsize=14)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_ylabel('Score Value')
                ax.legend(loc='upper left')
                
            plt.xlabel('Date')
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            return fig

        st.header("4. PC Factor Scores Time Series")
        st.markdown("This plot shows the historical movement of the **latent risk factors** (Level, Slope, and Curvature) over the chosen historical range. The scores are derived from the **Spread PCA (3.1)**.")
        fig_scores = plot_pc_scores(scores, explained_variance)
        if fig_scores:
            st.pyplot(fig_scores)
            
        
        # --- Historical Reconstruction (Based on Spread PCA) ---
        
        # 1. Reconstruct 3M Spreads using only selected PCs
        data_mean = spreads_3M_df_clean.mean()
        data_std = spreads_3M_df_clean.std()
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

        
        # --- HELPER FUNCTION FOR PLOTTING SNAPSHOTS ---
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
            curve_snapshot_original = historical_outrights_df.filter(regex='\(Original\)$').loc[[analysis_dt]].T
            curve_snapshot_pca = historical_outrights_df.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
            
            curve_snapshot_original.columns = ['Original']
            curve_snapshot_original.index = curve_snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)

            curve_snapshot_pca.columns = ['PCA Fair']
            curve_snapshot_pca.index = curve_snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)

            curve_comparison = pd.concat([curve_snapshot_original, curve_snapshot_pca], axis=1).dropna()
            
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
            
    else:
        st.error("PCA failed. Please check your data quantity and quality.")
