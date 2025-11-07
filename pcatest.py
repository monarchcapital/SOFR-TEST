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
    The column names are kept as the original contract codes (e.g., Z20, H21).
    """
    if price_df is None or future_expiries_df.empty:
        return pd.DataFrame(), []

    contract_order = future_expiries_df.index.tolist()
    
    # Filter price columns to only include those present in the contract order
    valid_contracts = [c for c in contract_order if c in price_df.columns]
    
    if not valid_contracts:
        st.warning("No matching contract columns found in price data for the selected analysis date range.")
        return pd.DataFrame(), []

    # Filter the price data to only include valid, ordered contract columns
    analysis_curve_df = price_df[valid_contracts]
    
    return analysis_curve_df, valid_contracts

def calculate_outright_spreads(analysis_curve_df):
    """
    Calculates the first differences (spreads) on a CME basis: C1 - C2, C2 - C3, etc.
    The labels now use the contract codes (e.g., Z20-H21).
    """
    if analysis_curve_df.empty:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    spreads_data = {}
    
    for i in range(num_contracts - 1):
        # CME Basis: Shorter maturity minus longer maturity
        short_maturity = analysis_curve_df.columns[i]
        long_maturity = analysis_curve_df.columns[i+1]
        
        spread_label = f"{short_maturity}-{long_maturity}"
        
        spreads_data[spread_label] = analysis_curve_df.iloc[:, i] - analysis_curve_df.iloc[:, i+1]
        
    return pd.DataFrame(spreads_data)

def calculate_butterflies(analysis_curve_df):
    """
    Calculates butterflies (flies) on a CME basis: (C1 - C2) - (C2 - C3) = C1 - 2*C2 + C3, etc.
    The labels now use the contract codes (e.g., Z20-2xH21+M21).
    """
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < 3:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    flies_data = {}

    for i in range(num_contracts - 2):
        short_maturity = analysis_curve_df.columns[i]    # C1
        center_maturity = analysis_curve_df.columns[i+1] # C2
        long_maturity = analysis_curve_df.columns[i+2]   # C3

        # Fly = C1 - 2*C2 + C3
        fly_label = f"{short_maturity}-2x{center_maturity}+{long_maturity}"

        flies_data[fly_label] = analysis_curve_df.iloc[:, i] - 2 * analysis_curve_df.iloc[:, i+1] + analysis_curve_df.iloc[:, i+2]

    return pd.DataFrame(flies_data)
# --- NEW HELPER FUNCTION: Calculate Custom Spreads ---
def calculate_custom_spreads(analysis_curve_df, contract_labels, months_span):
    """
    Calculates spreads based on a specific month span (e.g., 6 or 12 months).
    Example: 6-month spread is C(t) - C(t+6m).
    """
    if analysis_curve_df.empty or len(contract_labels) < 2:
        return pd.DataFrame()

    num_contracts = len(contract_labels)
    custom_spreads_data = {}
    
    # Determine the contract offset based on the CME convention (4 contracts per year, 3 months between)
    if months_span == 6:
        offset = 2 # 6 months = 2 contracts
    elif months_span == 12:
        offset = 4 # 12 months = 4 contracts
    else:
        return pd.DataFrame() # Unsupported span

    
    for i in range(num_contracts):
        # The 'short' contract is at index i
        short_maturity = contract_labels[i]
        
        # The 'long' contract is at index i + offset
        long_index = i + offset
        
        if long_index < num_contracts:
            long_maturity = contract_labels[long_index]
            
            # CME Basis: Shorter maturity minus longer maturity
            spread_label = f"{short_maturity}-{long_maturity}"
            
            custom_spreads_data[spread_label] = analysis_curve_df[short_maturity] - analysis_curve_df[long_maturity]
            
    return pd.DataFrame(custom_spreads_data)

# --- NEW HELPER FUNCTION: Calculate Custom Butterflies ---
def calculate_custom_butterflies(analysis_curve_df, contract_labels, months_span):
    """
    Calculates butterflies based on a specific month span (e.g., 6-month fly: C(t) - 2*C(t+6m) + C(t+12m)).
    """
    if analysis_curve_df.empty or len(contract_labels) < 3:
        return pd.DataFrame()

    num_contracts = len(contract_labels)
    flies_data = {}
    
    if months_span == 6:
        offset = 2 # 6 months = 2 contracts (Center contract offset)
    elif months_span == 12:
        offset = 4 # 12 months = 4 contracts (Center contract offset)
    else:
        return pd.DataFrame() # Unsupported span
    
    # The fly requires 3 contracts: C1, C2 (C1 + offset), C3 (C2 + offset = C1 + 2*offset)
    required_span_offset = 2 * offset # Total index span needed: 4 for 6-month fly, 8 for 12-month fly

    for i in range(num_contracts):
        # C1 (Short) is at index i
        index_c1 = i
        
        # C2 (Center) is at index i + offset
        index_c2 = i + offset
        
        # C3 (Long) is at index i + required_span_offset
        index_c3 = i + required_span_offset
        
        if index_c3 < num_contracts:
            c1_maturity = contract_labels[index_c1]
            c2_maturity = contract_labels[index_c2]
            c3_maturity = contract_labels[index_c3]

            # Fly = C1 - 2*C2 + C3
            fly_label = f"{c1_maturity}-2x{c2_maturity}+{c3_maturity}"

            flies_data[fly_label] = analysis_curve_df[c1_maturity] - 2 * analysis_curve_df[c2_maturity] + analysis_curve_df[c3_maturity]

    return pd.DataFrame(flies_data)
    
def perform_pca(data_df):
    """Performs PCA on the input DataFrame (expected to be spreads)."""
    # Drop rows with NaNs before standardization and PCA
    data_df_clean = data_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        st.error("Not enough complete data points (rows) to perform PCA on the spreads after dropping NaNs.")
        return None, None, None, None

    # Standardize the data (PCA is sensitive to scale)
    data_mean = data_df_clean.mean()
    data_std = data_df_clean.std()
    data_scaled = (data_df_clean - data_mean) / data_std
    
    # Determine optimal number of components (min(n_samples, n_features))
    n_components = min(data_scaled.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    
    # Component Loadings (the eigenvectors * sqrt(eigenvalues))
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    
    explained_variance = pca.explained_variance_ratio_
    
    # Principal Component Scores (the transformed data)
    scores = pd.DataFrame(
        pca.transform(data_scaled),
        index=data_df_clean.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings, explained_variance, scores, data_df_clean

# --- NEW HELPER FUNCTION: Calculate Z-Scores ---
def calculate_z_scores(scores_df, analysis_dt):
    """Calculates Z-scores for the latest date relative to the historical scores."""
    if scores_df.empty:
        return None, None
        
    scores_mean = scores_df.mean()
    scores_std = scores_df.std()
    
    # Calculate Z-scores for the entire historical period
    z_scores_historical = (scores_df - scores_mean) / scores_std
    
    # Get the Z-score for the specific analysis date
    try:
        # Use a list slice [[analysis_dt]] to ensure a DataFrame is returned even if the date is exactly matched
        current_z_score = z_scores_historical.loc[[analysis_dt]].T
        return z_scores_historical, current_z_score
    except KeyError:
        # If the exact date is not present, find the nearest one
        try:
            # Find the nearest date label
            nearest_date_label = (z_scores_historical.index - analysis_dt).to_series().abs().idxmin()
            current_z_score = z_scores_historical.loc[[nearest_date_label]].T
            return z_scores_historical, current_z_score
        except Exception:
            return z_scores_historical, None

# --- NEW HELPER FUNCTION: Calculate Outright Loadings ---
def calculate_outright_loadings(loadings_df, contract_labels):
    """
    Transforms spread loadings to outright price loadings (cumulative sum).
    Anchor (P1) is set to 1.
    """
    if loadings_df.empty or not contract_labels:
        return pd.DataFrame()
        
    # Initialize a DataFrame for outright loadings
    outright_loadings = pd.DataFrame(
        index=contract_labels, 
        columns=loadings_df.columns
    )
    
    for pc_col in loadings_df.columns:
        # P1 is the anchor, its loading is 1.0 (assuming a 1-to-1 impact of Level factor)
        outright_loadings.loc[contract_labels[0], pc_col] = 1.0 
        
        current_loading = 1.0 # P1 loading
        
        # Iterate through spread loadings (S1, S2, ...)
        for i in range(loadings_df.shape[0]): 
            spread_loading = loadings_df.iloc[i][pc_col]
            
            # P_long = P_short - S. Thus, L_P(i+1) = L_P(i) - L_S(i, i+1)
            # The loading of the next contract is the previous contract's loading minus the spread loading
            current_loading -= spread_loading 
            
            # The contract being calculated is P(i+1), which is at index i+1 in contract_labels
            outright_loadings.loc[contract_labels[i+1], pc_col] = current_loading

    return outright_loadings.astype(float)


def reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_df, derivatives_to_reconstruct, pc_count):
    """
    Reconstructs Outright Prices and Derivatives historically using a general PCA factor model.
    The outright price reconstruction is anchored to the original nearest contract price path (Level factor).
    The function handles reconstruction for outright spreads, butterflies, and custom spreads/butterflies.
    """
    
    # Align the analysis curve data with the PCA data (which drops NaNs)
    analysis_curve_df_aligned = analysis_curve_df.loc[reconstructed_spreads_df.index]
    
    # --- 1. Reconstruct Outright Prices (Anchored to Front) ---
    
    nearest_contract_original = analysis_curve_df_aligned.iloc[:, 0]
    nearest_contract_label = analysis_curve_df_aligned.columns[0]
    
    reconstructed_prices_df = pd.DataFrame(index=analysis_curve_df_aligned.index)
    reconstructed_prices_df[nearest_contract_label + ' (PCA)'] = nearest_contract_original
    
    # Iterate through all maturities starting from the second contract (index 1)
    for i in range(1, len(analysis_curve_df_aligned.columns)):
        prev_maturity = analysis_curve_df_aligned.columns[i-1]
        current_maturity = analysis_curve_df_aligned.columns[i]
        spread_label = f"{prev_maturity}-{current_maturity}"
        
        if spread_label in reconstructed_spreads_df.columns:
            # P_i = P_i-1 (PCA) - S_i-1,i (PCA)
            reconstructed_prices_df[current_maturity + ' (PCA)'] = (
                reconstructed_prices_df[prev_maturity + ' (PCA)'] - reconstructed_spreads_df[spread_label]
            )
        
    original_price_rename = {col: col + ' (Original)' for col in analysis_curve_df_aligned.columns}
    original_prices_df = analysis_curve_df_aligned.rename(columns=original_price_rename)
    historical_outrights = pd.merge(original_prices_df, reconstructed_prices_df, left_index=True, right_index=True)


    # --- 2. Reconstruct General Derivatives (Spreads and Flies) ---
    historical_derivatives = {}
    
    # For each derivative type passed (e.g., outright spreads, 6-month spreads, 12-month flies)
    for derivative_type, original_df in derivatives_to_reconstruct.items():
        if original_df.empty:
            historical_derivatives[derivative_type] = pd.DataFrame()
            continue

        # Align the derivative data with the PCA index (reconstructed_spreads_df index)
        original_df_aligned = original_df.loc[analysis_curve_df_aligned.index]
        
        # Calculate the PCA Fair value for the derivative using the outright price reconstruction
        if derivative_type == 'Outright Spreads':
            # Outright Spreads (S_i-1,i) are simply the reconstructed spreads from step 1
            # We must align the columns to ensure they match exactly
            spread_cols = [col for col in original_df_aligned.columns if col in reconstructed_spreads_df.columns]
            reconstructed_deriv = reconstructed_spreads_df.loc[original_df_aligned.index, spread_cols]
            
        elif 'Spread' in derivative_type:
            # Calculate custom spread from reconstructed outright prices
            reconstructed_deriv = pd.DataFrame(index=original_df_aligned.index)
            for col in original_df_aligned.columns:
                # Assuming spread label format is 'Cshort-Clong'
                try:
                    short, long = col.split('-')
                except ValueError: 
                    continue 

                # Reconstructed Spread = P_short(PCA) - P_long(PCA)
                if f'{short} (PCA)' in reconstructed_prices_df.columns and f'{long} (PCA)' in reconstructed_prices_df.columns:
                    reconstructed_deriv[col] = reconstructed_prices_df[short + ' (PCA)'] - reconstructed_prices_df[long + ' (PCA)']
                
        elif 'Butterfly' in derivative_type:
             # Calculate custom fly from reconstructed outright prices
            reconstructed_deriv = pd.DataFrame(index=original_df_aligned.index)
            for col in original_df_aligned.columns:
                # Assuming fly label format is 'C1-2xC2+C3'
                try:
                    parts = col.split('-2x')
                    c1 = parts[0]
                    parts2 = parts[1].split('+')
                    c2 = parts2[0]
                    c3 = parts2[1]
                except (IndexError, ValueError):
                    continue
                
                # Reconstructed Fly = P_C1(PCA) - 2*P_C2(PCA) + P_C3(PCA)
                if all(f'{c} (PCA)' in reconstructed_prices_df.columns for c in [c1, c2, c3]):
                    reconstructed_deriv[col] = (
                        reconstructed_prices_df[c1 + ' (PCA)'] - 
                        2 * reconstructed_prices_df[c2 + ' (PCA)'] + 
                        reconstructed_prices_df[c3 + ' (PCA)']
                    )
        
        else:
            continue
            
        # Drop columns in the reconstructed derivatives where the calculation might have failed due to missing outright prices
        reconstructed_deriv = reconstructed_deriv.dropna(axis=1, how='all')

        # Filter the original dataframe to only include columns that were successfully reconstructed
        original_df_renamed = original_df_aligned[reconstructed_deriv.columns].rename(
            columns={col: col + ' (Original)' for col in reconstructed_deriv.columns}
        )
        pca_df_renamed = reconstructed_deriv.rename(
            columns={col: col + ' (PCA)' for col in reconstructed_deriv.columns}
        )
        
        historical_derivatives[derivative_type] = pd.merge(original_df_renamed, pca_df_renamed, left_index=True, right_index=True)

    # Return the three standard outputs plus the full map of all derivatives
    return historical_outrights, historical_derivatives.get('Outright Spreads', pd.DataFrame()), historical_derivatives.get('Outright Butterflies', pd.DataFrame()), historical_derivatives


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
    
    # Ensure start_date and end_date are used as datetime objects for filtering
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    
    price_df_filtered = price_df[(price_df.index >= start_dt) & (price_df.index <= end_dt)]
    
    # --- Analysis Date Selector (Maturity Roll) ---
    st.sidebar.header("3. Curve Analysis Date")
    
    # Analysis date should be within the historical range for stability
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
    
    # Ensure analysis_date is a datetime object for comparison
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
        
    # 3. Calculate Spreads and Butterflies (Inputs for PCA and comparison)
    st.header("1. Data Derivatives Check (Contracts relevant to selected Analysis Date)")
    
    # Calculate Spreads
    spreads_df = calculate_outright_spreads(analysis_curve_df)
    st.markdown("##### Outright Spreads (e.g., Z20-H21, H21-M21, etc.)")
    
    # Display logic adjusted to show data around analysis date for better context
    try:
        idx = spreads_df.index.get_loc(analysis_dt, method='nearest')
        start_idx = max(0, idx - 2)
        end_idx = min(len(spreads_df), idx + 3)
        st.dataframe(spreads_df.iloc[start_idx:end_idx].fillna('None'))
        st.caption(f"Showing spreads around the Analysis Date: {analysis_date.strftime('%Y-%m-%d')}")
    except Exception:
        st.dataframe(spreads_df.head(5).fillna('None'))
        st.caption("Showing latest 5 dates (Date lookup failed).")

    
    if spreads_df.empty:
        st.warning("Spreads could not be calculated. Need at least two contracts in the analysis curve.")
        st.stop()
        
    # Calculate Butterflies
    butterflies_df = calculate_butterflies(analysis_curve_df)
    st.markdown("##### Butterflies (e.g., Z20-2xH21+M21, etc.)")
    
    # Display logic adjusted to show data around analysis date for better context
    try:
        idx = butterflies_df.index.get_loc(analysis_dt, method='nearest')
        start_idx = max(0, idx - 2)
        end_idx = min(len(butterflies_df), idx + 3)
        st.dataframe(butterflies_df.iloc[start_idx:end_idx].fillna('None'))
        st.caption(f"Showing butterflies around the Analysis Date: {analysis_date.strftime('%Y-%m-%d')}")
    except Exception:
        st.dataframe(butterflies_df.head(5).fillna('None'))
        st.caption("Showing latest 5 dates (Date lookup failed).")


    # 4. Perform PCA on Spreads
    # PCA is performed only on spreads, as they are the most stationary features
    loadings, explained_variance, scores, spreads_df_clean = perform_pca(spreads_df)

    if loadings is not None:
        
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
            
        # Determine how many components to use for fair curve reconstruction
        default_pc_count = min(3, len(explained_variance))
        with col_pca_select:
            st.subheader("Fair Curve Setup")
            pc_count = st.slider(
                "Select number of Principal Components (PCs) for Fair Curve:",
                min_value=1,
                max_value=len(explained_variance),
                value=default_pc_count,
                help="Typically, the first 3 components (Level, Slope, Curve) explain over 95% of variance in spread changes."
            )
            
            total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
            st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the spreads.")
        
        
        # --- Component Loadings Heatmap (PC vs. Spreads) ---
        st.header("3. PC Loadings Heatmap (PC vs. Spreads)")
        st.markdown("""
            This heatmap shows the weights of the first few PCs on each **Spread**. These weights define the fundamental Level, Slope, and Curvature factors.
        """)
        
        plt.style.use('default') 
        fig, ax = plt.subplots(figsize=(12, 6))
        
        loadings_plot = loadings.iloc[:, :default_pc_count]

        sns.heatmap(
            loadings_plot, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            linewidths=0.5, 
            linecolor='gray', 
            cbar_kws={'label': 'Loading Weight'}
        )
        ax.set_title(f'Component Loadings for First {default_pc_count} Principal Components (Spreads)', fontsize=16)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Spread Contract')
        st.pyplot(fig)
        
        # ----------------------------------------------------------------------
        # --- MODIFIED SECTION 4: Outright Loadings Plots (Heatmap + Line) ---
        # ----------------------------------------------------------------------
        st.header("4. PC Loadings vs. Outright Contract Levels (e.g., Z25, H26)")
        outright_loadings_df = calculate_outright_loadings(loadings, contract_labels)
        
        if not outright_loadings_df.empty:
            
            num_factors_to_plot = min(3, outright_loadings_df.shape[1]) 
            loadings_plot = outright_loadings_df.iloc[:, :num_factors_to_plot]

            # --- NEW: Heatmap of Outright Loadings (PC vs. Contracts) ---
            st.markdown("###### Heatmap: PC Loadings vs. Outright Contracts")
            st.markdown("""
                This heatmap shows the raw **sensitivity** of each outright contract's price to a 1-unit move in the Level, Slope, and Curve factors.
            """)
            plt.style.use('default') 
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 6))
            
            # Transpose the data so contracts are on the X-axis (maturity axis)
            sns.heatmap(
                loadings_plot.T, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5, 
                linecolor='gray', 
                cbar_kws={'label': 'Loading Weight'},
                vmin=-1.0, vmax=1.0 # Standardize color scale
            )
            ax_heatmap.set_title(f'Component Loadings for Outright Contracts (First {num_factors_to_plot} PCs)', fontsize=16)
            ax_heatmap.set_xlabel('Contract Maturity')
            ax_heatmap.set_ylabel('Principal Component')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_heatmap)
            
            # --- Existing: Line Plot of Outright Loadings ---
            st.markdown("###### Line Plot: PC Impact on Outright Curve Shape")
            st.markdown("This plot illustrates the **shape** of the factor curve (how the loading changes across maturities).")
            
            plt.style.use('default') 
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i in range(num_factors_to_plot):
                pc_label = f'PC{i+1}'
                ax.plot(
                    outright_loadings_df.index, 
                    outright_loadings_df[pc_label], 
                    marker='o', 
                    linestyle='-', 
                    label=pc_label
                )
                
            ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
            ax.set_title(f'Impact of Factors (PC Loadings) on Outright Prices', fontsize=16)
            ax.set_xlabel('Contract Maturity')
            ax.set_ylabel('Outright Price Loading (Factor Sensitivity)')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.6)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        
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

        st.header("5. PC Factor Scores Time Series")
        st.markdown("This plot shows the historical movement of the **latent risk factors** (Level, Slope, and Curvature) over the chosen historical range.")
        fig_scores = plot_pc_scores(scores, explained_variance)
        if fig_scores:
            st.pyplot(fig_scores)
            
        # --- NEW SECTION: PC Factor Z-Scores ---
        st.header("6. PC Factor Z-Scores (Trading Signal)")
        
        z_scores_historical, current_z_score = calculate_z_scores(scores, analysis_dt)

        if current_z_score is not None:
            st.markdown(f"Z-Scores for the curve on **{current_z_score.columns[0].strftime('%Y-%m-%d')}** relative to the historical distribution (using nearest date):")
            
            # Use the first few columns (PCs) for display
            num_pcs_to_display = min(5, current_z_score.shape[0])
            
            st.dataframe(
                current_z_score.T.iloc[:, :num_pcs_to_display].style.applymap(
                    lambda x: 'background-color: #ffcccc' if abs(x) >= 2 else '', subset=current_z_score.index[:num_pcs_to_display]
                ).format("{:.2f}"),
                use_container_width=True
            )
            st.caption("A Z-score outside the $\pm 2.0$ range is often considered a statistical extreme (signal).")
        else:
             st.warning(f"Z-Score calculation skipped. Analysis Date {analysis_date.strftime('%Y-%m-%d')} not found or failed to match nearest date in PC Scores data.")
            
        
        # --- Historical Reconstruction ---
        
        # 1. Reconstruct Spreads using only selected PCs
        data_mean = spreads_df_clean.mean()
        data_std = spreads_df_clean.std()
        scores_aligned = scores.loc[spreads_df_clean.index]
        scores_used = scores_aligned.values[:, :pc_count]
        loadings_used = loadings.values[:, :pc_count]
        
        reconstructed_scaled = scores_used @ loadings_used.T
        
        reconstructed_spreads_df = pd.DataFrame(
            reconstructed_scaled * data_std.values + data_mean.values,
            index=spreads_df_clean.index, 
            columns=spreads_df_clean.columns
        )

        # 2. Calculate New Custom Derivatives (6-month and 12-month)
        spreads_6m_df = calculate_custom_spreads(analysis_curve_df, contract_labels, 6)
        spreads_12m_df = calculate_custom_spreads(analysis_curve_df, contract_labels, 12)
        flies_6m_df = calculate_custom_butterflies(analysis_curve_df, contract_labels, 6)
        flies_12m_df = calculate_custom_butterflies(analysis_curve_df, contract_labels, 12)

        derivatives_to_reconstruct = {
            'Outright Spreads': spreads_df,
            'Outright Butterflies': butterflies_df, # Standard 3-month flies
            '6-Month Spreads': spreads_6m_df,
            '12-Month Spreads': spreads_12m_df,
            '6-Month Butterflies': flies_6m_df,
            '12-Month Butterflies': flies_12m_df
        }

        # 3. Reconstruct All Products (Outright Prices, Spreads, Flies)
        # We pass the full map of all derivatives for general reconstruction
        historical_outrights_df, historical_spreads_df, historical_butterflies_df, historical_derivatives_map = \
            reconstruct_prices_and_derivatives(
                analysis_curve_df, 
                reconstructed_spreads_df, 
                derivatives_to_reconstruct,
                pc_count
            )

        
        
        # --- Cross-Sectional Curve Plot for Single Date ---
        st.header("7. Curve Snapshot Analysis: " + analysis_date.strftime('%Y-%m-%d'))
        st.markdown("This section plots the **current market values** (Original) against the **PCA Fair curve/spread/fly** for the selected date. The vertical difference is the theoretical mispricing.")

        # --- 7.1 Outright Price Snapshot ---
        st.subheader("7.1 Outright Price Curve")
        
        # Get the single-day snapshot for Outright Prices
        try:
            # 1. Select the single day's data, ensuring DataFrame structure
            curve_snapshot_original = historical_outrights_df.filter(regex='\(Original\)$').loc[[analysis_dt]].T
            curve_snapshot_pca = historical_outrights_df.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
            
            # 2. Rename column (which is the datetime key) and clean the index labels
            curve_snapshot_original.columns = ['Original']
            curve_snapshot_original.index = curve_snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)

            curve_snapshot_pca.columns = ['PCA Fair']
            curve_snapshot_pca.index = curve_snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)

            # 3. Concatenate and drop NaNs (if any value is missing for a contract)
            curve_comparison = pd.concat([curve_snapshot_original, curve_snapshot_pca], axis=1).dropna()
            
            if curve_comparison.empty:
                st.warning(f"No complete Outright Price data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')} after combining Original and PCA Fair values.")
            else:
                # --- Plot the Curve ---
                fig_curve, ax_curve = plt.subplots(figsize=(15, 7))
                
                # Plot Original Curve
                ax_curve.plot(curve_comparison.index, curve_comparison['Original'], 
                              label='Original Market Curve', marker='o', linestyle='-', linewidth=2.5, color='blue')
                
                # Plot PCA Fair Curve
                ax_curve.plot(curve_comparison.index, curve_comparison['PCA Fair'], 
                              label=f'PCA Fair Curve ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='red')
                
                # Plot Mispricing (Original - PCA Fair)
                mispricing = curve_comparison['Original'] - curve_comparison['PCA Fair']
                
                # Annotate the contracts with the largest absolute mispricing
                max_abs_mispricing = mispricing.abs().max()
                if max_abs_mispricing > 0:
                    mispricing_contract = mispricing.abs().idxmax()
                    mispricing_value = mispricing.loc[mispricing_contract] * 10000 # Convert to BPS
                    
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
                
                # Create the detailed table
                detailed_comparison = curve_comparison.copy()
                detailed_comparison.index.name = 'Contract'
                
                # Calculate Rates (Yields) and Mispricing in BPS
                detailed_comparison['Original Rate (%)'] = 100.0 - detailed_comparison['Original']
                detailed_comparison['PCA Fair Rate (%)'] = 100.0 - detailed_comparison['PCA Fair']
                # Mispricing in Price terms (Original Price - PCA Fair Price) * 10,000 to get BPS
                detailed_comparison['Mispricing (BPS)'] = (detailed_comparison['Original'] - detailed_comparison['PCA Fair']) * 10000

                # Reorder columns and rename Price columns
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

        
        # --- 7.2 Spread Snapshot ---
        st.subheader("7.2 Outright Spread Snapshot (e.g., Z20-H21)")
        historical_spreads_df = historical_derivatives_map.get('Outright Spreads')
        
        if historical_spreads_df.empty:
            st.warning("Outright Spreads could not be reconstructed. Skipping snapshot.")
        else:
            try:
                # 1. Select the single day's data, ensuring DataFrame structure
                spread_snapshot_original = historical_spreads_df.filter(regex='\(Original\)$').loc[[analysis_dt]].T
                spread_snapshot_pca = historical_spreads_df.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
                
                # 2. Rename column (which is the datetime key) and clean the index labels
                spread_snapshot_original.columns = ['Original']
                spread_snapshot_original.index = spread_snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)

                spread_snapshot_pca.columns = ['PCA Fair']
                spread_snapshot_pca.index = spread_snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)

                # 3. Concatenate and drop NaNs
                spread_comparison = pd.concat([spread_snapshot_original, spread_snapshot_pca], axis=1).dropna()
                
                if spread_comparison.empty:
                    st.warning(f"No complete Spread data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')} after combining Original and PCA Fair values.")
                else:
                    # --- Plot the Spreads ---
                    fig_spread, ax_spread = plt.subplots(figsize=(15, 7))
                    
                    # Plot Original Spread
                    ax_spread.plot(spread_comparison.index, spread_comparison['Original'], 
                                label='Original Market Spread', marker='o', linestyle='-', linewidth=2.5, color='darkgreen')
                    
                    # Plot PCA Fair Spread
                    ax_spread.plot(spread_comparison.index, spread_comparison['PCA Fair'], 
                                label=f'PCA Fair Spread ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='orange')
                    
                    # Plot Mispricing (Original - PCA Fair)
                    mispricing = spread_comparison['Original'] - spread_comparison['PCA Fair']
                    ax_spread.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7) # Add zero line for reference
                    
                    # Annotate the spread with the largest absolute mispricing
                    max_abs_mispricing = mispricing.abs().max()
                    if max_abs_mispricing > 0:
                        mispricing_spread = mispricing.abs().idxmax()
                        mispricing_value = mispricing.loc[mispricing_spread] * 10000 # Convert to BPS
                        
                        ax_spread.annotate(
                            f"Mispricing: {mispricing_value:.2f} BPS",
                            (mispricing_spread, spread_comparison.loc[mispricing_spread]['Original']),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                        )
                    
                    ax_spread.set_title('Market Spread vs. PCA Fair Spread', fontsize=16)
                    ax_spread.set_xlabel('Spread Contract (Short-Long)')
                    ax_spread.set_ylabel('Spread Value (Price Difference)')
                    ax_spread.legend(loc='upper right')
                    ax_spread.grid(True, linestyle=':', alpha=0.6)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig_spread)
                    
                    # --- Detailed Spread Table ---
                    st.markdown("###### Spread Mispricing")
                    detailed_comparison_spread = spread_comparison.copy()
                    detailed_comparison_spread.index.name = 'Spread Contract'
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


        # --- 7.3 Butterfly (Fly) Snapshot (Outright / 3-Month) ---
        st.subheader("7.3 Outright Butterfly (Fly) Snapshot (e.g., Z20-2xH21+M21)")
        historical_butterflies_df = historical_derivatives_map.get('Outright Butterflies')

        if historical_butterflies_df.empty:
            st.info("Not enough contracts (need 3 or more) to calculate or reconstruct the Outright Butterfly.")
        else:
            try:
                # 1. Select the single day's data, ensuring DataFrame structure
                fly_snapshot_original = historical_butterflies_df.filter(regex='\(Original\)$').loc[[analysis_dt]].T
                fly_snapshot_pca = historical_butterflies_df.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
                
                # 2. Rename column (which is the datetime key) and clean the index labels
                fly_snapshot_original.columns = ['Original']
                fly_snapshot_original.index = fly_snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)

                fly_snapshot_pca.columns = ['PCA Fair']
                fly_snapshot_pca.index = fly_snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)

                # 3. Concatenate and drop NaNs
                fly_comparison = pd.concat([fly_snapshot_original, fly_snapshot_pca], axis=1).dropna()
                
                if fly_comparison.empty:
                    st.warning(f"No complete Butterfly (Fly) data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')} after combining Original and PCA Fair values.")
                else:
                    # --- Plot the Flies ---
                    fig_fly, ax_fly = plt.subplots(figsize=(15, 7))
                    
                    # Plot Original Fly
                    ax_fly.plot(fly_comparison.index, fly_comparison['Original'], 
                                  label='Original Market Fly', marker='o', linestyle='-', linewidth=2.5, color='purple')
                    
                    # Plot PCA Fair Fly
                    ax_fly.plot(fly_comparison.index, fly_comparison['PCA Fair'], 
                                  label=f'PCA Fair Fly ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color='brown')
                    
                    # Plot Mispricing (Original - PCA Fair)
                    mispricing = fly_comparison['Original'] - fly_comparison['PCA Fair']
                    ax_fly.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7) # Add zero line for reference
                    
                    # Annotate the fly with the largest absolute mispricing
                    max_abs_mispricing = mispricing.abs().max()
                    if max_abs_mispricing > 0:
                        mispricing_fly = mispricing.abs().idxmax()
                        mispricing_value = mispricing.loc[mispricing_fly] * 10000 # Convert to BPS
                        
                        ax_fly.annotate(
                            f"Mispricing: {mispricing_value:.2f} BPS",
                            (mispricing_fly, fly_comparison.loc[mispricing_fly]['Original']),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                        )
                    
                    ax_fly.set_title('Market Butterfly vs. PCA Fair Butterfly', fontsize=16)
                    ax_fly.set_xlabel('Butterfly Contract (C1-2xC2+C3)')
                    ax_fly.set_ylabel('Butterfly Value')
                    ax_fly.legend(loc='upper right')
                    ax_fly.grid(True, linestyle=':', alpha=0.6)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig_fly)

                    # --- Detailed Fly Table ---
                    st.markdown("###### Butterfly (Fly) Mispricing")
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
        
        # --- NEW: 7.4 Custom Spread and Butterfly Snapshots ---
        
        custom_derivative_types = [
            '6-Month Spreads', '12-Month Spreads',
            '6-Month Butterflies', '12-Month Butterflies'
        ]

        # Start section numbering from 7.4
        starting_sub_section_num = 4 

        for i, derivative_type in enumerate(custom_derivative_types):
            historical_data = historical_derivatives_map.get(derivative_type)
            
            st.subheader(f"7.{starting_sub_section_num + i}. {derivative_type} Snapshot")
            
            if historical_data is None or historical_data.empty:
                # Check if the reason is lack of contracts
                if derivative_type == '6-Month Spreads' and analysis_curve_df.shape[1] < 3:
                     st.info(f"No derivative contracts could be constructed for {derivative_type} from the available curve. Need at least 3 contracts (e.g., Z5, H6, M6) for one 6-month spread (Z5-M6).")
                elif derivative_type == '12-Month Spreads' and analysis_curve_df.shape[1] < 5:
                     st.info(f"No derivative contracts could be constructed for {derivative_type} from the available curve. Need at least 5 contracts (e.g., Z5...Z6) for one 12-month spread (Z5-Z6).")
                elif derivative_type == '6-Month Butterflies' and analysis_curve_df.shape[1] < 5:
                     st.info(f"No derivative contracts could be constructed for {derivative_type} from the available curve. Need at least 5 contracts (e.g., Z5...U6) for one 6-month fly (Z5-2xM6+U6).")
                elif derivative_type == '12-Month Butterflies' and analysis_curve_df.shape[1] < 9:
                     st.info(f"No derivative contracts could be constructed for {derivative_type} from the available curve. Need at least 9 contracts (e.g., Z5...U7) for one 12-month fly (Z5-2xZ6+U7).")
                else:
                    st.warning(f"No derivative contracts could be reconstructed for {derivative_type} from the available curve. Skipping snapshot.")
                continue

            try:
                # 1. Select the single day's data
                snapshot_original = historical_data.filter(regex='\(Original\)$').loc[[analysis_dt]].T
                snapshot_pca = historical_data.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
                
                # 2. Clean the index labels
                snapshot_original.columns = ['Original']
                snapshot_original.index = snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)

                snapshot_pca.columns = ['PCA Fair']
                snapshot_pca.index = snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)

                # 3. Concatenate and drop NaNs
                comparison = pd.concat([snapshot_original, snapshot_pca], axis=1).dropna()
                
                if comparison.empty:
                    st.warning(f"No complete data available for {derivative_type} on the selected analysis date. The curve may be too short or data is missing.")
                    continue

                # --- Plot the Derivative ---
                plt.style.use('default') 
                fig_deriv, ax_deriv = plt.subplots(figsize=(15, 7))
                
                # Use different colors for custom derivatives
                if 'Spread' in derivative_type:
                    color_original = 'teal'
                    color_pca = 'cyan'
                else: # Butterfly
                    color_original = 'darkred'
                    color_pca = 'pink'

                ax_deriv.plot(comparison.index, comparison['Original'], 
                              label=f'Original Market {derivative_type.split(" ")[-1]}', marker='o', linestyle='-', linewidth=2.5, color=color_original)
                
                ax_deriv.plot(comparison.index, comparison['PCA Fair'], 
                              label=f'PCA Fair {derivative_type.split(" ")[-1]} ({pc_count} PCs)', marker='x', linestyle='--', linewidth=2.5, color=color_pca)
                
                mispricing = comparison['Original'] - comparison['PCA Fair']
                ax_deriv.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7) 

                # Annotate Mispricing
                max_abs_mispricing = mispricing.abs().max()
                if max_abs_mispricing > 0:
                    mispricing_contract = mispricing.abs().idxmax()
                    mispricing_value = mispricing.loc[mispricing_contract] * 10000 # Convert to BPS
                    
                    ax_deriv.annotate(
                        f"Mispricing: {mispricing_value:.2f} BPS",
                        (mispricing_contract, comparison.loc[mispricing_contract]['Original']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                    )
                
                ax_deriv.set_title(f'Market {derivative_type} vs. PCA Fair {derivative_type}', fontsize=16)
                ax_deriv.set_xlabel('Contract Structure')
                ax_deriv.set_ylabel('Value (Price Difference)')
                ax_deriv.legend(loc='upper right')
                ax_deriv.grid(True, linestyle=':', alpha=0.6)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_deriv)

                # --- Detailed Table ---
                st.markdown(f"###### {derivative_type} Mispricing")
                detailed_comparison = comparison.copy()
                detailed_comparison.index.name = 'Contract Structure'
                detailed_comparison['Mispricing (BPS)'] = mispricing * 10000
                detailed_comparison = detailed_comparison.rename(
                    columns={'Original': 'Original Value', 'PCA Fair': 'PCA Fair Value'}
                )
                
                st.dataframe(
                    detailed_comparison.style.format({
                        'Original Value': "{:.4f}",
                        'PCA Fair Value': "{:.4f}",
                        'Mispricing (BPS)': "{:.2f}"
                    }),
                    use_container_width=True
                )

            except KeyError:
                st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the data for {derivative_type}.")
        
    else:
        st.error("PCA failed. Please check your data quantity and quality.")
