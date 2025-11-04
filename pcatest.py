# -*- coding: utf-8 -*-
# SOFR/Treasury Curve PCA Backtesting Engine (sofr_pca_backtest.py)
#
# This script performs a walk-forward validation (backtest) of the PCA-based
# yield curve forecasting models for the US SOFR/Treasury curve.
#
# It calculates and backtests: Outright Rates, Spreads, and Butterflies (Flies).
# ---------------------------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import datetime
from datetime import timedelta

# --- Setup ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="SOFR Curve PCA Backtesting Engine")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------------
# CORE DATA PROCESSING FUNCTIONS (Adapted for Fixed Tenors)
# ---------------------------------------------------------------------------------

def safe_to_datetime(s):
    """Robustly converts a string to a datetime object."""
    if pd.isna(s): return pd.NaT
    formats_to_try = ['%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d']
    for fmt in formats_to_try:
        try: return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError): continue
    return pd.to_datetime(s, errors='coerce')

def normalize_rate_input(val, unit):
    """Converts rate input (e.g., 13.45%) to decimal fraction (e.g., 0.1345)."""
    if pd.isna(val): return np.nan
    v = float(val)
    if "Percent" in unit: return v / 100.0
    if "Basis" in unit: return v / 10000.0
    return v

def denormalize_to_percent(frac):
    """Converts decimal fraction to percentage (e.g., 0.1345 to 13.45)."""
    if pd.isna(frac): return np.nan
    return 100.0 * float(frac)

def parse_tenor_to_ttm(col_name):
    """Converts a tenor string (e.g., '3M', '5Y') to TTM in years (0.25, 5.0)."""
    col_upper = col_name.upper().strip()
    if 'M' in col_upper:
        try: return float(col_upper.replace('M', '')) / 12.0 # Months to Years
        except ValueError: return np.nan
    elif 'Y' in col_upper:
        try: return float(col_upper.replace('Y', '')) # Years
        except ValueError: return np.nan
    return np.nan

def build_std_grid_by_rule(max_year=30.0):
    """
    Defines the standard set of TTM tenors for the US SOFR/Treasury curve.
    Uses standard points: 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 30Y.
    """
    # Define standard US tenors in years
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
    
    # Filter tenors to the chosen max_year
    std_grid_tenors = tenors[tenors <= max_year]
    
    # Create column names (e.g., 1.0 -> '1Y')
    std_grid_cols = [f"{int(t)}Y" for t in std_grid_tenors]
            
    return list(std_grid_tenors), std_grid_cols


def row_to_std_grid(row_series, available_tenor_cols, std_arr, rate_unit, interp_method):
    """Interpolates a single day's fixed-tenor rates to the standard TTM grid."""
    ttm_list, rate_list = [], []
    
    for col in available_tenor_cols:
        t = parse_tenor_to_ttm(col)
        raw_val = row_series.get(col, np.nan)
        
        if not np.isnan(t) and t > 0 and pd.notna(raw_val):
            rate_frac = normalize_rate_input(raw_val, rate_unit)
            rate_list.append(denormalize_to_percent(rate_frac))
            ttm_list.append(t)

    # Check if we have enough distinct points for interpolation
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        # Filter for points that are within the range of the standard grid
        min_ttm = min(ttm_list)
        max_ttm = max(ttm_list)
        
        # Only interpolate for points covered by the input data
        target_ttm = np.array([t for t in std_arr if t >= min_ttm and t <= max_ttm])
        
        if len(target_ttm) == 0:
            return np.full(len(std_arr), np.nan, dtype=float)
        
        try:
            order = np.argsort(ttm_list)
            f = interp1d(np.array(ttm_list)[order], np.array(rate_list)[order], 
                         kind=interp_method, bounds_error=False, 
                         fill_value=(rate_list[order[0]], rate_list[order[-1]]), # Linear extrapolation if needed
                         assume_sorted=True)
            
            interpolated_rates = f(target_ttm)
            
            # Map the interpolated rates back to the full standard grid
            result_rates = np.full(len(std_arr), np.nan, dtype=float)
            for i, t in enumerate(std_arr):
                if t in target_ttm:
                    result_rates[i] = interpolated_rates[np.where(target_ttm == t)[0][0]]
            return result_rates

        except Exception: 
            return np.full(len(std_arr), np.nan, dtype=float)
    return np.full(len(std_arr), np.nan, dtype=float)


def build_pca_matrix(yields_df_train, std_arr, std_cols, rate_unit, interp_method):
    """
    Builds the full PCA input matrix (Outright Rates, Spreads, Flies)
    from interpolated training data.
    """
    # Identify columns that represent tenors (contain 'Y' or 'M')
    available_tenor_cols = [col for col in yields_df_train.columns if any(s in col.upper() for s in ['M', 'Y'])]
    
    pca_df_zeros = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    
    # Step 1: Interpolate to Standard Grid (Outright Rates)
    for dt in yields_df_train.index:
        pca_df_zeros.loc[dt] = row_to_std_grid(
            yields_df_train.loc[dt], available_tenor_cols,
            std_arr, rate_unit, interp_method
        )

    # Remove rows where interpolation failed (all NaNs)
    pca_df_zeros = pca_df_zeros.dropna(how='all')
    if pca_df_zeros.empty:
        return pd.DataFrame()

    # Step 2: Calculate Spreads (Slope)
    spread_cols = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
    pca_df_spreads = pd.DataFrame(index=pca_df_zeros.index, columns=spread_cols, dtype=float)
    for i in range(1, len(std_cols)):
        col_name = f"{std_cols[i]}-{std_cols[i-1]}"
        pca_df_spreads[col_name] = pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]]

    # Step 3: Calculate Flies (Curvature)
    fly_cols = [f"{std_cols[i+1]}x{std_cols[i]}x{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
    pca_df_flies = pd.DataFrame(index=pca_df_zeros.index, columns=fly_cols, dtype=float)
    for i in range(1, len(std_cols) - 1):
        col_name = f"{std_cols[i+1]}x{std_cols[i]}x{std_cols[i-1]}"
        # Standard butterfly: (Long - Mid) - (Mid - Short)
        spread_long = pca_df_zeros[std_cols[i+1]] - pca_df_zeros[std_cols[i]]
        spread_short = pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]]
        pca_df_flies[col_name] = spread_long - spread_short

    # Step 4: Combine all series
    pca_df_combined = pd.concat([pca_df_zeros, pca_df_spreads, pca_df_flies], axis=1)

    # Handle NaNs: Fill with column mean (only for remaining single NaNs after the 'all' drop)
    pca_vals = pca_df_combined.values.astype(float)
    col_means = np.nanmean(pca_vals, axis=0)
    if np.isnan(col_means).any():
        overall_mean = np.nanmean(col_means[~np.isnan(col_means)]) if np.any(~np.isnan(col_means)) else 0.0
        col_means = np.where(np.isnan(col_means), overall_mean, col_means)
    inds = np.where(np.isnan(pca_vals))
    if inds[0].size > 0:
        pca_vals[inds] = np.take(col_means, inds[1])

    return pd.DataFrame(pca_vals, index=pca_df_combined.index, columns=pca_df_combined.columns)


# ---------------------------------------------------------------------------------
# PCA AND FORECASTING FUNCTIONS
# ---------------------------------------------------------------------------------

def forecast_pcs_var(PCs_df, lags=1):
    """Forecasts the next Principal Component vector using VAR."""
    if len(PCs_df) < lags + 5: return PCs_df.iloc[-1:].values
    results = VAR(PCs_df).fit(lags)
    return results.forecast(PCs_df.values[-lags:], steps=1)

def forecast_pcs_arima(PCs_df):
    """Forecasts the next Principal Component vector using ARIMA(1,1,0) per component."""
    forecasts = []
    for _, series in PCs_df.items():
        if len(series) < 10:
            forecasts.append(series.iloc[-1])
            continue
        try:
            forecasts.append(ARIMA(series, order=(1, 1, 0)).fit().forecast(steps=1).iloc[0])
        except Exception:
             forecasts.append(series.iloc[-1])
    return np.array(forecasts).reshape(1, -1)


# ---------------------------------------------------------------------------------
# STREAMLIT UI AND EXECUTION
# ---------------------------------------------------------------------------------

def main():
    st.title("🏛️ SOFR/Treasury Curve PCA Backtesting Engine")
    st.markdown("This tool performs a walk-forward backtest of PCA-based curve forecasting models for fixed-tenor yields, analyzing **Outright Rates, Spreads, and Flies**.")
    st.markdown("---")

    # --- Sidebar Inputs ---
    st.sidebar.header("1) Upload Data")
    yield_file = st.sidebar.file_uploader("Yield data CSV (Dates and Tenor Columns: e.g., '1Y', '5Y')", type="csv")
    
    # Removed expiry_file and holiday_file for SOFR

    st.sidebar.header("2) Configure Backtest")
    training_window_days = st.sidebar.number_input("Rolling Training Window (business days)", min_value=120, max_value=500, value=252, step=1)

    st.sidebar.header("3) Model Parameters")
    n_components_sel = st.sidebar.slider("Number of PCA components", 1, 10, 3)
    forecast_model_type = st.sidebar.selectbox("Forecasting Model to Test", ["VAR (Vector Autoregression)", "ARIMA (per Component)", "PCA Fair Value"])
    var_lags = 1
    if forecast_model_type == "VAR (Vector Autoregression)":
        var_lags = st.sidebar.number_input("VAR model lags", min_value=1, max_value=5, value=1, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Conventions")
    rate_unit = st.sidebar.selectbox("Input rate unit", ["Percent (e.g. 13.45)", "Decimal (e.g. 0.1345)", "Basis points (e.g. 1345)"])
    
    # --- Run Button ---
    run_backtest = st.sidebar.button("Run Backtest", type="primary")

    # --- Initialize session state for UI state and results ---
    if 'results_df' not in st.session_state: st.session_state.results_df = None
    if 'unique_dates' not in st.session_state: st.session_state.unique_dates = []
    if 'selected_date_index' not in st.session_state: st.session_state.selected_date_index = 0
    if 'selected_spread_date_index' not in st.session_state: st.session_state.selected_spread_date_index = 0

    # --- Callbacks for Next/Prev Buttons ---
    def next_date(key):
        if key == 'rates' and st.session_state.selected_date_index < len(st.session_state.unique_dates) - 1:
            st.session_state.selected_date_index += 1
        elif key == 'spreads' and st.session_state.selected_spread_date_index < len(st.session_state.unique_dates) - 1:
            st.session_state.selected_spread_date_index += 1

    def prev_date(key):
        if key == 'rates' and st.session_state.selected_date_index > 0:
            st.session_state.selected_date_index -= 1
        elif key == 'spreads' and st.session_state.selected_spread_date_index > 0:
            st.session_state.selected_spread_date_index -= 1

    # --- Data Loading and Validation ---
    if run_backtest:
        st.session_state.results_df = None # Clear previous results

        if not yield_file:
            st.error("Please upload the Yield data CSV file.")
            st.stop()

        @st.cache_data(show_spinner="Loading and sanitizing data...")
        def load_all_data(yield_file):
            # Yields
            yields_df = pd.read_csv(io.StringIO(yield_file.getvalue().decode("utf-8")))
            date_col = yields_df.columns[0]
            yields_df[date_col] = yields_df[date_col].apply(safe_to_datetime)
            yields_df = yields_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            yields_df.columns = [str(c).strip() for c in yields_df.columns]
            for c in yields_df.columns:
                # Only attempt conversion for columns that look like tenors
                if any(s in c.upper() for s in ['M', 'Y']):
                    yields_df[c] = pd.to_numeric(yields_df[c], errors="coerce")
            
            return yields_df

        yields_df = load_all_data(yield_file)

        # --- Date Range Selection (Moved here to use loaded data) ---
        st.sidebar.subheader("Backtest Period")
        max_date = yields_df.index.max().date()
        min_date = yields_df.index.min().date()

        initial_start_date = min_date + timedelta(days=training_window_days * 1.5)
        if initial_start_date >= max_date:
            initial_start_date = max_date - timedelta(days=training_window_days * 1.5)
            if initial_start_date <= min_date: initial_start_date = min_date
            
        date_range = st.sidebar.date_input(
            "Select Start and End Dates",
            value=[initial_start_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) != 2:
            st.warning("Please select a start and an end date for the backtest.")
            st.stop()

        backtest_start_date = pd.to_datetime(date_range[0])
        backtest_end_date = pd.to_datetime(date_range[1])

        if backtest_start_date >= backtest_end_date:
            st.error("Backtest Start Date must be before the End Date.")
            st.stop()
        
        # --------------------------
        # BACKTESTING LOOP
        # --------------------------
        st.subheader("Running Walk-Forward Backtest...")
        
        results = []
        backtest_range = pd.bdate_range(start=backtest_start_date, end=backtest_end_date)
        all_available_dates = yields_df.index

        # Setup grid
        std_arr, std_cols = build_std_grid_by_rule(max_year=30.0)
        spread_cols = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
        fly_cols = [f"{std_cols[i+1]}x{std_cols[i]}x{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
        all_cols_full = std_cols + spread_cols + fly_cols

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, current_date in enumerate(backtest_range):
            if current_date not in all_available_dates:
                continue

            # T-1 is the training end date
            training_end_date = current_date - pd.Timedelta(days=1)
            # Fetch a window large enough to select `training_window_days` of business days
            training_start_date = training_end_date - pd.DateOffset(days=training_window_days * 1.5)
            
            train_mask = (yields_df.index >= training_start_date) & (yields_df.index <= training_end_date)
            yields_df_train = yields_df.loc[train_mask].sort_index().tail(training_window_days)

            if len(yields_df_train) < training_window_days / 2: continue

            # Build the full PCA matrix (interpolated TTMs, spreads, flies)
            pca_df_filled = build_pca_matrix(yields_df_train, std_arr, std_cols, rate_unit, interp_method='linear')
            
            if pca_df_filled.empty: continue

            # PCA Decomposition and Training
            scaler = StandardScaler(with_std=False)
            X = scaler.fit_transform(pca_df_filled.values.astype(float))
            n_components_sel_capped = min(n_components_sel, X.shape[1])
            pca = PCA(n_components=n_components_sel_capped)
            PCs = pca.fit_transform(X)
            pc_cols = [f"PC{i+1}" for i in range(n_components_sel_capped)]
            PCs_df = pd.DataFrame(PCs, index=pca_df_filled.index, columns=pc_cols)

            # Forecasting
            last_actual_full_curve = pca_df_filled.iloc[-1].values
            
            if forecast_model_type == "PCA Fair Value":
                last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
                reconstructed_centered = pca.inverse_transform(last_pcs)
                pred_full_curve = scaler.inverse_transform(reconstructed_centered).flatten()
            else:
                if forecast_model_type == "VAR (Vector Autoregression)":
                    pcs_next = forecast_pcs_var(PCs_df, lags=var_lags)
                else: # ARIMA
                    pcs_next = forecast_pcs_arima(PCs_df)
                
                # Forecasting the next level via Delta approach
                last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
                delta_pcs = pcs_next - last_pcs
                delta_curve = pca.inverse_transform(delta_pcs)
                pred_full_curve = last_actual_full_curve + delta_curve.flatten()

            # Build the ACTUAL curve for comparison (using the interpolated rates of the actual day T)
            actual_full_curve_series = build_pca_matrix(
                yields_df.loc[[current_date]], # Just the one day
                std_arr, std_cols, rate_unit, interp_method='linear'
            )
            
            if actual_full_curve_series.empty: continue
            
            actual_full_curve = actual_full_curve_series.iloc[0].values
            
            results.append({
                "Date": current_date,
                "Predicted_Curve": pred_full_curve,
                "Actual_Curve": actual_full_curve,
                "Column_Names": all_cols_full
            })

            progress_bar.progress((i + 1) / len(backtest_range))
            status_text.text(f"Processing: {current_date.date()}...")

        status_text.success("Backtest complete!")
        progress_bar.empty()
        
        if results:
            st.session_state.results_df = pd.DataFrame(results)
        else:
            st.error("Backtest failed to generate results. Check data range and window size.")


    # --------------------------
    # RESULTS ANALYSIS
    # --------------------------
    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df.copy()
        
        # Filter out rows where the actual curve is all NaN
        mask = results_df['Actual_Curve'].apply(lambda x: isinstance(x, float) or np.isnan(x).all())
        results_df = results_df[~mask].copy()

        if results_df.empty:
            st.error("No valid results to analyze.")
            st.stop()

        # Get column names
        all_cols = results_df.iloc[0]['Column_Names']
        std_cols = [c for c in all_cols if 'x' not in c and '-' not in c]
        spread_cols = [c for c in all_cols if '-' in c and 'x' not in c]
        fly_cols = [c for c in all_cols if 'x' in c]

        # Calculate daily errors
        def calculate_errors(row, cols):
            indices = [all_cols.index(c) for c in cols]
            pred = row['Predicted_Curve'][indices]
            actual = row['Actual_Curve'][indices]
            if np.isnan(actual).all(): return pd.Series({'RMSE': np.nan, 'MAE': np.nan})
            return pd.Series({
                'RMSE': np.sqrt(np.nanmean((pred - actual)**2)),
                'MAE': np.nanmean(np.abs(pred - actual))
            })
        
        results_df[['Daily_RMSE_Rates', 'Daily_MAE_Rates']] = results_df.apply(lambda row: calculate_errors(row, std_cols), axis=1)
        results_df[['Daily_RMSE_Spreads', 'Daily_MAE_Spreads']] = results_df.apply(lambda row: calculate_errors(row, spread_cols), axis=1)
        results_df[['Daily_RMSE_Flies', 'Daily_MAE_Flies']] = results_df.apply(lambda row: calculate_errors(row, fly_cols), axis=1)

        st.header("📊 Backtest Performance Analysis")

        # --------------------------
        # Outright Rates Results
        # --------------------------
        st.subheader("1. Outright Rates Performance")
        
        col1, col2 = st.columns(2)
        col1.metric("Avg Daily Curve RMSE (Rates, %)", f"{results_df['Daily_RMSE_Rates'].mean():.4f}")
        col2.metric("Avg Daily Curve MAE (Rates, %)", f"{results_df['Daily_MAE_Rates'].mean():.4f}")

        st.markdown("---")
        st.subheader("Daily Curve Visualization (Outright Rates)")

        unique_dates = results_df['Date'].dt.date.unique()
        st.session_state.unique_dates = unique_dates
        std_arr, _ = build_std_grid_by_rule(max_year=30.0)
        
        col_p, col_d, col_n = st.columns([1, 4, 1])
        col_p.button("◀ Previous", key="rates_prev", on_click=prev_date, args=('rates',))
        col_n.button("Next ▶", key="rates_next", on_click=next_date, args=('rates',))
        
        selected_date = col_d.selectbox(
            "Select a date to inspect",
            options=unique_dates,
            index=st.session_state.selected_date_index
        )
        
        if selected_date:
            plot_data = results_df[results_df['Date'].dt.date == selected_date]
            if not plot_data.empty:
                indices = [all_cols.index(c) for c in std_cols]
                actual_c = plot_data['Actual_Curve'].iloc[0][indices]
                pred_c = plot_data['Predicted_Curve'].iloc[0][indices]

                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(std_arr, actual_c, label=f"Actual on {selected_date}", color='royalblue', marker='o', linestyle='-')
                ax.plot(std_arr, pred_c, label=f"Predicted for {selected_date}", color='darkorange', marker='x', linestyle='--')
                
                ax.set_title(f"Yield Curve Forecast vs. Actual for {selected_date}", fontsize=16)
                ax.set_xlabel("Standardized Maturity (Years)")
                ax.set_ylabel(f"Rate ({rate_unit})")
                ax.set_xticks(std_arr)
                ax.set_xticklabels([f"{int(m)}Y" for m in std_arr], rotation=45, ha="right")
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)

        # --------------------------
        # Spreads and Flies Results
        # --------------------------
        st.markdown("---")
        st.subheader("2. Spreads and Flies Performance")
        
        col3, col4 = st.columns(2)
        col3.metric("Avg Daily Curve RMSE (Spreads, %)", f"{results_df['Daily_RMSE_Spreads'].mean():.4f}")
        col4.metric("Avg Daily Curve MAE (Flies, %)", f"{results_df['Daily_MAE_Flies'].mean():.4f}")

        st.markdown("---")
        st.subheader("Daily Visualization (Spreads and Flies)")

        col_p, col_d, col_n = st.columns([1, 4, 1])
        col_p.button("◀ Previous", key="spreads_prev", on_click=prev_date, args=('spreads',))
        col_n.button("Next ▶", key="spreads_next", on_click=next_date, args=('spreads',))
        
        selected_spread_date = col_d.selectbox(
            "Select a date to inspect spreads/flies",
            options=unique_dates,
            index=st.session_state.selected_spread_date_index
        )
        
        if selected_spread_date:
            plot_data = results_df[results_df['Date'].dt.date == selected_spread_date]
            if not plot_data.empty:
                # Indices for Spreads
                spread_indices = [all_cols.index(c) for c in spread_cols]
                actual_spreads = plot_data['Actual_Curve'].iloc[0][spread_indices]
                pred_spreads = plot_data['Predicted_Curve'].iloc[0][spread_indices]
                
                # Indices for Flies
                fly_indices = [all_cols.index(c) for c in fly_cols]
                actual_flies = plot_data['Actual_Curve'].iloc[0][fly_indices]
                pred_flies = plot_data['Predicted_Curve'].iloc[0][fly_indices]

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Spreads Plot
                ax1.plot(spread_cols, actual_spreads, marker='o', linestyle='-', color='royalblue', label=f"Actual Spreads")
                ax1.plot(spread_cols, pred_spreads, marker='x', linestyle='--', color='darkorange', label="Predicted Spreads")
                ax1.set_title(f"Spread Forecast vs. Actual for {selected_spread_date}")
                ax1.set_ylabel("Spread (%)")
                ax1.set_xticklabels(spread_cols, rotation=45, ha="right")
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax1.legend()
                
                # Flies Plot
                ax2.plot(fly_cols, actual_flies, marker='o', linestyle='-', color='royalblue', label=f"Actual Flies")
                ax2.plot(fly_cols, pred_flies, marker='x', linestyle='--', color='darkorange', label="Predicted Flies")
                ax2.set_title(f"Butterfly Forecast vs. Actual for {selected_spread_date}")
                ax2.set_ylabel("Fly (%)")
                ax2.set_xticklabels(fly_cols, rotation=45, ha="right")
                ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)

        st.markdown("---")
        st.subheader("Raw Results Dataframe")
        st.write("Contains the raw predicted and actual outright, spread, and fly vectors for each day of the backtest.")

        raw_df = pd.DataFrame(results_df['Date']).set_index('Date')
        
        # Get the start and end indices for all metric types
        rates_start_idx = 0
        spreads_start_idx = len(std_cols)
        flies_start_idx = spreads_start_idx + len(spread_cols)

        # Extract predicted and actual components
        for i, col in enumerate(std_cols):
            raw_df[f"Predicted_Rate_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[rates_start_idx + i])
            raw_df[f"Actual_Rate_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[rates_start_idx + i])

        for i, col in enumerate(spread_cols):
            raw_df[f"Predicted_Spread_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[spreads_start_idx + i])
            raw_df[f"Actual_Spread_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[spreads_start_idx + i])
            
        for i, col in enumerate(fly_cols):
            raw_df[f"Predicted_Fly_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[flies_start_idx + i])
            raw_df[f"Actual_Fly_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[flies_start_idx + i])

        st.dataframe(raw_df)

if __name__ == "__main__":
    main()
