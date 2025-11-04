# -*- coding: utf-8 -*-
# SOFR Contract PCA — Backtesting Engine (sofr_contract_pca_backtest_v5.py)
#
# This script performs a walk-forward validation (backtest) of the PCA-based
# yield curve forecasting models for SOFR instruments using contract data.
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
import warnings
import datetime
from datetime import timedelta

# --- Setup ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="SOFR Contract PCA Backtesting Engine")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------------
# CORE DATA PROCESSING FUNCTIONS (Adapted for Contracts)
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
    """Converts rate input (e.g., 5.45%) to decimal fraction (e.g., 0.0545)."""
    if pd.isna(val): return np.nan
    v = float(val)
    if "Percent" in unit: return v / 100.0
    if "Basis" in unit: return v / 10000.0
    return v

def denormalize_to_percent(frac):
    """Converts decimal fraction to percentage (e.g., 0.0545 to 5.45)."""
    if pd.isna(frac): return np.nan
    return 100.0 * float(frac)

def np_busdays_exclusive(start_dt, end_dt, holidays_np):
    """Calculates business days between two dates (exclusive of start, inclusive of end)."""
    if pd.isna(start_dt) or pd.isna(end_dt): return 0
    s = np.datetime64(pd.Timestamp(start_dt).date()) + np.timedelta64(1, "D")
    e = np.datetime64(pd.Timestamp(end_dt).date())
    if e < s: return 0
    return int(np.busday_count(s, e, weekmask="1111100", holidays=holidays_np))

def calculate_ttm(valuation_ts, expiry_ts, holidays_np, year_basis):
    """Calculates Time-To-Maturity (TTM) in years based on business days and year basis."""
    bd = np_busdays_exclusive(valuation_ts, expiry_ts, holidays_np)
    return np.nan if bd <= 0 else bd / float(year_basis)

@st.cache_data
def build_std_grid_by_rule(rule_name, max_year):
    """
    Defines the standard set of TTM tenors for the curve based on a selection rule.
    """
    max_year = float(max_year)
    if rule_name == "US Standard":
        # Standard US tenors: 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 30Y
        tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
        std_grid_tenors = tenors[tenors <= max_year]
        std_grid_cols = [f"{int(t)}Y" for t in std_grid_tenors]
        
    elif rule_name == "0.25Y Increments (DI-style)":
        # Fine grid: 0.25Y, 0.5Y, 0.75Y, 1.0Y, 1.25Y, ..., 2.0Y, ..., 5.0Y, ...
        
        # Fine grid for the short end
        tenors_fine = np.round(np.arange(0.25, 3.0 + 0.001, 0.25), 2)
        # Medium grid for mid-curve
        tenors_medium = np.round(np.arange(3.5, 7.0 + 0.001, 0.5), 2)
        # Coarse grid for long end
        tenors_long = np.round(np.arange(8.0, 30.0 + 0.001, 1.0), 2)
        
        tenors = np.unique(np.concatenate([tenors_fine, tenors_medium, tenors_long]))
        tenors = tenors[tenors <= max_year]

        std_grid_tenors = tenors
        # Clean labels: 0.25Y, 1.00Y (avoiding .25Y, 1Y mixed styles)
        std_grid_cols = [f"{t:.2f}Y" for t in std_grid_tenors]
        
    else:
        return build_std_grid_by_rule("0.25Y Increments (DI-style)", max_year)
    
    return list(std_grid_tenors), std_grid_cols

def row_to_std_grid(dt, row_series, available_contracts, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method):
    """
    Interpolates a single day's raw contract rates to the standard TTM grid.
    TTM is calculated using the contract's expiry date.
    """
    ttm_list, rate_list = [], []
    for col in available_contracts:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index: continue
        exp = expiry_df.loc[mat_up, "DATE"]
        
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date(): continue
        
        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0: continue
        
        raw_val = row_series.get(col, np.nan)
        if pd.isna(raw_val): continue
        
        rate_frac = normalize_rate_input(raw_val, rate_unit)
        rate_list.append(denormalize_to_percent(rate_frac))
        ttm_list.append(t)
        
    # Check if we have enough distinct points for interpolation
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        try:
            min_ttm = min(ttm_list)
            max_ttm = max(ttm_list)
            target_ttm = np.array([t for t in std_arr if t >= min_ttm and t <= max_ttm])
            
            if len(target_ttm) == 0:
                return np.full(len(std_arr), np.nan, dtype=float)

            order = np.argsort(ttm_list)
            f = interp1d(np.array(ttm_list)[order], np.array(rate_list)[order], 
                         kind=interp_method, bounds_error=False, 
                         fill_value=(rate_list[order[0]], rate_list[order[-1]]),
                         assume_sorted=True)
            
            interpolated_rates = f(target_ttm)
            
            result_rates = np.full(len(std_arr), np.nan, dtype=float)
            for i, t in enumerate(std_arr):
                if t in target_ttm:
                    match_index = np.where(np.isclose(target_ttm, t))[0]
                    if match_index.size > 0:
                        result_rates[i] = interpolated_rates[match_index[0]]
            return result_rates

        except Exception: 
            return np.full(len(std_arr), np.nan, dtype=float)
    return np.full(len(std_arr), np.nan, dtype=float)

def build_pca_matrix(yields_df_train, expiry_df, std_arr, std_cols, holidays_np, year_basis, rate_unit, interp_method):
    """
    Builds the full PCA input matrix (Outright Rates, Spreads, Flies)
    from interpolated training data based on contract expiries.
    """
    pca_df_zeros = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    available_contracts = [col for col in yields_df_train.columns if col in expiry_df.index]
    
    # Step 1: Interpolate to Standard Grid (Outright Rates)
    for dt in yields_df_train.index:
        pca_df_zeros.loc[dt] = row_to_std_grid(
            dt, yields_df_train.loc[dt], available_contracts, expiry_df,
            std_arr, holidays_np, year_basis, rate_unit, interp_method
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
        spread_long = pca_df_zeros[std_cols[i+1]] - pca_df_zeros[std_cols[i]]
        spread_short = pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]]
        pca_df_flies[col_name] = spread_long - spread_short

    # Step 4: Combine all series
    pca_df_combined = pd.concat([pca_df_zeros, pca_df_spreads, pca_df_flies], axis=1)

    # Handle NaNs: Fill with column mean
    pca_vals = pca_df_combined.values.astype(float)
    col_means = np.nanmean(pca_vals, axis=0)
    if np.isnan(col_means).any():
        overall_mean = np.nanmean(col_means[~np.isnan(col_means)]) if np.any(~np.isnan(col_means)) else 0.0
        col_means = np.where(np.isnan(col_means), overall_mean, col_means)
    inds = np.where(np.isnan(pca_vals))
    if inds[0].size > 0:
        pca_vals[inds] = np.take(col_means, inds[1])

    return pd.DataFrame(pca_vals, index=pca_df_combined.index, columns=pca_df_combined.columns)

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

@st.cache_data(show_spinner="Loading and sanitizing data...")
def load_all_data(yield_file, expiry_file, holiday_file):
    # Yields
    yields_df = pd.read_csv(io.StringIO(yield_file.getvalue().decode("utf-8")))
    date_col = yields_df.columns[0]
    yields_df[date_col] = yields_df[date_col].apply(safe_to_datetime)
    yields_df = yields_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    yields_df.columns = [str(c).strip() for c in yields_df.columns]
    for c in yields_df.columns:
        yields_df[c] = pd.to_numeric(yields_df[c], errors="coerce")
    
    # Expiry
    expiry_raw = pd.read_csv(io.StringIO(expiry_file.getvalue().decode("utf-8")))
    expiry_df = expiry_raw.iloc[:, :2].copy()
    expiry_df.columns = ["MATURITY", "DATE"]
    expiry_df["MATURITY"] = expiry_df["MATURITY"].astype(str).str.strip().str.upper()
    expiry_df["DATE"] = expiry_df["DATE"].apply(safe_to_datetime)
    expiry_df = expiry_df.dropna(subset=["DATE"]).set_index("MATURITY")

    # Holidays
    holidays_np = np.array([], dtype="datetime64[D]")
    if holiday_file:
        hol_df = pd.read_csv(io.StringIO(holiday_file.getvalue().decode("utf-8")))
        hol_series = hol_df.iloc[:, 0].apply(safe_to_datetime).dropna()
        if not hol_series.empty:
            holidays_np = np.array(hol_series.dt.date, dtype="datetime64[D]")
            
    return yields_df, expiry_df, holidays_np

def main():
    st.title("🏛️ SOFR Contract PCA Backtesting Engine")
    st.markdown("This tool performs a walk-forward backtest of PCA-based curve forecasting models for **contract-based** yields.")
    st.markdown("---")

    # --- 1) File Uploads ---
    st.sidebar.header("1) Upload Data")
    st.sidebar.info("Requires Yield data, Expiry Map (contract tickers), and Holidays.")
    yield_file = st.sidebar.file_uploader("Yield data CSV (Daily Rates by Contract Ticker)", type="csv")
    expiry_file = st.sidebar.file_uploader("Expiry mapping CSV (Contract Ticker -> Date)", type="csv")
    holiday_file = st.sidebar.file_uploader("Holiday dates CSV (US Federal Holidays)", type="csv")

    # --- 2) Model and Backtest Config (Always visible) ---
    st.sidebar.header("2) Configure Backtest")
    training_window_days = st.sidebar.number_input("Rolling Training Window (business days)", min_value=120, max_value=500, value=252, step=1)
    
    st.sidebar.subheader("Model Parameters")
    n_components_sel = st.sidebar.slider("Number of PCA components", 1, 10, 3)
    forecast_model_type = st.sidebar.selectbox("Forecasting Model to Test", ["VAR (Vector Autoregression)", "ARIMA (per Component)", "PCA Fair Value"])
    var_lags = 1
    if forecast_model_type == "VAR (Vector Autoregression)":
        var_lags = st.sidebar.number_input("VAR model lags", min_value=1, max_value=5, value=1, step=1)
        
    st.sidebar.subheader("Data Conventions")
    rate_unit = st.sidebar.selectbox("Input rate unit", ["Percent (e.g. 5.45)", "Decimal (e.g. 0.0545)", "Basis points (e.g. 545)"])
    year_basis = int(st.sidebar.selectbox("Year Basis for TTM (e.g. 252 or 365)", [252, 365], index=0))
    interp_method = "linear" 

    # --- 3) Curve Definition & Grid (Always visible) ---
    st.sidebar.header("3) Curve Definition & Grid")
    grid_style_sel = st.sidebar.selectbox("Standard Tenor Grid Style", ["0.25Y Increments (DI-style)", "US Standard"])
    max_tenor_pca = st.sidebar.slider("Maximum Tenor for PCA Model (Years)", min_value=5, max_value=30, value=10, step=1)
    # This slider sets the USER'S PREFERENCE for the max plot tenor
    max_tenor_plot_pref = st.sidebar.slider("Maximum Tenor for Graphs (Years)", min_value=0.5, max_value=float(max_tenor_pca), value=5.0, step=0.5)

    # --- Initialize session state for UI state and results ---
    if 'results_df' not in st.session_state: st.session_state.results_df = None
    if 'unique_dates' not in st.session_state: st.session_state.unique_dates = []
    if 'selected_date_index' not in st.session_state: st.session_state.selected_date_index = 0
    if 'selected_spread_date_index' not in st.session_state: st.session_state.selected_spread_date_index = 0

    # --- CALLBACK DEFINITIONS (FIXED NAVIGATION LOGIC) ---
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
            
    def sync_rates_index():
        """Updates the rates index in session state when the selectbox is manually changed."""
        unique_dates = st.session_state.unique_dates
        if unique_dates is not None and st.session_state.rates_date_select in unique_dates:
            st.session_state.selected_date_index = unique_dates.tolist().index(st.session_state.rates_date_select)

    def sync_spreads_index():
        """Updates the spreads index in session state when the selectbox is manually changed."""
        unique_dates = st.session_state.unique_dates
        if unique_dates is not None and st.session_state.spreads_date_select in unique_dates:
            st.session_state.selected_spread_date_index = unique_dates.tolist().index(st.session_state.spreads_date_select)
            
    # ----------------------------

    # --- Load Data and Determine Date Range (Dynamic) ---
    yields_df, expiry_df, holidays_np = None, None, None
    backtest_start_date, backtest_end_date = None, None
    
    if all([yield_file, expiry_file]):
        try:
            yields_df, expiry_df, holidays_np = load_all_data(yield_file, expiry_file, holiday_file)
        except Exception as e:
            st.error(f"Error loading or parsing data: {e}")
            st.stop()
        
        if yields_df is not None and not yields_df.empty:
            st.sidebar.header("4) Select Backtest Period")
            
            max_date = yields_df.index.max().date()
            min_data_start = yields_df.index.min().date()
            
            min_lookback_dt = yields_df.index.min() + pd.DateOffset(days=training_window_days * 1.5)
            min_valid_start_date = min_lookback_dt.date()
            
            min_valid_start_date = max(min_valid_start_date, min_data_start)
            
            if min_valid_start_date >= max_date:
                st.sidebar.error("Data period is too short for the selected training window.")
            else:
                default_start_date = min_valid_start_date
                
                date_range = st.sidebar.date_input(
                    "Select Start and End Dates",
                    value=[default_start_date, max_date],
                    min_value=min_data_start,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    backtest_start_date = pd.to_datetime(date_range[0])
                    backtest_end_date = pd.to_datetime(date_range[1])
                elif len(date_range) == 1:
                    st.sidebar.warning("Please select an end date.")
                

    st.sidebar.markdown("---")
    run_backtest = st.sidebar.button("Run Backtest", type="primary")

    # --- Backtest Execution ---
    if run_backtest:
        if yields_df is None or expiry_df is None:
            st.error("Please upload the required data files (Yields, Expiry).")
            st.stop()
        
        if backtest_start_date is None or backtest_end_date is None or backtest_start_date >= backtest_end_date:
            st.error("Please select a valid, non-empty Backtest Period in Section 4.")
            st.stop()
            
        min_required_date = yields_df.index.min() + pd.DateOffset(days=training_window_days * 1.5)
        if backtest_start_date < min_required_date:
            st.warning(f"Starting backtest on {backtest_start_date.date()} may result in a partial or invalid training window (less than {training_window_days} business days) for the initial few steps.")


        st.subheader("Running Walk-Forward Backtest...")
        
        results = []
        backtest_range = pd.bdate_range(start=backtest_start_date, end=backtest_end_date)
        all_available_dates = yields_df.index

        # Setup custom grid for PCA model training (This is the fixed full grid)
        std_arr, std_cols = build_std_grid_by_rule(rule_name=grid_style_sel, max_year=float(max_tenor_pca))
        spread_cols = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
        fly_cols = [f"{std_cols[i+1]}x{std_cols[i]}x{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
        all_cols_full = std_cols + spread_cols + fly_cols

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, current_date in enumerate(backtest_range):
            if current_date not in all_available_dates:
                continue

            training_end_date = current_date - pd.Timedelta(days=1)
            training_start_date = training_end_date - pd.DateOffset(days=training_window_days * 1.5)
            
            train_mask = (yields_df.index >= training_start_date) & (yields_df.index <= training_end_date)
            yields_df_train = yields_df.loc[train_mask].sort_index().tail(training_window_days)

            if len(yields_df_train) < training_window_days / 2: continue

            # Build the full PCA matrix from interpolated training data
            pca_df_filled = build_pca_matrix(yields_df_train, expiry_df, std_arr, std_cols, holidays_np, year_basis, rate_unit, interp_method)
            
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
                
                last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
                delta_pcs = pcs_next - last_pcs
                delta_curve = pca.inverse_transform(delta_pcs)
                pred_full_curve = last_actual_full_curve + delta_curve.flatten()

            # Build the ACTUAL curve for comparison 
            actual_full_curve_series = build_pca_matrix(
                yields_df.loc[[current_date]], expiry_df, std_arr, std_cols, holidays_np, year_basis, rate_unit, interp_method
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
            st.session_state.results_df = None


    # --------------------------
    # RESULTS ANALYSIS
    # --------------------------
    if 'results_df' in st.session_state and st.session_state.results_df is not None:
        results_df = st.session_state.results_df.copy()
        
        # Data validation and error calculation setup
        mask = results_df['Actual_Curve'].apply(lambda x: isinstance(x, float) or np.isnan(x).all())
        results_df = results_df[~mask].copy()

        if results_df.empty:
            st.error("No valid results to analyze. Ensure enough contract data was available in the backtest period.")
            st.stop()

        all_cols = results_df.iloc[0]['Column_Names']
        std_cols_full = [c for c in all_cols if 'x' not in c and '-' not in c]
        spread_cols_full = [c for c in all_cols if '-' in c and 'x' not in c]
        fly_cols_full = [c for c in all_cols if 'x' in c]
        std_arr_full = build_std_grid_by_rule(rule_name=grid_style_sel, max_year=float(max_tenor_pca))[0]

        def calculate_errors(row, cols):
            indices = [all_cols.index(c) for c in cols]
            pred = row['Predicted_Curve'][indices]
            actual = row['Actual_Curve'][indices]
            if np.isnan(actual).all(): return pd.Series({'RMSE': np.nan, 'MAE': np.nan})
            return pd.Series({
                'RMSE': np.sqrt(np.nanmean((pred - actual)**2)),
                'MAE': np.nanmean(np.abs(pred - actual))
            })
        
        results_df[['Daily_RMSE_Rates', 'Daily_MAE_Rates']] = results_df.apply(lambda row: calculate_errors(row, std_cols_full), axis=1)
        results_df[['Daily_RMSE_Spreads', 'Daily_MAE_Spreads']] = results_df.apply(lambda row: calculate_errors(row, spread_cols_full), axis=1)
        results_df[['Daily_RMSE_Flies', 'Daily_MAE_Flies']] = results_df.apply(lambda row: calculate_errors(row, fly_cols_full), axis=1)

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

        col_p, col_d, col_n = st.columns([1, 4, 1])
        
        col_p.button("◀ Previous", key="rates_prev", on_click=prev_date, args=('rates',))
        col_n.button("Next ▶", key="rates_next", on_click=next_date, args=('rates',))
        
        # Use st.session_state.selected_date_index to force the selectbox position
        selected_date = col_d.selectbox(
            "Select a date to inspect",
            options=unique_dates,
            index=st.session_state.selected_date_index,
            key="rates_date_select",  # Added key
            on_change=sync_rates_index # Added sync handler
        )
        
        if selected_date:
            plot_data = results_df[results_df['Date'].dt.date == selected_date]
            if not plot_data.empty:
                full_actual_curve_vector = plot_data['Actual_Curve'].iloc[0]
                
                # --- Dynamic TTM Cap Logic for Rates ---
                rates_indices = [all_cols.index(c) for c in std_cols_full]
                actual_rates = full_actual_curve_vector[rates_indices]

                # Find the index of the last non-NaN rate
                valid_indices = np.where(~np.isnan(actual_rates))[0]
                
                if valid_indices.size > 0:
                    last_valid_rate_index = valid_indices[-1]
                    data_max_tenor = std_arr_full[last_valid_rate_index]
                else:
                    data_max_tenor = 0.0

                # Effective max tenor is the minimum of user's setting and data availability
                effective_max_tenor_plot = min(max_tenor_plot_pref, data_max_tenor)

                # Re-filter plotting arrays based on the effective max tenor
                max_plot_index = next((i for i, t in enumerate(std_arr_full) if t > effective_max_tenor_plot), len(std_arr_full))
                
                if max_plot_index == 0 or effective_max_tenor_plot < 0.25:
                    st.warning(f"No valid contract data available for interpolation up to the requested tenor on {selected_date}.")
                    st.stop()
                    
                plot_std_arr = std_arr_full[:max_plot_index]
                plot_std_cols = std_cols_full[:max_plot_index]

                if grid_style_sel == "0.25Y Increments (DI-style)":
                    plot_labels = [f"{t:.2f}Y" for t in plot_std_arr]
                else:
                    plot_labels = [f"{int(t)}Y" for t in plot_std_arr]
                
                # Now, indices are taken from the full column list, but using the filtered plot_std_cols
                indices = [all_cols.index(c) for c in plot_std_cols]
                actual_c = full_actual_curve_vector[indices]
                pred_full_curve_vector = plot_data['Predicted_Curve'].iloc[0]
                pred_c = pred_full_curve_vector[indices]

                # --- Plotting ---
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(plot_std_arr, actual_c, label=f"Actual on {selected_date}", color='royalblue', marker='o', linestyle='-')
                ax.plot(plot_std_arr, pred_c, label=f"Predicted for {selected_date}", color='darkorange', marker='x', linestyle='--')
                
                ax.set_title(f"Yield Curve Forecast vs. Actual for {selected_date} (Max Tenor: {effective_max_tenor_plot:.2f}Y)", fontsize=16)
                ax.set_xlabel("Standardized Maturity (Years)")
                ax.set_ylabel(f"Rate ({rate_unit})")
                ax.set_xticks(plot_std_arr)
                ax.set_xticklabels(plot_labels, rotation=45, ha="right")
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
        
        # Use st.session_state.selected_spread_date_index to force the selectbox position
        selected_spread_date = col_d.selectbox(
            "Select a date to inspect spreads/flies",
            options=unique_dates,
            index=st.session_state.selected_spread_date_index,
            key="spreads_date_select", # Added key
            on_change=sync_spreads_index # Added sync handler
        )
        
        if selected_spread_date:
            plot_data = results_df[results_df['Date'].dt.date == selected_spread_date]
            if not plot_data.empty:
                full_actual_curve_vector = plot_data['Actual_Curve'].iloc[0]
                
                # --- Dynamic TTM Cap Logic (Re-used) ---
                rates_indices = [all_cols.index(c) for c in std_cols_full]
                actual_rates = full_actual_curve_vector[rates_indices]
                valid_indices = np.where(~np.isnan(actual_rates))[0]
                
                if valid_indices.size > 0:
                    last_valid_rate_index = valid_indices[-1]
                    data_max_tenor = std_arr_full[last_valid_rate_index]
                else:
                    data_max_tenor = 0.0

                effective_max_tenor_plot = min(max_tenor_plot_pref, data_max_tenor)
                max_plot_index = next((i for i, t in enumerate(std_arr_full) if t > effective_max_tenor_plot), len(std_arr_full))
                
                if max_plot_index < 3:
                    st.warning(f"Not enough data to form spreads or flies (requires min 3 rates) on {selected_spread_date} within the selected tenor range.")
                    st.stop()
                    
                # Use the calculated plot_std_cols/arr from the Rates section for consistency
                plot_std_cols = std_cols_full[:max_plot_index]
                
                # Filter Spread and Fly columns for the new dynamically capped Plotting Range
                plot_std_cols_set = set(plot_std_cols)
                plot_spread_cols = [c for c in spread_cols_full if all(t in plot_std_cols_set for t in c.split('-'))]
                plot_fly_cols = [c for c in fly_cols_full if all(t in plot_std_cols_set for t in c.split('x'))]

                # Indices for Spreads
                spread_indices = [all_cols.index(c) for c in plot_spread_cols]
                actual_spreads = full_actual_curve_vector[spread_indices]
                pred_spreads = plot_data['Predicted_Curve'].iloc[0][spread_indices]
                
                # Indices for Flies
                fly_indices = [all_cols.index(c) for c in plot_fly_cols]
                actual_flies = full_actual_curve_vector[fly_indices]
                pred_flies = plot_data['Predicted_Curve'].iloc[0][fly_indices]

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Spreads Plot
                ax1.plot(plot_spread_cols, actual_spreads, marker='o', linestyle='-', color='royalblue', label=f"Actual Spreads")
                ax1.plot(plot_spread_cols, pred_spreads, marker='x', linestyle='--', color='darkorange', label="Predicted Spreads")
                ax1.set_title(f"Spread Forecast vs. Actual for {selected_spread_date} (Max Tenor: {effective_max_tenor_plot:.2f}Y)")
                ax1.set_ylabel("Spread (%)")
                ax1.set_xticklabels(plot_spread_cols, rotation=45, ha="right")
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax1.legend()
                
                # Flies Plot
                ax2.plot(plot_fly_cols, actual_flies, marker='o', linestyle='-', color='royalblue', label=f"Actual Flies")
                ax2.plot(plot_fly_cols, pred_flies, marker='x', linestyle='--', color='darkorange', label="Predicted Flies")
                ax2.set_title(f"Butterfly Forecast vs. Actual for {selected_spread_date} (Max Tenor: {effective_max_tenor_plot:.2f}Y)")
                ax2.set_ylabel("Fly (%)")
                ax2.set_xticklabels(plot_fly_cols, rotation=45, ha="right")
                ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)

        st.markdown("---")
        st.subheader("Raw Results Dataframe (Full PCA Tenors)")
        st.write("Contains the raw predicted and actual outright, spread, and fly vectors for each day of the backtest.")

        raw_df = pd.DataFrame(results_df['Date']).set_index('Date')
        
        rates_start_idx = 0
        spreads_start_idx = len(std_cols_full)
        flies_start_idx = spreads_start_idx + len(spread_cols_full)

        for i, col in enumerate(std_cols_full):
            raw_df[f"Predicted_Rate_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[rates_start_idx + i])
            raw_df[f"Actual_Rate_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[rates_start_idx + i])

        for i, col in enumerate(spread_cols_full):
            raw_df[f"Predicted_Spread_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[spreads_start_idx + i])
            raw_df[f"Actual_Spread_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[spreads_start_idx + i])
            
        for i, col in enumerate(fly_cols_full):
            raw_df[f"Predicted_Fly_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[flies_start_idx + i])
            raw_df[f"Actual_Fly_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[flies_start_idx + i])

        st.dataframe(raw_df)

if __name__ == "__main__":
    main()
