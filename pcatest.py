# -*- coding: utf-8 -*-
# DI Futures PCA — Backtesting Engine (gdqh (3).py)
#
# This script performs a walk-forward backtest of the PCA-based
# yield curve forecasting models for SOFR/DI instruments using contract data.
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

# --- Setup ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="DI Curve Backtesting Engine")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------------
# CORE DATA PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------------

def safe_to_datetime(s):
    """Robustly converts a string to a datetime object by trying multiple common formats."""
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

def build_std_grid_by_rule(max_year=7.0):
    """Defines the standard set of TTM tenors for the curve based on 0.25Y increments (DI-style)."""
    # Fine grid: 0.25Y increments up to 3.0 years
    a = list(np.round(np.arange(0.25, 3.0 + 0.001, 0.25), 2))
    # Mid grid: 0.5Y increments up to 5.0 years
    b = list(np.round(np.arange(3.5, 5.0 + 0.001, 0.5), 2))
    # Long grid: 1.0Y increments up to max_year
    c = list(np.round(np.arange(6.0, max_year + 0.001, 1.0), 2))
    
    tenors = np.unique(np.array(a + b + c))
    return list(tenors[tenors <= max_year])


def row_to_std_grid(dt, row_series, available_contracts, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method):
    """Interpolates a single day's raw contract rates to the standard TTM grid."""
    ttm_list, zero_list = [], []
    for col in available_contracts:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index: continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date(): continue
        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0: continue
        raw_val = row_series.get(col, np.nan)
        if pd.isna(raw_val): continue
        r_frac = normalize_rate_input(raw_val, rate_unit)
        zero_frac = r_frac # Assumes identity compounding for simplicity in backtest
        ttm_list.append(t)
        zero_list.append(denormalize_to_percent(zero_frac))
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        try:
            order = np.argsort(ttm_list)
            f = interp1d(np.array(ttm_list)[order], np.array(zero_list)[order], kind=interp_method, bounds_error=False, fill_value=np.nan, assume_sorted=True)
            return f(std_arr)
        except Exception: return np.full_like(std_arr, np.nan, dtype=float)
    return np.full_like(std_arr, np.nan, dtype=float)

def build_pca_matrix(yields_df_train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method):
    """Builds the full PCA input matrix (Rates, Spreads, Flies) from interpolated data."""
    
    std_cols = [f"{m:.2f}Y" for m in std_arr]
    pca_df_zeros = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    available_contracts = yields_df_train.columns
    for dt in yields_df_train.index:
        pca_df_zeros.loc[dt] = row_to_std_grid(
            dt, yields_df_train.loc[dt], available_contracts, expiry_df,
            std_arr, holidays_np, year_basis, rate_unit, interp_method
        )
    
    pca_df_zeros = pca_df_zeros.dropna(how='all')
    if pca_df_zeros.empty:
        return pd.DataFrame()

    # Step 2: Calculate Spreads
    spread_cols = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
    pca_df_spreads = pd.DataFrame(np.nan, index=pca_df_zeros.index, columns=spread_cols, dtype=float)
    for i in range(1, len(std_cols)):
        col_name = f"{std_cols[i]}-{std_cols[i-1]}"
        pca_df_spreads[col_name] = pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]]

    # Step 3: Calculate Flies
    fly_cols = [f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
    pca_df_flies = pd.DataFrame(np.nan, index=pca_df_zeros.index, columns=fly_cols, dtype=float)
    for i in range(1, len(std_cols) - 1):
        col_name = f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}"
        pca_df_flies[col_name] = (pca_df_zeros[std_cols[i+1]] - pca_df_zeros[std_cols[i]]) - (pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]])

    # Step 4: Combine all series into a single PCA matrix
    pca_df_combined = pd.concat([pca_df_zeros, pca_df_spreads, pca_df_flies], axis=1)

    # Fill NaN values with the column mean (after calculation)
    pca_vals = pca_df_combined.values.astype(float)
    col_means = np.nanmean(pca_vals, axis=0)
    if np.isnan(col_means).any():
        overall_mean = np.nanmean(col_means[~np.isnan(col_means)]) if np.any(~np.isnan(col_means)) else 0.0
        col_means = np.where(np.isnan(col_means), overall_mean, col_means)
    inds = np.where(np.isnan(pca_vals))
    if inds[0].size > 0:
        pca_vals[inds] = np.take(col_means, inds[1])

    return pd.DataFrame(pca_vals, index=pca_df_combined.index, columns=pca_df_combined.columns)

def calculate_raw_metrics(dt, row_series, available_contracts, expiry_df, rate_unit, holidays_np, year_basis):
    """
    Calculates the 'Actual Curve' for the forecast date by mapping standard maturities
    to the first, second, third, etc. available live contracts (simple sequential mapping).
    """
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
    
    raw_data = []
    for col in available_contracts:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index: continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() <= dt.date(): continue
        raw_val = row_series.get(col, np.nan)
        if pd.isna(raw_val): continue
        rate_percent = denormalize_to_percent(normalize_rate_input(raw_val, rate_unit))
        raw_data.append({'maturity': mat_up, 'expiry': exp, 'rate': rate_percent})

    raw_data.sort(key=lambda x: x['expiry'])
    
    rates_map = {std_arr[i]: np.nan for i in range(len(std_arr))}
    for i in range(len(std_arr)):
        if i < len(raw_data):
            rates_map[std_arr[i]] = raw_data[i]['rate']
    
    rates = np.array([rates_map.get(t, np.nan) for t in std_arr])
    
    spreads = np.full(len(std_arr) - 1, np.nan)
    flies = np.full(len(std_arr) - 2, np.nan)

    for i in range(len(spreads)):
        rate1 = rates[i+1]
        rate2 = rates[i]
        if not np.isnan(rate1) and not np.isnan(rate2):
            spreads[i] = rate1 - rate2

    for i in range(len(flies)):
        rate1 = rates[i+2]
        rate2 = rates[i+1]
        rate3 = rates[i]
        if not np.isnan(rate1) and not np.isnan(rate2) and not np.isnan(rate3):
            flies[i] = (rate1 - rate2) - (rate2 - rate3)
            
    return rates, spreads, flies

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

def transform_curve_for_display(curve_vector, all_cols_full, display_unit):
    """
    Transforms the full curve vector (Rates, Spreads, Flies) from Rate-space 
    to Price-space (or keeps it in Rate-space) for visualization.
    """
    if display_unit == "Rates (%)":
        return curve_vector

    rates_cols = [c for c in all_cols_full if '-' not in c]
    diff_cols = [c for c in all_cols_full if '-' in c]
    
    rates_indices = np.where(np.isin(all_cols_full, rates_cols))[0]
    diff_indices = np.where(np.isin(all_cols_full, diff_cols))[0]

    transformed_curve = curve_vector.copy()

    # 1. Transform Rates to Price (Price = 100 - Rate)
    transformed_curve[rates_indices] = 100.0 - transformed_curve[rates_indices]

    # 2. Transform Spreads/Flies (Differences) by negating them
    # Price Spread = P_B - P_A = (100-R_B) - (100-R_A) = R_A - R_B = - (R_B - R_A) = - (Rate Spread)
    transformed_curve[diff_indices] = -transformed_curve[diff_indices]

    return transformed_curve
# ---------------------------------------------------------------------------------

def main():
    st.title("🏛️ DI Curve Backtesting Engine")
    st.markdown("This tool performs a walk-forward backtest of PCA-based curve forecasting models.")
    st.markdown("---")

    # --- 1) File Uploads ---
    st.sidebar.header("1) Upload Data")
    yield_file = st.sidebar.file_uploader("Yield data CSV", type="csv")
    expiry_file = st.sidebar.file_uploader("Expiry mapping CSV", type="csv")
    holiday_file = st.sidebar.file_uploader("Holiday dates CSV (optional)", type="csv")

    # --- 2) Model and Backtest Config ---
    st.sidebar.header("2) Configure Backtest")
    # Using specific dates here for robustness, you may change these
    backtest_start_date = st.sidebar.date_input("Backtest Start Date", pd.to_datetime('2024-01-01').date()) 
    backtest_end_date = st.sidebar.date_input("Backtest End Date", pd.to_datetime('2024-03-31').date())
    training_window_days = st.sidebar.number_input("Rolling Training Window (business days)", min_value=120, max_value=500, value=252, step=1)
    
    st.sidebar.subheader("Model Parameters")
    n_components_sel = st.sidebar.slider("Number of PCA components", 1, 10, 3)
    forecast_model_type = st.sidebar.selectbox("Forecasting Model to Test", ["PCA Fair Value", "VAR (Vector Autoregression)", "ARIMA (per Component)"])
    var_lags = 1
    if forecast_model_type == "VAR (Vector Autoregression)":
        var_lags = st.sidebar.number_input("VAR model lags", min_value=1, max_value=20, value=1, step=1)
        
    st.sidebar.subheader("Data Conventions")
    rate_unit = st.sidebar.selectbox("Input rate unit", ["Percent (e.g. 13.45)", "Decimal (e.g. 0.1345)", "Basis points (e.g. 1345)"])
    year_basis = int(st.sidebar.selectbox("Business days in year", [252, 360], index=0))
    interp_method = "linear"

    # --- 3) Display Unit Selector ---
    st.sidebar.header("3) Visualization Options")
    display_unit = st.sidebar.selectbox("Display Results as", ["Rates (%)", "Price (Index)"])

    # --- Initialize session state ---
    if 'results_df' not in st.session_state: st.session_state.results_df = None
    if 'unique_dates' not in st.session_state: st.session_state.unique_dates = []
    if 'selected_date_index' not in st.session_state: st.session_state.selected_date_index = 0
    if 'selected_spread_date_index' not in st.session_state: st.session_state.selected_spread_date_index = 0
    
    # --- Callbacks for Next/Prev Buttons and Syncing (FIXED) ---
    def next_date():
        """Updates the index for the Outright Rates visualization."""
        if st.session_state.selected_date_index < len(st.session_state.unique_dates) - 1:
            st.session_state.selected_date_index += 1

    def prev_date():
        """Updates the index for the Outright Rates visualization."""
        if st.session_state.selected_date_index > 0:
            st.session_state.selected_date_index -= 1

    def next_spread_date():
        """Updates the index for the Spreads/Flies visualization."""
        if st.session_state.selected_spread_date_index < len(st.session_state.unique_dates) - 1:
            st.session_state.selected_spread_date_index += 1

    def prev_spread_date():
        """Updates the index for the Spreads/Flies visualization."""
        if st.session_state.selected_spread_date_index > 0:
            st.session_state.selected_spread_date_index -= 1
            
    def sync_rates_index():
        """Updates the rates index when the selectbox is manually changed."""
        unique_dates = st.session_state.unique_dates
        if unique_dates.size > 0 and st.session_state.rates_date_select in unique_dates:
            st.session_state.selected_date_index = unique_dates.tolist().index(st.session_state.rates_date_select)

    def sync_spreads_index():
        """Updates the spreads index when the selectbox is manually changed."""
        unique_dates = st.session_state.unique_dates
        if unique_dates.size > 0 and st.session_state.spreads_date_select in unique_dates:
            st.session_state.selected_spread_date_index = unique_dates.tolist().index(st.session_state.spreads_date_select)

    # --- Run/Reset Buttons ---
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    run_backtest = col1.button("Run Backtest", type="primary")

    if col2.button("Reset"):
        st.session_state.results_df = None
        st.session_state.selected_date_index = 0
        st.session_state.selected_spread_date_index = 0
        st.rerun()

    # --- Backtest Execution ---
    if run_backtest and st.session_state.results_df is None:
        if not all([yield_file, expiry_file]):
            st.error("Please upload both Yield and Expiry data files.")
            st.stop()
        if backtest_start_date >= backtest_end_date:
            st.error("Backtest Start Date must be before the End Date.")
            st.stop()

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

        yields_df, expiry_df, holidays_np = load_all_data(yield_file, expiry_file, holiday_file)

        # --- BACKTESTING LOOP ---
        st.subheader("Running Walk-Forward Backtest...")
        
        results = []
        backtest_range = pd.bdate_range(start=backtest_start_date, end=backtest_end_date)
        all_available_dates = yields_df.index

        # Setup grid
        std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
        std_cols = [f"{m:.2f}Y" for m in std_arr]
        spread_cols_pca = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
        fly_cols_pca = [f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
        all_cols_full = std_cols + spread_cols_pca + fly_cols_pca

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, current_date in enumerate(backtest_range):
            if current_date not in all_available_dates: continue

            training_end_date = current_date - pd.Timedelta(days=1)
            training_start_date = training_end_date - pd.DateOffset(days=training_window_days * 1.5)
            
            train_mask = (yields_df.index >= training_start_date) & (yields_df.index <= training_end_date)
            yields_df_train = yields_df.loc[train_mask].sort_index().tail(training_window_days)

            if len(yields_df_train) < training_window_days / 2: continue

            pca_df_filled = build_pca_matrix(yields_df_train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method)
            if pca_df_filled.empty: continue

            # PCA Decomposition and Training
            scaler = StandardScaler(with_std=False)
            X = scaler.fit_transform(pca_df_filled.values.astype(float))
            n_components_sel_capped = min(n_components_sel, X.shape[1])
            pca = PCA(n_components=n_components_sel_capped)
            PCs = pca.fit_transform(X)
            pc_cols = [f"PC{i+1}" for i in range(n_components_sel_capped)]
            PCs_df = pd.DataFrame(PCs, index=pca_df_filled.index, columns=pc_cols)

            # Forecasting (in rate space)
            if forecast_model_type == "PCA Fair Value":
                last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
                reconstructed_centered = pca.inverse_transform(last_pcs)
                pred_full_curve = scaler.inverse_transform(reconstructed_centered).flatten()
            else:
                pcs_next = forecast_pcs_var(PCs_df, lags=var_lags) if forecast_model_type == "VAR (Vector Autoregression)" else forecast_pcs_arima(PCs_df)
                last_actual_full_curve = pca_df_filled.iloc[-1].values
                last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
                delta_pcs = pcs_next - last_pcs
                delta_curve = pca.inverse_transform(delta_pcs)
                pred_full_curve = last_actual_full_curve + delta_curve.flatten()

            # Build the ACTUAL curve (in rate space)
            actual_rates_raw, actual_spreads_raw, actual_flies_raw = calculate_raw_metrics(
                current_date, yields_df.loc[current_date], yields_df.columns, expiry_df, rate_unit, holidays_np, year_basis
            )
            actual_full_curve = np.concatenate([actual_rates_raw, actual_spreads_raw, actual_flies_raw])
            
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
    
    # --------------------------
    # RESULTS ANALYSIS
    # --------------------------
    if st.session_state.results_df is not None:
        st.header("📊 Backtest Results Analysis")
        results_df = st.session_state.results_df.copy()
        
        mask = results_df['Actual_Curve'].apply(lambda x: isinstance(x, float) or np.isnan(x).all())
        results_df = results_df[~mask].copy()

        if results_df.empty:
            st.error("No valid results to analyze after cleaning. Check data availability and backtest range.")
            st.stop()

        all_cols = results_df.iloc[0]['Column_Names']
        std_cols_f = [c for c in all_cols if '-' not in c]
        spread_cols_f = [c for c in all_cols if '-' in c and len(c.split('-')) == 2]
        fly_cols_f = [c for c in all_cols if '-' in c and len(c.split('-')) == 3]

        # Function to calculate errors (magnitude is the same in Rate and Price space)
        def calculate_errors(row, cols):
            indices = np.where(np.isin(all_cols, cols))[0]
            pred = row['Predicted_Curve'][indices]
            actual = row['Actual_Curve'][indices]
            
            if np.isnan(actual).all() or len(actual) == 0: return pd.Series({'RMSE': np.nan, 'MAE': np.nan})
            return pd.Series({
                'RMSE': np.sqrt(np.nanmean((pred - actual)**2)),
                'MAE': np.nanmean(np.abs(pred - actual))
            })
        
        # Errors are calculated on the rate-space curve 
        results_df[['Daily_RMSE_Rates', 'Daily_MAE_Rates']] = results_df.apply(lambda row: calculate_errors(row, std_cols_f), axis=1)
        results_df[['Daily_RMSE_Spreads', 'Daily_MAE_Spreads']] = results_df.apply(lambda row: calculate_errors(row, spread_cols_f), axis=1)
        results_df[['Daily_RMSE_Flies', 'Daily_MAE_Flies']] = results_df.apply(lambda row: calculate_errors(row, fly_cols_f), axis=1)

        unit_label = "Price (Index)" if display_unit == "Price (Index)" else "Rate (%)"

        # --------------------------
        # Outright Rates Results
        # --------------------------
        st.subheader(f"1. Outright {unit_label} Performance")
        
        col1, col2 = st.columns(2)
        col1.metric(f"Avg Daily Curve RMSE ({unit_label})", f"{results_df['Daily_RMSE_Rates'].mean():.4f}")
        col2.metric(f"Avg Daily Curve MAE ({unit_label})", f"{results_df['Daily_MAE_Rates'].mean():.4f}")

        st.markdown("---")
        st.subheader(f"Daily Curve Visualization (Outright {unit_label})")

        unique_dates = results_df['Date'].dt.date.unique()
        st.session_state.unique_dates = unique_dates

        prev_col, date_col, next_col = st.columns([1, 4, 1])
        
        prev_col.button("◀ Previous", key="rates_prev", on_click=prev_date)
        next_col.button("Next ▶", key="rates_next", on_click=next_date)
        
        selected_date = date_col.selectbox(
            "Select a date to inspect",
            options=unique_dates,
            index=st.session_state.selected_date_index,
            key="rates_date_select",
            on_change=sync_rates_index
        )
        
        if selected_date:
            plot_data = results_df[results_df['Date'].dt.date == selected_date]
            if not plot_data.empty:
                full_actual_curve_vector = plot_data['Actual_Curve'].iloc[0]
                full_pred_curve_vector = plot_data['Predicted_Curve'].iloc[0]

                # --- Transform to display unit ---
                actual_transformed = transform_curve_for_display(full_actual_curve_vector, all_cols, display_unit)
                pred_transformed = transform_curve_for_display(full_pred_curve_vector, all_cols, display_unit)
                
                # Slicing for outright rates
                indices = np.where(np.isin(all_cols, std_cols_f))[0]
                actual_c = actual_transformed[indices]
                pred_c = pred_transformed[indices]

                std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)

                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(std_arr, actual_c, label=f"Actual on {selected_date}", color='royalblue', marker='o', linestyle='-')
                ax.plot(std_arr, pred_c, label=f"Predicted for {selected_date}", color='darkorange', marker='x', linestyle='--')
                
                ax.set_title(f"Yield Curve Forecast vs. Actual for {selected_date} (Outright {unit_label})", fontsize=16)
                ax.set_xlabel("Standardized Maturity (Years)")
                ax.set_ylabel(unit_label)
                ax.set_xticks(std_arr)
                ax.set_xticklabels([f"{m:.2f}Y" for m in std_arr], rotation=45, ha="right")
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)

        # --------------------------
        # Spreads and Flies Results
        # --------------------------
        st.markdown("---")
        st.subheader(f"2. Spreads and Flies Performance ({unit_label} Differences)")
        
        col3, col4 = st.columns(2)
        col3.metric(f"Avg Daily Curve RMSE (Spreads, {unit_label})", f"{results_df['Daily_RMSE_Spreads'].mean():.4f}")
        col4.metric(f"Avg Daily Curve MAE (Flies, {unit_label})", f"{results_df['Daily_MAE_Flies'].mean():.4f}")

        st.markdown("---")
        st.subheader(f"Daily Visualization (Spreads and Flies - {unit_label} Differences)")

        prev_col, spread_date_col, next_col = st.columns([1, 4, 1])
        
        prev_col.button("◀ Previous", key="spread_prev", on_click=prev_spread_date)
        next_col.button("Next ▶", key="spread_next", on_click=next_spread_date)
        
        selected_spread_date = spread_date_col.selectbox(
            "Select a date to inspect spreads/flies",
            options=unique_dates,
            index=st.session_state.selected_spread_date_index,
            key="spreads_date_select",
            on_change=sync_spreads_index
        )
        
        if selected_spread_date:
            plot_data = results_df[results_df['Date'].dt.date == selected_spread_date]
            if not plot_data.empty:
                full_actual_curve_vector = plot_data['Actual_Curve'].iloc[0]
                full_pred_curve_vector = plot_data['Predicted_Curve'].iloc[0]
                
                # --- Transform to display unit ---
                actual_transformed = transform_curve_for_display(full_actual_curve_vector, all_cols, display_unit)
                pred_transformed = transform_curve_for_display(full_pred_curve_vector, all_cols, display_unit)

                # Slicing for spreads/flies
                spread_indices = np.where(np.isin(all_cols, spread_cols_f))[0]
                actual_spreads = actual_transformed[spread_indices]
                pred_spreads = pred_transformed[spread_indices]
                
                fly_indices = np.where(np.isin(all_cols, fly_cols_f))[0]
                actual_flies = actual_transformed[fly_indices]
                pred_flies = pred_transformed[fly_indices]

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Spreads Plot
                ax1.plot(spread_cols_f, actual_spreads, marker='o', linestyle='-', color='royalblue', label=f"Actual Spreads")
                ax1.plot(spread_cols_f, pred_spreads, marker='x', linestyle='--', color='darkorange', label="Predicted Spreads")
                ax1.set_title(f"Spread Forecast vs. Actual for {selected_spread_date}")
                ax1.set_ylabel(f"Spread ({unit_label} Diff)")
                ax1.set_xticklabels(spread_cols_f, rotation=45, ha="right")
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax1.legend()
                
                # Flies Plot
                ax2.plot(fly_cols_f, actual_flies, marker='o', linestyle='-', color='royalblue', label=f"Actual Flies")
                ax2.plot(fly_cols_f, pred_flies, marker='x', linestyle='--', color='darkorange', label="Predicted Flies")
                ax2.set_title(f"Butterfly Forecast vs. Actual for {selected_spread_date}")
                ax2.set_ylabel(f"Fly ({unit_label} Diff)")
                ax2.set_xticklabels(fly_cols_f, rotation=45, ha="right")
                ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)

        st.markdown("---")
        st.subheader(f"Raw Results Dataframe (Full Curve in {unit_label})")
        st.write("Contains the raw predicted and actual outright rates, spread, and fly vectors after conversion to the selected display unit.")

        raw_df = pd.DataFrame(results_df['Date']).set_index('Date')
        
        # Apply transformation to the entire results dataframe before extraction
        results_df['Predicted_Display'] = results_df.apply(lambda row: transform_curve_for_display(row['Predicted_Curve'], all_cols, display_unit), axis=1)
        results_df['Actual_Display'] = results_df.apply(lambda row: transform_curve_for_display(row['Actual_Curve'], all_cols, display_unit), axis=1)

        # Extract transformed data
        rates_start_idx = 0
        spreads_start_idx = len(std_cols_f)
        flies_start_idx = spreads_start_idx + len(spread_cols_f)

        for i, col in enumerate(std_cols_f):
            raw_df[f"Predicted_{col}"] = results_df['Predicted_Display'].apply(lambda x: x[rates_start_idx + i])
            raw_df[f"Actual_{col}"] = results_df['Actual_Display'].apply(lambda x: x[rates_start_idx + i])

        for i, col in enumerate(spread_cols_f):
            raw_df[f"Predicted_Spread_{col}"] = results_df['Predicted_Display'].apply(lambda x: x[spreads_start_idx + i])
            raw_df[f"Actual_Spread_{col}"] = results_df['Actual_Display'].apply(lambda x: x[spreads_start_idx + i])
            
        for i, col in enumerate(fly_cols_f):
            raw_df[f"Predicted_Fly_{col}"] = results_df['Predicted_Display'].apply(lambda x: x[flies_start_idx + i])
            raw_df[f"Actual_Fly_{col}"] = results_df['Actual_Display'].apply(lambda x: x[flies_start_idx + i])

        st.dataframe(raw_df)

    else:
        st.info("Upload files, configure the backtest parameters, and click **Run Backtest**.")

# --------------------------
# FIX: Call the main function to run the Streamlit UI
# --------------------------
if __name__ == '__main__':
    main()
