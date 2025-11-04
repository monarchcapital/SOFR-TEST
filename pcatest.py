# -*- coding: utf-8 -*-
# SOFR Futures PCA — Backtesting Engine
#
# CRITICAL FIX: The backtesting loop now projects the forecasted Standard Grid 
# curve onto the *actual* Time-to-Maturity (TTM) of the contracts trading on 
# the forecast date (t+1), providing a true prediction of the daily market curve.
#
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
from datetime import timedelta, date as dt_date

# --- Setup ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="SOFR Curve Backtesting Engine")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------------
# CORE UTILITY FUNCTIONS
# ---------------------------------------------------------------------------------
def safe_to_datetime(s):
    return pd.to_datetime(s, errors='coerce', dayfirst=False)

def denormalize_to_percent(frac):
    if pd.isna(frac): return np.nan
    return 100.0 * float(frac)

def np_busdays_exclusive(start_dt, end_dt, holidays_np):
    if pd.isna(start_dt) or pd.isna(end_dt): return 0
    s = np.datetime64(pd.Timestamp(start_dt).date()) + np.timedelta64(1, "D")
    e = np.datetime64(pd.Timestamp(end_dt).date())
    if e < s: return 0
    return int(np.busday_count(s, e, weekmask="1111100", holidays=holidays_np))

def calculate_ttm(valuation_ts, expiry_ts, holidays_np, year_basis):
    bd = np_busdays_exclusive(valuation_ts, expiry_ts, holidays_np, )
    return np.nan if bd <= 0 else bd / float(year_basis)

@st.cache_data(show_spinner=False)
def _quick_read_csv(upload):
    return pd.read_csv(io.StringIO(upload.getvalue().decode("utf-8")))

def build_std_grid_by_rule(max_year=7.0):
    a = list(np.round(np.arange(0.25, 3.0 + 0.001, 0.25), 2))
    b = list(np.round(np.arange(3.5, 5.0 + 0.001, 0.5), 2))
    c = list(np.round(np.arange(6.0, max_year + 0.001, 1.0), 2))
    return a + b + c

# ---------------------------------------------------------------------------------
# SOFR-ADAPTED CORE LOGIC FUNCTIONS
# ---------------------------------------------------------------------------------

def row_to_std_grid(dt, row_series, available_contracts, expiry_df, std_arr, holidays_np, year_basis, interp_method):
    ttm_list, zero_list = [], []
    for col in available_contracts:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index: continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date(): continue
        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0: continue
        
        # raw_val is the futures PRICE (e.g., 95.2275)
        raw_val = row_series.get(col, np.nan)
        if pd.isna(raw_val): continue

        # CRITICAL SOFR LOGIC: PRICE (P) to ANNUALIZED ZERO RATE (R_zero)
        rate_percent = 100.0 - float(raw_val)
        zero_frac = rate_percent / 100.0
        
        ttm_list.append(t)
        # Store the rate in percentage form for the interpolation grid (e.g., 4.50)
        zero_list.append(denormalize_to_percent(zero_frac))
        
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        try:
            order = np.argsort(ttm_list)
            # Use the market rates (zero_list) at their market TTMs (ttm_list) to build the curve
            f = interp1d(
                np.array(ttm_list)[order],
                np.array(zero_list)[order],
                kind=interp_method,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            # Interpolate the curve onto the fixed standard grid (std_arr)
            return f(std_arr)
        except Exception:
            return np.full_like(std_arr, np.nan, dtype=float)
    return np.full_like(std_arr, np.nan, dtype=float)


def calculate_spreads_flies(std_curve_rates, std_cols):
    # Calculates spreads and flies from a standard zero rate curve vector
    std_curve_series = pd.Series(std_curve_rates, index=std_cols)
    
    spread_vals = std_curve_series[std_cols[1:]].values - std_curve_series[std_cols[:-1]].values
    flies_vals = (std_curve_series[std_cols[2:]].values - std_curve_series[std_cols[1:-1]].values) - \
                 (std_curve_series[std_cols[1:-1]].values - std_curve_series[std_cols[:-2]].values)

    return spread_vals, flies_vals


def build_pca_matrix(yields_df_train, expiry_df, std_arr, std_cols, holidays_np, year_basis, interp_method):
    # Step 1: Create a DataFrame for zero rates on the standard grid
    pca_df_zeros = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    available_contracts = yields_df_train.columns
    
    for dt in yields_df_train.index:
        pca_df_zeros.loc[dt] = row_to_std_grid(
            dt, yields_df_train.loc[dt], available_contracts, expiry_df, 
            std_arr, holidays_np, year_basis, interp_method
        )
    
    # Step 2 & 3: Calculate Spreads and Butterflies
    spread_cols = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
    fly_cols = [f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
    
    # Calculate spreads and flies for all historical days
    pca_df_spreads = pd.DataFrame(index=pca_df_zeros.index, columns=spread_cols, dtype=float)
    pca_df_flies = pd.DataFrame(index=pca_df_zeros.index, columns=fly_cols, dtype=float)

    for dt in pca_df_zeros.index:
        std_curve_rates = pca_df_zeros.loc[dt].values
        spread_vals, flies_vals = calculate_spreads_flies(std_curve_rates, std_cols)
        pca_df_spreads.loc[dt] = spread_vals
        pca_df_flies.loc[dt] = flies_vals

    # Step 4: Combine all series and drop rows where ALL columns are NaN
    pca_df_combined = pd.concat([pca_df_zeros, pca_df_spreads, pca_df_flies], axis=1).dropna(how='all')

    # Step 5: Impute remaining NaNs with the column mean (creates the filled matrix)
    pca_vals = pca_df_combined.values.astype(float)
    col_means = np.nanmean(pca_vals, axis=0)
    
    # Simple imputation: fill any remaining NaNs with overall mean
    if np.isnan(col_means).any():
        overall_mean = np.nanmean(col_means[~np.isnan(col_means)]) if np.any(~np.isnan(col_means)) else 0.0
        col_means = np.where(np.isnan(col_means), overall_mean, col_means)
    
    inds = np.where(np.isnan(pca_vals))
    if inds[0].size > 0:
        pca_vals[inds] = np.take(col_means, inds[1])
    
    return pd.DataFrame(pca_vals, index=pca_df_combined.index, columns=pca_df_combined.columns), spread_cols, fly_cols


# ---------------------------------------------------------------------------------
# FORECASTING UTILITIES (Unchanged)
# ---------------------------------------------------------------------------------
def forecast_pcs_avg_delta(PCs_matrix, window=5, pc_damp=0.5):
    if PCs_matrix.shape[0] == 0: return np.zeros((1, PCs_matrix.shape[1]))
    n, k = PCs_matrix.shape
    w = int(min(max(1, window), n))
    last = PCs_matrix[-1].reshape(1, -1)
    if w <= 1 or n == 1: avg_delta = np.zeros((k,))
    else: avg_delta = np.mean(np.diff(PCs_matrix[-w:], axis=0), axis=0)
    damp = np.ones(k)
    if k > 1: damp[1:] = float(pc_damp)
    return last + (avg_delta * damp).reshape(1, -1)

def forecast_pcs_var(PCs_df, lags=1):
    if len(PCs_df) < lags + 5:
        return PCs_df.iloc[-1:].values
    results = VAR(PCs_df).fit(lags)
    return results.forecast(PCs_df.values[-lags:], steps=1)

def forecast_pcs_arima(PCs_df):
    forecasts = []
    for name, series in PCs_df.items():
        if len(series) < 10:
            forecasts.append(series.iloc[-1])
            continue
        try:
             forecasts.append(ARIMA(series, order=(1, 1, 0)).fit().forecast(steps=1).iloc[0])
        except Exception:
             forecasts.append(series.iloc[-1]) # Fallback on ARIMA fail
    return np.array(forecasts).reshape(1, -1)


# ---------------------------------------------------------------------------------
# STREAMLIT UI & LOGIC
# ---------------------------------------------------------------------------------

st.title("SOFR Futures PCA Backtesting Engine 📈")
st.markdown("This tool performs a walk-forward validation of the PCA curve forecasting model.")

# --- Sidebar Inputs ---
st.sidebar.header("Backtest Data")
yield_file = st.sidebar.file_uploader("1) Price/Rate data CSV", type="csv", key="yield_file_bt")
expiry_file = st.sidebar.file_uploader("2) Expiry mapping CSV", type="csv", key="expiry_file_bt")
holiday_file = st.sidebar.file_uploader("3) Holiday dates CSV (optional)", type="csv", key="holiday_file_bt")

# Note: These rate/compounding options are largely ignored as SOFR P->R is fixed, 
# but retained for historical compatibility with the original DI code structure.
rate_unit = st.sidebar.selectbox(
    "Input rate unit (Select decimal, but value is treated as Futures PRICE)",
    ["Percent (e.g. 4.50)", "Decimal (e.g. 0.0450)", "Basis points (e.g. 450)"],
    index=1,
    key="rate_unit_bt"
)
year_basis = int(st.sidebar.selectbox("Business days in year", [252, 360], index=0, key="year_basis_bt"))
interp_method = st.sidebar.selectbox("Interpolation method", ["linear", "cubic", "quadratic", "nearest"], key="interp_method_bt")
n_components_sel = st.sidebar.slider("Number of PCA components", 1, 12, 3, key="n_components_sel_bt")

compounding_model = st.sidebar.radio(
    "Compounding model",
    ["identity (Futures Rate R = 100-P assumed as Annual Zero)", "Custom Annual Zero (Requires Re-Compounding)"],
    index=0,
    key="compounding_model_bt"
)

use_grid_rule = st.sidebar.checkbox("Use standard grid rule", value=True, key="use_grid_rule_bt")
std_maturities_txt = st.sidebar.text_input(
    "Or custom standard maturities (years, comma-separated)",
    "0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.50,4.00,4.50,5.00,6.00,7.00",
    key="std_maturities_txt_bt"
)

st.sidebar.markdown("---")
st.sidebar.header("Backtest Configuration")
pca_window = st.sidebar.number_input("PCA training window (days)", min_value=30, max_value=500, value=252, step=1, key="pca_window_bt")
forecast_model_type = st.sidebar.selectbox(
    "Forecasting Model",
    ["None (PCA Fair Value)", "Average Delta (Momentum)", "VAR (Vector Autoregression)", "ARIMA (per Component)"],
    key="forecast_model_type_bt"
)

# --- Model-specific parameters ---
if forecast_model_type == "Average Delta (Momentum)":
    rolling_window_days = st.sidebar.number_input("PCs avg-delta window (days)", min_value=1, max_value=20, value=5, step=1, key="rolling_window_days_bt")
    pc_damp = st.sidebar.slider("Damping for PC2..PCn (0 = ignore, 1 = full)", 0.0, 1.0, 0.5, 0.05, key="pc_damp_bt")
elif forecast_model_type == "VAR (Vector Autoregression)":
    var_lags = st.sidebar.number_input("VAR model lags", min_value=1, max_value=20, value=1, step=1, key="var_lags_bt")

# --- Date Inputs ---
# ... (Date input logic remains the same) ...
yields_df_all = None
if yield_file is not None:
    try:
        tmp = _quick_read_csv(yield_file)
        if not tmp.empty:
            dc = tmp.columns[0]
            tmp[dc] = safe_to_datetime(tmp[dc])
            tmp = tmp.dropna(subset=[dc]).set_index(dc)
            yields_df_all = tmp.sort_index()
    except Exception as e:
        st.error(f"Error reading yield file: {e}")

if yields_df_all is not None and not yields_df_all.empty:
    min_date_available = yields_df_all.index.min().date()
    max_date_available = yields_df_all.index.max().date()
else:
    min_date_available = dt_date(2000, 1, 1)
    max_date_available = dt_date(2100, 1, 1)

date_backtest_start = st.sidebar.date_input("Backtest Start Date", value=min_date_available, min_value=min_date_available, max_value=max_date_available, key="bt_start_date")
date_backtest_end = st.sidebar.date_input("Backtest End Date", value=max_date_available, min_value=min_date_available, max_value=max_date_available, key="bt_end_date")

if date_backtest_start > date_backtest_end:
    st.sidebar.error("Backtest Start Date cannot be after End Date.")

inputs_ready = (yield_file is not None and expiry_file is not None and date_backtest_start <= date_backtest_end)

if st.sidebar.button("Run Backtest"):
    if not inputs_ready:
        st.error("Please upload all required files and ensure dates are correct.")
        st.stop()
    
    # ---------------------------------------------------------------------------------
    # DATA LOADING AND INITIALIZATION
    # ---------------------------------------------------------------------------------
    @st.cache_data
    def load_files(yield_f, expiry_f, holiday_f):
        yields_df = _quick_read_csv(yield_f)
        date_col = yields_df.columns[0]
        yields_df[date_col] = safe_to_datetime(yields_df[date_col])
        yields_df = yields_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        for c in yields_df.columns:
            yields_df[c] = pd.to_numeric(yields_df[c], errors="coerce")
        
        expiry_raw = _quick_read_csv(expiry_f)
        expiry_df = expiry_raw.iloc[:, :2].copy()
        expiry_df.columns = ["MATURITY", "DATE"]
        expiry_df["MATURITY"] = expiry_df["MATURITY"].astype(str).str.strip().str.upper()
        expiry_df["DATE"] = safe_to_datetime(expiry_df["DATE"])
        expiry_df = expiry_df.dropna(subset=["DATE"]).set_index("MATURITY")
        
        holidays_np = np.array([], dtype="datetime64[D]")
        if holiday_f:
            hol_df = _quick_read_csv(holiday_f)
            hol_series = safe_to_datetime(hol_df.iloc[:, 0]).dropna()
            if not hol_series.empty:
                holidays_np = np.array(hol_series.dt.date, dtype="datetime64[D]")
        return yields_df, expiry_df, holidays_np

    yields_df_all, expiry_df, holidays_np = load_files(yield_file, expiry_file, holiday_file)

    # Standard maturities grid setup
    if use_grid_rule:
        std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
    else:
        try:
            std_maturities = [float(x.strip()) for x in std_maturities_txt.split(",") if x.strip() != ""]
            std_arr = np.array(sorted(std_maturities), dtype=float)
        except Exception:
            st.error("Error parsing standard maturities.")
            st.stop()
    std_cols = [f"{m:.2f}Y" for m in std_arr]
    
    # --- Backtest setup ---
    backtest_dates = yields_df_all.index.normalize()
    backtest_dates = backtest_dates[(backtest_dates.date >= date_backtest_start) & (backtest_dates.date <= date_backtest_end)].unique().sort_values()
    
    if len(backtest_dates) < 2:
        st.error("Not enough data points in the selected backtesting range.")
        st.stop()

    start_idx = pca_window # Start the backtest only after the initial training window is available
    if start_idx >= len(backtest_dates):
        st.error(f"PCA window ({pca_window} days) is longer than the available data ({len(backtest_dates)} days).")
        st.stop()

    results = []
    
    # Get the unique list of contract tickers for consistent output columns
    all_contracts = sorted([c for c in yields_df_all.columns if c in expiry_df.index])

    # ---------------------------------------------------------------------------------
    # WALK-FORWARD VALIDATION LOOP
    # ---------------------------------------------------------------------------------
    st.subheader(f"Running Walk-Forward Backtest ({forecast_model_type})")
    progress_bar = st.progress(0)
    
    for i in range(start_idx, len(backtest_dates) - 1): 
        
        train_end_date = backtest_dates[i]
        forecast_date = backtest_dates[i+1] # The next business day in the dataset

        # 1. Define Training Window
        train_start_date = backtest_dates[i - pca_window + 1]
        train_df = yields_df_all.loc[train_start_date:train_end_date].copy()
        
        # 2. Build PCA Matrix (Rates, Spreads, Flies on Standard Grid)
        try:
            pca_df_filled, spread_cols, fly_cols = build_pca_matrix(
                train_df, expiry_df, std_arr, std_cols, holidays_np, year_basis, 
                interp_method
            )
        except Exception as e:
            # We skip days where the training set is too sparse
            st.warning(f"Skipping {train_end_date.date()} due to PCA matrix build error: {e}")
            continue

        if train_end_date not in pca_df_filled.index:
             st.warning(f"Skipping {train_end_date.date()}. It was dropped during PCA matrix cleaning.")
             continue 

        # 3. Run PCA and Calculate Last Actual Curve (Input for Forecast)
        scaler = StandardScaler(with_std=False)
        X = scaler.fit_transform(pca_df_filled.values.astype(float))
        
        n_components = min(int(n_components_sel), X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        PCs = pca.transform(X)
        PCs_df = pd.DataFrame(PCs, index=pca_df_filled.index, columns=[f"PC{k+1}" for k in range(n_components)])
        
        last_actual_vals = pca_df_filled.loc[train_end_date].values # The imputed full vector (Std Rates, Spreads, Flies)

        # 4. Forecast the Next Day's FULL STANDARD CURVE VECTOR (Rates, Spreads, Flies)
        if forecast_model_type == "None (PCA Fair Value)":
            last_day_pcs_index = PCs_df.index.get_loc(train_end_date)
            # Reconstruct the last day's curve from PCs (the "Fair Value" forecast)
            predicted_standard_curve_vector = pca.inverse_transform(PCs[last_day_pcs_index].reshape(1, -1)).ravel()
            predicted_standard_curve_vector = scaler.inverse_transform(predicted_standard_curve_vector.reshape(1, -1)).ravel()
        else:
            if forecast_model_type == "Average Delta (Momentum)":
                pcs_next = forecast_pcs_avg_delta(PCs, window=int(rolling_window_days), pc_damp=float(pc_damp))
            elif forecast_model_type == "VAR (Vector Autoregression)":
                pcs_next = forecast_pcs_var(PCs_df, lags=int(var_lags))
            elif forecast_model_type == "ARIMA (per Component)":
                pcs_next = forecast_pcs_arima(PCs_df)
            
            # Use the forecasted PCs to calculate the delta in the underlying curve vector
            last_pcs = PCs_df.loc[train_end_date].values.reshape(1, -1)
            delta_pcs = pcs_next - last_pcs
            delta_vals = pca.inverse_transform(delta_pcs).flatten()
            
            # Add the delta to the last actual full vector
            predicted_standard_curve_vector = last_actual_vals + delta_vals
        
        # 5. Extract the forecasted Standard Rates (the interpolated yield curve)
        predicted_std_rates = predicted_standard_curve_vector[0:len(std_cols)] # In percentage (e.g., 4.50)

        # 6. CRITICAL FIX: PROJECT FORECASTED STANDARD CURVE ONTO ACTUAL CONTRACT MATURITIES
        # This is the step that translates the forecast from the abstract Standard Grid 
        # back into the observable market contracts for the forecast date.
        
        # a) Build the interpolation function (The Predicted Yield Curve for t+1)
        # Note: std_arr are the TTMs, predicted_std_rates are the predicted rates.
        f_predicted = interp1d(
            std_arr, predicted_std_rates,
            kind=interp_method,
            bounds_error=False,
            fill_value='extrapolate' # Extrapolate beyond the standard grid for safety
        )
        
        # b) Collect TTMs and actual rates for the forecast date (The Ground Truth)
        if forecast_date not in yields_df_all.index:
            st.warning(f"Skipping comparison for {forecast_date.date()}. No raw market data for this date.")
            continue
            
        raw_target_data = yields_df_all.loc[forecast_date]
        forecast_ttms, actual_rates_percent, contract_names = [], [], []
        
        for contract in all_contracts: # Use all contracts for consistent columns
            if contract not in expiry_df.index: continue
            exp = expiry_df.loc[contract, "DATE"]
            t = calculate_ttm(forecast_date, exp, holidays_np, year_basis)
            if np.isnan(t) or t <= 0: continue
            
            # Actual Futures Price for the forecast date
            raw_val = raw_target_data.get(contract, np.nan)
            
            if not pd.isna(raw_val):
                # Convert Price (P) to Rate in Percentage (R%) for the actual
                actual_rate_percent = 100.0 - float(raw_val)
                actual_rates_percent.append(actual_rate_percent)
            else:
                # If the actual rate is missing, mark it NaN
                actual_rates_percent.append(np.nan)

            forecast_ttms.append(t)
            contract_names.append(contract)
        
        if len(contract_names) < 2:
            st.warning(f"Skipping comparison for {forecast_date.date()}. Not enough contract data.")
            continue
        
        # c) Use the Predicted Curve to get rates at the actual TTMs
        predicted_rates_percent = f_predicted(forecast_ttms).tolist()

        # 7. Aggregate Results (on Contract Maturities)
        
        # The stored curves are now vectors of [Rate_C1, Rate_C2, ... Rate_Cn]
        # where C1..Cn are the SOFR contracts.
        results.append({
            "Date": forecast_date.date(),
            "Contract_TTMs": forecast_ttms,
            "Contract_Names": contract_names,
            "Predicted_Curve_Contract": np.array(predicted_rates_percent),
            "Actual_Curve_Contract": np.array(actual_rates_percent)
        })
        
        progress_bar.progress((i - start_idx) / (len(backtest_dates) - 1 - start_idx))

    progress_bar.empty()
    st.success(f"Backtest complete. {len(results)} days analyzed.")
    
    if not results:
        st.info("No successful backtest days. Please check data quality and date range.")
        st.stop()
    
    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------------------------------
    # PERFORMANCE METRICS (Now calculated based on Contract Rates)
    # ---------------------------------------------------------------------------------
    
    st.markdown("---")
    st.header("Backtest Performance Metrics (Contract Rate Error Analysis)")
    
    # Pad the predicted/actual arrays so they are all the same length for vstack
    max_len = max(len(row) for row in results_df['Actual_Curve_Contract'])
    def pad_array(arr):
        if len(arr) < max_len:
            # Pad with NaN for missing contracts on sparse days
            return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=np.nan)
        return arr
    
    actual_contract_rates = np.vstack(results_df['Actual_Curve_Contract'].apply(pad_array).tolist())
    predicted_contract_rates = np.vstack(results_df['Predicted_Curve_Contract'].apply(pad_array).tolist())
    
    # Identify the full list of contracts present across the backtest for metrics columns
    contract_cols = sorted(list(set(c for sublist in results_df['Contract_Names'] for c in sublist)))
    
    # Filter for non-NaN indices where both predicted and actual values exist
    valid_mask = ~np.isnan(actual_contract_rates) & ~np.isnan(predicted_contract_rates)
    
    # Calculate errors in Basis Points (bps)
    errors_bps = (predicted_contract_rates - actual_contract_rates) * 100 # In basis points (Rate in % * 100)

    # Overall RMSE and MAE across ALL contracts and ALL days (using only valid points)
    total_rmse = np.sqrt(np.nanmean(errors_bps**2))
    total_mae = np.nanmean(np.abs(errors_bps))

    st.markdown(f"""
    **Overall Performance:**
    - **Total RMSE:** **{total_rmse:.2f} bps**
    - **Total MAE:** **{total_mae:.2f} bps**
    """)
    
    st.markdown("##### Error by Contract")
    
    # Calculate RMSE/MAE for each contract (column)
    contract_rmse = np.sqrt(np.nanmean(errors_bps**2, axis=0))
    contract_mae = np.nanmean(np.abs(errors_bps), axis=0)

    metrics_by_contract = pd.DataFrame({
        "Contract": contract_cols,
        "RMSE (bps)": contract_rmse,
        "MAE (bps)": contract_mae
    }).set_index("Contract").dropna(how='all')

    st.dataframe(metrics_by_contract.style.format("{:.2f}"))

    # ---------------------------------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------------------------------
    st.markdown("---")
    st.header("Backtest Visualization")
    
    # Plot 1: Time series of prediction errors for a selected contract
    st.subheader("Prediction Error Time Series (by Contract)")
    
    selected_contract = st.selectbox("Select Contract Ticker to Plot", contract_cols)
    
    if selected_contract:
        contract_index = contract_cols.index(selected_contract)
        error_series = pd.Series(errors_bps[:, contract_index], index=results_df['Date'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        error_series.plot(ax=ax, linewidth=1.5, color='darkred')
        
        ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
        ax.set_title(f"Prediction Error Time Series for {selected_contract}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Error (Predicted Rate - Actual Rate, in bps)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Raw Backtest Data Output")
    st.write("Contains the predicted vs. actual **contract rates (in percent)** for each day.")
    
    # Prepare raw output table
    raw_output_df = pd.DataFrame({'Date': results_df['Date']}).set_index('Date')
    
    for i, contract in enumerate(contract_cols):
        raw_output_df[f"Actual_{contract}"] = [row[i] if i < len(row) and not np.isnan(row[i]) else np.nan for row in actual_contract_rates]
        raw_output_df[f"Predicted_{contract}"] = [row[i] if i < len(row) and not np.isnan(row[i]) else np.nan for row in predicted_contract_rates]

    st.dataframe(raw_output_df.tail(10).style.format("{:.3f}"))
    
    @st.cache_data
    def convert_df_for_download(df):
        return df.to_csv().encode('utf-8')

    csv_data = convert_df_for_download(raw_output_df)
    st.download_button(
        label="Download Full Backtest Contract Results CSV",
        data=csv_data,
        file_name='sofr_pca_contract_backtest_results.csv',
        mime='text/csv',
    )