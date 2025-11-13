# sofr_pca_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

# -----------------------------
# App config
# -----------------------------
st.set_page_config(layout="wide", page_title="SOFR Futures PCA Analyzer (Sections 1-9)")

BP_TO_USD = 25  # $ per basis point for SOFR futures - adjust if different

# -----------------------------
# Helper: Data loading & transforms
# -----------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        content = uploaded_file.getvalue().decode("utf-8")
        uploaded_file.seek(0)

        # Detect expiry file (header contains MATURITY,DATE)
        first_line = content.splitlines()[0].upper() if content else ""
        if 'MATURITY,DATE' in first_line or 'MATURITY' in first_line and 'DATE' in first_line:
            df = pd.read_csv(uploaded_file, sep=',')
            # normalize column names
            df = df.rename(columns={col: col.strip() for col in df.columns})
            if 'MATURITY' in df.columns and 'DATE' in df.columns:
                df = df.rename(columns={'MATURITY': 'Contract', 'DATE': 'ExpiryDate'})
            elif 'Contract' in df.columns and 'ExpiryDate' in df.columns:
                pass
            df = df.set_index('Contract')
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
            df.index.name = 'Contract'
            return df

        # Otherwise price file
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True, sep=',', header=0)
        df.index.name = 'Date'
        df = df.dropna(axis=1, how='all')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(how='all')
        df = df[df.index.notna()]
        if df.empty or df.shape[1] == 0:
            raise ValueError("DataFrame empty after processing.")
        return df

    except Exception as e:
        st.error(f"Error loading file {getattr(uploaded_file, 'name', '')}: {e}")
        return None


@st.cache_data
def get_analysis_contracts(expiry_df, analysis_date):
    if expiry_df is None:
        return pd.DataFrame()
    future_expiries = expiry_df[expiry_df['ExpiryDate'] >= analysis_date].copy()
    future_expiries = future_expiries.sort_values(by='ExpiryDate')
    return future_expiries


@st.cache_data
def transform_to_analysis_curve(price_df, future_expiries_df):
    if price_df is None or future_expiries_df.empty:
        return pd.DataFrame(), []
    contract_order = future_expiries_df.index.tolist()
    valid_contracts = [c for c in contract_order if c in price_df.columns]
    if not valid_contracts:
        return pd.DataFrame(), []
    analysis_curve_df = price_df[valid_contracts].copy()
    return analysis_curve_df, valid_contracts

# -----------------------------
# Derivative calculators
# -----------------------------
@st.cache_data
def calculate_k_step_spreads(analysis_curve_df, k):
    if analysis_curve_df is None or analysis_curve_df.empty or analysis_curve_df.shape[1] < k+1:
        return pd.DataFrame()
    num_contracts = analysis_curve_df.shape[1]
    spreads_data = {}
    cols = analysis_curve_df.columns.tolist()
    for i in range(num_contracts - k):
        short = cols[i]
        long = cols[i+k]
        label = f"{short}-{long}"
        spreads_data[label] = analysis_curve_df.iloc[:, i] - analysis_curve_df.iloc[:, i+k]
    return pd.DataFrame(spreads_data)

@st.cache_data
def calculate_k_step_butterflies(analysis_curve_df, k):
    if analysis_curve_df is None or analysis_curve_df.empty or analysis_curve_df.shape[1] < 2*k+1:
        return pd.DataFrame()
    num_contracts = analysis_curve_df.shape[1]
    flies = {}
    cols = analysis_curve_df.columns.tolist()
    for i in range(num_contracts - 2*k):
        c1 = cols[i]
        c2 = cols[i+k]
        c3 = cols[i+2*k]
        label = f"{c1}-2x{c2}+{c3}"
        flies[label] = analysis_curve_df.iloc[:, i] - 2*analysis_curve_df.iloc[:, i+k] + analysis_curve_df.iloc[:, i+2*k]
    return pd.DataFrame(flies)

@st.cache_data
def calculate_k_step_double_butterflies(analysis_curve_df, k):
    # C_i - 3C_{i+k} + 3C_{i+2k} - C_{i+3k}
    if analysis_curve_df is None or analysis_curve_df.empty or analysis_curve_df.shape[1] < 3*k+1:
        return pd.DataFrame()
    num_contracts = analysis_curve_df.shape[1]
    dbflies = {}
    cols = analysis_curve_df.columns.tolist()
    for i in range(num_contracts - 3*k):
        c1 = cols[i]
        c2 = cols[i+k]
        c3 = cols[i+2*k]
        c4 = cols[i+3*k]
        label = f"{c1}-3x{c2}+3x{c3}-{c4}"
        dbflies[label] = (
            analysis_curve_df.iloc[:, i]
            - 3*analysis_curve_df.iloc[:, i+k]
            + 3*analysis_curve_df.iloc[:, i+2*k]
            - analysis_curve_df.iloc[:, i+3*k]
        )
    return pd.DataFrame(dbflies)

# -----------------------------
# PCA functions
# -----------------------------
def perform_pca(data_df):
    data_df_clean = data_df.dropna()
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None, None, None, None
    data_mean = data_df_clean.mean()
    data_std = data_df_clean.std()
    data_scaled = (data_df_clean - data_mean) / data_std
    n_components = min(data_scaled.shape)
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=data_df_clean.columns)
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    scores = pd.DataFrame(pca.transform(data_scaled), index=data_df_clean.index, columns=[f'PC{i+1}' for i in range(n_components)])
    return loadings, explained_variance_ratio, eigenvalues, scores, data_df_clean

def perform_pca_on_prices(price_df):
    data_df_clean = price_df.dropna()
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None
    data_centered = data_df_clean - data_df_clean.mean()
    n_components = min(data_centered.shape)
    pca = PCA(n_components=n_components)
    pca.fit(data_centered)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=data_df_clean.columns)
    explained_variance = pca.explained_variance_ratio_
    return loadings, explained_variance

# -----------------------------
# Reconstruction helpers
# -----------------------------
def _reconstruct_derivative(original_df, reconstructed_prices, derivative_type='spread'):
    if original_df is None or original_df.empty:
        return pd.DataFrame()
    valid_indices = reconstructed_prices.index.intersection(original_df.index)
    original_df_aligned = original_df.loc[valid_indices]
    reconstructed_prices_aligned = reconstructed_prices.loc[valid_indices]
    reconstructed_data = {}
    for label in original_df_aligned.columns:
        try:
            if derivative_type == 'spread':
                core_label = label.split(': ')[1] if ':' in label else label
                c1, c_long = core_label.split('-')
                reconstructed_data[label + ' (PCA)'] = reconstructed_prices_aligned[c1 + ' (PCA)'] - reconstructed_prices_aligned[c_long + ' (PCA)']
            elif derivative_type == 'fly':
                core_label = label.split(': ')[1] if ':' in label else label
                parts = core_label.split('-', 1)
                c1 = parts[0]
                sub_parts = parts[1].split('+')
                c2_label = sub_parts[0].split('x')[1]
                c3_label = sub_parts[1]
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)']
                    - 2 * reconstructed_prices_aligned[c2_label + ' (PCA)']
                    + reconstructed_prices_aligned[c3_label + ' (PCA)']
                )
            elif derivative_type == 'dbfly':
                core_label = label.split(': ')[1] if ':' in label else label
                parts = core_label.split('-', 1)
                c1 = parts[0]
                sub_parts_1 = parts[1].split('+')
                c2_label = sub_parts_1[0].split('x')[1]
                sub_parts_2 = sub_parts_1[1].split('-')
                c3_label = sub_parts_2[0].split('x')[1]
                c4_label = sub_parts_2[1]
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)']
                    - 3 * reconstructed_prices_aligned[c2_label + ' (PCA)']
                    + 3 * reconstructed_prices_aligned[c3_label + ' (PCA)']
                    - reconstructed_prices_aligned[c4_label + ' (PCA)']
                )
        except Exception:
            continue
    reconstructed_df = pd.DataFrame(reconstructed_data, index=reconstructed_prices_aligned.index)
    original_rename = {col: col + ' (Original)' for col in original_df_aligned.columns}
    original_df_renamed = original_df_aligned.rename(columns=original_rename)
    return pd.merge(original_df_renamed, reconstructed_df, left_index=True, right_index=True)

def reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_3M_df, spreads_3M_df, spreads_6M_df,
                                       butterflies_3M_df, butterflies_6M_df, spreads_12M_df, butterflies_12M_df,
                                       double_butterflies_3M_df, double_butterflies_6M_df, double_butterflies_12M_df):
    # Align index
    analysis_curve_df_aligned = analysis_curve_df.loc[reconstructed_spreads_3M_df.index]
    nearest_contract_original = analysis_curve_df_aligned.iloc[:, 0]
    nearest_contract_label = analysis_curve_df_aligned.columns[0]
    reconstructed_prices_df = pd.DataFrame(index=analysis_curve_df_aligned.index)
    reconstructed_prices_df[nearest_contract_label + ' (PCA)'] = nearest_contract_original
    # Reconstruct subsequent contracts from 3M spreads (assumes spreads_3M_df columns label matches)
    for i in range(1, len(analysis_curve_df_aligned.columns)):
        prev = analysis_curve_df_aligned.columns[i-1]
        curr = analysis_curve_df_aligned.columns[i]
        spread_label_no_prefix = f"{prev}-{curr}"
        if spread_label_no_prefix in reconstructed_spreads_3M_df.columns:
            reconstructed_prices_df[curr + ' (PCA)'] = (
                reconstructed_prices_df[prev + ' (PCA)'] - reconstructed_spreads_3M_df[spread_label_no_prefix]
            )
        else:
            reconstructed_prices_df[curr + ' (PCA)'] = reconstructed_prices_df[prev + ' (PCA)']
    original_price_rename = {col: col + ' (Original)' for col in analysis_curve_df_aligned.columns}
    original_prices_df = analysis_curve_df_aligned.rename(columns=original_price_rename)
    historical_outrights = pd.merge(original_prices_df, reconstructed_prices_df, left_index=True, right_index=True)

    # Prepare derivative DataFrames for reconstruction (prefixes)
    spreads_3M_pref = spreads_3M_df.rename(columns=lambda x: f"3M Spread: {x}")
    butterflies_3M_pref = butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}")
    double_butterflies_3M_pref = double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}")
    spreads_6M_pref = spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}")
    butterflies_6M_pref = butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}")
    double_butterflies_6M_pref = double_butterflies_6M_df.rename(columns=lambda x: f"6M Double Fly: {x}")
    spreads_12M_pref = spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}")
    butterflies_12M_pref = butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}")
    double_butterflies_12M_pref = double_butterflies_12M_df.rename(columns=lambda x: f"12M Double Fly: {x}")

    historical_spreads_3M = _reconstruct_derivative(spreads_3M_pref, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_3M = _reconstruct_derivative(butterflies_3M_pref, reconstructed_prices_df, derivative_type='fly')
    historical_double_butterflies_3M = _reconstruct_derivative(double_butterflies_3M_pref, reconstructed_prices_df, derivative_type='dbfly')

    historical_spreads_6M = _reconstruct_derivative(spreads_6M_pref, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_6M = _reconstruct_derivative(butterflies_6M_pref, reconstructed_prices_df, derivative_type='fly')
    historical_double_butterflies_6M = _reconstruct_derivative(double_butterflies_6M_pref, reconstructed_prices_df, derivative_type='dbfly')

    historical_spreads_12M = _reconstruct_derivative(spreads_12M_pref, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_12M = _reconstruct_derivative(butterflies_12M_pref, reconstructed_prices_df, derivative_type='fly')
    historical_double_butterflies_12M = _reconstruct_derivative(double_butterflies_12M_pref, reconstructed_prices_df, derivative_type='dbfly')

    return (
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
        spreads_3M_df
    )

# -----------------------------
# Hedging & covariance functions
# -----------------------------
def calculate_reconstructed_covariance(loadings_df, eigenvalues, spread_std_dev, pc_count):
    L_p = loadings_df.iloc[:, :pc_count].values
    lambda_p = eigenvalues[:pc_count]
    Sigma_scaled = L_p @ np.diag(lambda_p) @ L_p.T
    Sigma = Sigma_scaled * np.outer(spread_std_dev.values, spread_std_dev.values)
    Sigma_df = pd.DataFrame(Sigma, index=loadings_df.index, columns=loadings_df.index)
    return Sigma_df

def calculate_best_and_worst_hedge_3M(trade_label, loadings_df, eigenvalues, pc_count, spreads_3M_df_clean):
    if trade_label not in loadings_df.index:
        return None, None, None
    spread_std = spreads_3M_df_clean.std()
    Sigma_reconstructed = calculate_reconstructed_covariance(loadings_df, eigenvalues, spread_std, pc_count)
    trade_spread = trade_label
    results = []
    potential_hedges = [col for col in Sigma_reconstructed.columns if col != trade_spread]
    for hedge_spread in potential_hedges:
        Var_Trade = Sigma_reconstructed.loc[trade_spread, trade_spread]
        Var_Hedge = Sigma_reconstructed.loc[hedge_spread, hedge_spread]
        Cov_TH = Sigma_reconstructed.loc[trade_spread, hedge_spread]
        k_star = 0 if Var_Hedge == 0 else Cov_TH / Var_Hedge
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance)
        Residual_Volatility_BPS = np.sqrt(Residual_Variance) * 10000
        results.append({
            'Hedge Spread': hedge_spread,
            'Hedge Ratio (k*)': k_star,
            'Residual Volatility (BPS)': Residual_Volatility_BPS
        })
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return None, None, None
    best = results_df.sort_values(by='Residual Volatility (BPS)', ascending=True).iloc[0]
    worst = results_df.sort_values(by='Residual Volatility (BPS)', ascending=False).iloc[0]
    return best, worst, results_df

def calculate_derivatives_covariance_generalized(all_derivatives_df, scores_df, eigenvalues, pc_count):
    aligned_index = all_derivatives_df.index.intersection(scores_df.index)
    derivatives_aligned = all_derivatives_df.loc[aligned_index].dropna(axis=1)
    scores_aligned = scores_df.loc[aligned_index]
    if derivatives_aligned.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    derivatives_mean = derivatives_aligned.mean()
    derivatives_std = derivatives_aligned.std()
    derivatives_scaled = (derivatives_aligned - derivatives_mean) / derivatives_std
    loadings_data = {}
    X = scores_aligned.iloc[:, :pc_count].values
    for col in derivatives_scaled.columns:
        y = derivatives_scaled[col].values
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        loadings_data[col] = reg.coef_
    loadings_df = pd.DataFrame(loadings_data, index=[f'PC{i+1}' for i in range(pc_count)]).T
    L_D = loadings_df.values
    lambda_p = eigenvalues[:pc_count]
    Sigma_Std = L_D @ np.diag(lambda_p) @ L_D.T
    Sigma_Raw = Sigma_Std * np.outer(derivatives_std.values, derivatives_std.values)
    Sigma_Raw_df = pd.DataFrame(Sigma_Raw, index=derivatives_aligned.columns, columns=derivatives_aligned.columns)
    return Sigma_Raw_df, derivatives_aligned, loadings_df

def calculate_best_and_worst_hedge_generalized(trade_label, Sigma_Raw_df):
    if trade_label not in Sigma_Raw_df.index:
        return None, None, None
    results = []
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label]
    for hedge_instrument in potential_hedges:
        Var_Trade = Sigma_Raw_df.loc[trade_label, trade_label]
        Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument]
        Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument]
        k_star = 0 if Var_Hedge <= 1e-9 else Cov_TH / Var_Hedge
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance)
        Residual_Volatility_BPS = np.sqrt(Residual_Variance) * 10000
        results.append({
            'Hedge Instrument': hedge_instrument,
            'Hedge Ratio (k*)': k_star,
            'Residual Volatility (BPS)': Residual_Volatility_BPS
        })
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return None, None, None
    best = results_df.sort_values(by='Residual Volatility (BPS)', ascending=True).iloc[0]
    worst = results_df.sort_values(by='Residual Volatility (BPS)', ascending=False).iloc[0]
    return best, worst, results_df

# -----------------------------
# Factor sensitivities & factor hedging
# -----------------------------
def calculate_factor_sensitivities(loadings_df_gen, pc_count):
    if loadings_df_gen is None or loadings_df_gen.empty:
        return pd.DataFrame()
    pc_map = {'PC1': 'Level (Whole Curve Shift)', 'PC2': 'Slope (Steepening/Flattening)', 'PC3': 'Curvature (Fly Risk)'}
    available_pcs = list(pc_map.keys())
    available_pcs = [pc for pc in available_pcs if pc in loadings_df_gen.columns]
    factor_sensitivities = loadings_df_gen.filter(items=available_pcs, axis=1).copy()
    factor_sensitivities.columns = [pc_map[col] for col in available_pcs]
    return factor_sensitivities

def calculate_all_factor_hedges(trade_label, factor_name, factor_sensitivities_df, Sigma_Raw_df):
    if trade_label not in factor_sensitivities_df.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not found in sensitivities."
    if factor_name not in factor_sensitivities_df.columns:
        return pd.DataFrame(), f"Factor '{factor_name}' not found."
    if trade_label not in Sigma_Raw_df.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not in covariance matrix."
    results = []
    Trade_Exposure = factor_sensitivities_df.loc[trade_label, factor_name]
    Var_Trade = Sigma_Raw_df.loc[trade_label, trade_label]
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label]
    for hedge_instrument in potential_hedges:
        try:
            Hedge_Exposure = factor_sensitivities_df.loc[hedge_instrument, factor_name]
            Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument]
            Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument]
            if abs(Hedge_Exposure) < 1e-9:
                k_factor = 0.0
                Residual_Volatility_BPS = np.nan
            else:
                k_factor = Trade_Exposure / Hedge_Exposure
                Residual_Variance = Var_Trade + (k_factor**2 * Var_Hedge) - (2 * k_factor * Cov_TH)
                Residual_Variance = max(0, Residual_Variance)
                Residual_Volatility_BPS = np.sqrt(Residual_Variance) * 10000
            results.append({
                'Hedge Instrument': hedge_instrument,
                'Trade Sensitivity': Trade_Exposure,
                'Hedge Sensitivity': Hedge_Exposure,
                'Factor Hedge Ratio (k_factor)': k_factor,
                'Residual Volatility (BPS)': Residual_Volatility_BPS
            })
        except Exception:
            continue
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame(), "No valid hedge candidates found."
    results_df = results_df.sort_values(by='Residual Volatility (BPS)', ascending=True, na_position='last')
    return results_df, None

# -----------------------------
# Excess returns, ranking, classification
# -----------------------------
def calculate_hedge_excess_return(hedge_label, historical_df):
    if historical_df is None or historical_df.empty:
        return None, None, "Historical data unavailable."
    latest_date = historical_df.index.max()
    try:
        market_price = historical_df[f"{hedge_label} (Original)"].loc[latest_date]
        fair_price = historical_df[f"{hedge_label} (PCA)"].loc[latest_date]
    except KeyError:
        return None, None, f"Hedge label '{hedge_label}' not found."
    excess_bps = (market_price - fair_price) * 10000
    excess_usd = excess_bps * BP_TO_USD
    return excess_bps, excess_usd, None

def classify_instrument_type(label):
    label_u = label.upper()
    if "DOUBLE FLY" in label_u or "DBF" in label_u or "3X" in label_u:
        return "Double Fly"
    elif "FLY" in label_u or re.search(r"-2X", label_u):
        return "Fly"
    elif "SPREAD" in label_u or "-" in label_u and "SPREAD" not in label_u:
        # Many spread labels are like "Z25-H26" so treat them as Spread if not marked Fly
        if "FLY" not in label_u:
            return "Spread"
    return "Other"

def calculate_reward_to_risk(excess_bps, residual_bps):
    if residual_bps is None or pd.isna(residual_bps) or abs(residual_bps) <= 1e-9:
        return np.nan
    return excess_bps / residual_bps

def rank_all_hedge_excess_returns_with_risk(historical_df, all_results_df_full_gen=None):
    if historical_df is None or historical_df.empty:
        return pd.DataFrame(), "No historical data available."
    latest_date = historical_df.index.max()
    all_cols = historical_df.columns
    pairs = [col.replace(" (Original)", "") for col in all_cols if col.endswith("(Original)") and col.replace(" (Original)", " (PCA)") in all_cols]
    results = []
    for label in pairs:
        mkt = historical_df[f"{label} (Original)"].loc[latest_date]
        fair = historical_df[f"{label} (PCA)"].loc[latest_date]
        exc_bps = (mkt - fair) * 10000
        exc_usd = exc_bps * BP_TO_USD
        results.append({
            "Instrument": label,
            "Market Price": mkt,
            "PCA Fair Price": fair,
            "Excess Return (bps)": exc_bps,
            "Excess Return ($)": exc_usd,
            "Instrument Type": classify_instrument_type(label)
        })
    df = pd.DataFrame(results)
    if all_results_df_full_gen is not None and not all_results_df_full_gen.empty:
        vol_df = all_results_df_full_gen[["Hedge Instrument", "Residual Volatility (BPS)"]].rename(columns={"Hedge Instrument": "Instrument"})
        df = df.merge(vol_df, on="Instrument", how="left")
        df["Reward-to-Risk"] = df.apply(lambda x: calculate_reward_to_risk(x["Excess Return (bps)"], x.get("Residual Volatility (BPS)", np.nan)), axis=1)
    else:
        df["Residual Volatility (BPS)"] = np.nan
        df["Reward-to-Risk"] = np.nan
    # default sorting: Reward-to-Risk desc (NaNs last)
    df = df.sort_values(by="Reward-to-Risk", ascending=False, na_position='last').reset_index(drop=True)
    return df, None

# -----------------------------
# Streamlit UI: Uploads & date selection (top-level)
# -----------------------------
st.title("SOFR Futures PCA Analyzer (Sections 1–9)")

st.sidebar.header("1. Data Uploads")
price_file = st.sidebar.file_uploader("Upload Historical Price Data (CSV)", type=['csv'], key='price_upload')
expiry_file = st.sidebar.file_uploader("Upload Contract Expiry Dates (CSV)", type=['csv'], key='expiry_upload')

price_df = load_data(price_file)
expiry_df = load_data(expiry_file)

if price_df is None or expiry_df is None:
    st.info("Please upload both Price and Expiry CSV files to proceed.")
    st.stop()

st.sidebar.header("2. Historical Date Range")
min_date = price_df.index.min().date()
max_date = price_df.index.max().date()
start_date, end_date = st.sidebar.date_input("Select Historical Data Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
price_df_filtered = price_df[(price_df.index.date >= start_date) & (price_df.index.date <= end_date)]

st.sidebar.header("3. Curve Analysis Date")
default_analysis_date = end_date if end_date >= min_date else min_date
analysis_date = st.sidebar.date_input("Select Single Date for Curve Snapshot", value=default_analysis_date, min_value=min_date, max_value=max_date)
analysis_dt = datetime.combine(analysis_date, datetime.min.time())

if price_df_filtered.empty:
    st.warning("No price data in the selected date range.")
    st.stop()

# -----------------------------
# Core processing
# -----------------------------
future_expiries_df = get_analysis_contracts(expiry_df, analysis_dt)
if future_expiries_df.empty:
    st.warning("No contracts expire on/after chosen analysis date.")
    st.stop()

analysis_curve_df, contract_labels = transform_to_analysis_curve(price_df_filtered, future_expiries_df)
if analysis_curve_df.empty:
    st.warning("Could not map any contracts from expiry file to price file. Check contract names.")
    st.stop()

# -----------------------------
# 1. Data Derivatives Check
# -----------------------------
st.header("1. Data Derivatives Check (Contracts relevant to selected Analysis Date)")
spreads_3M_df_raw = calculate_k_step_spreads(analysis_curve_df, 1)
butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, 1)
double_butterflies_3M_df = calculate_k_step_double_butterflies(analysis_curve_df, 1)

spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, 2)
butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, 2)
double_butterflies_6M_df = calculate_k_step_double_butterflies(analysis_curve_df, 2)

spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, 4)
butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, 4)
double_butterflies_12M_df = calculate_k_step_double_butterflies(analysis_curve_df, 4)

st.markdown("##### 3-Month Outright Spreads (k=1)")
st.dataframe(spreads_3M_df_raw.head(5))
st.markdown("##### 3-Month Double Butterflies (k=1)")
st.dataframe(double_butterflies_3M_df.head(5))

if spreads_3M_df_raw.empty:
    st.warning("3M Spreads could not be calculated. Need at least two contracts.")
    st.stop()

# -----------------------------
# 2-4 PCA calibration, loadings, scores
# -----------------------------
loadings_spread, explained_variance_ratio, eigenvalues, scores, spreads_3M_df_clean = perform_pca(spreads_3M_df_raw)
loadings_outright_direct, explained_variance_outright_direct = perform_pca_on_prices(analysis_curve_df)

if loadings_spread is None or loadings_outright_direct is None:
    st.error("PCA could not be performed (not enough data).")
    st.stop()

# Explained variance table and selection
st.header("2. Explained Variance")
variance_df = pd.DataFrame({'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))], 'Explained Variance (%)': explained_variance_ratio * 100})
variance_df['Cumulative Variance (%)'] = variance_df['Explained Variance (%)'].cumsum()
col_var, col_pca_select = st.columns([1, 1])
with col_var:
    st.dataframe(variance_df, use_container_width=True)
default_pc_count = min(3, len(explained_variance_ratio))
with col_pca_select:
    st.subheader("Fair Curve & Hedging Setup")
    pc_count = st.slider("Select number of Principal Components (PCs) for Fair Curve & Hedging:", min_value=1, max_value=len(explained_variance_ratio), value=default_pc_count, key='pc_slider')
    total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
    st.info(f"The selected {pc_count} PCs explain {total_explained:.2f}% of spread variance.")

# PCA loadings heatmap (spreads)
st.header("3. PC Loadings")
st.subheader("3.1 PC Loadings Heatmap (PC vs. 3M Spreads)")
plt.style.use('default')
fig_spread_loading, ax_spread_loading = plt.subplots(figsize=(12, 6))
loadings_spread_plot = loadings_spread.iloc[:, :pc_count]
sns.heatmap(loadings_spread_plot, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Loading Weight'}, ax=ax_spread_loading)
ax_spread_loading.set_title(f'Component Loadings for First {pc_count} PCs (on Spreads)')
ax_spread_loading.set_xlabel('Principal Component'); ax_spread_loading.set_ylabel('Spread Contract')
st.pyplot(fig_spread_loading)

st.subheader("3.2 PC Loadings Heatmap (PC vs. Outright Contracts - Absolute Sensitivity)")
pc1_outright_variance = explained_variance_outright_direct[0] * 100 if explained_variance_outright_direct is not None else 0
st.markdown(f"PC1 (absolute price) explains {pc1_outright_variance:.2f}% of price variance.")
fig_outright_loading, ax_outright_loading = plt.subplots(figsize=(12, 6))
loadings_outright_plot = loadings_outright_direct.iloc[:, :pc_count]
max_abs = loadings_outright_plot.abs().max().max()
sns.heatmap(loadings_outright_plot, annot=True, cmap='coolwarm', fmt=".4f", linewidths=0.5, linecolor='gray', vmin=-max_abs, vmax=max_abs, cbar_kws={'label': 'Absolute Price Sensitivity (Eigenvector Weight)'}, ax=ax_outright_loading)
ax_outright_loading.set_title(f'Component Loadings for First {pc_count} PCs (Unstandardized Outright Prices)')
ax_outright_loading.set_xlabel('Principal Component'); ax_outright_loading.set_ylabel('Outright Contract')
st.pyplot(fig_outright_loading)

# Scores time series
def plot_pc_scores(scores_df, explained_variance_ratio):
    pc_labels = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
    num_pcs = min(3, scores_df.shape[1])
    if num_pcs == 0:
        return None
    fig, axes = plt.subplots(nrows=num_pcs, ncols=1, figsize=(15, 4 * num_pcs), sharex=True)
    if num_pcs == 1: axes = [axes]
    plt.suptitle("Time Series of Principal Component Scores (Risk Factors)", fontsize=16, y=1.02)
    for i in range(num_pcs):
        ax = axes[i]
        variance_pct = explained_variance_ratio[i] * 100
        ax.plot(scores_df.index, scores_df.iloc[:, i], label=f'{pc_labels[i]} ({variance_pct:.2f}% Var.)', linewidth=1.5)
        ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
        ax.set_title(f'{pc_labels[i]} Factor Score (Explaining {variance_pct:.2f}% of Spread Variance)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylabel('Score Value')
        ax.legend(loc='upper left')
    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig

st.header("4. PC Factor Scores Time Series")
fig_scores = plot_pc_scores(scores, explained_variance_ratio)
if fig_scores:
    st.pyplot(fig_scores)

# -----------------------------
# Reconstruct using selected PCs
# -----------------------------
data_mean = spreads_3M_df_clean.mean()
data_std = spreads_3M_df_clean.std()
scores_used = scores.values[:, :pc_count]
loadings_used = loadings_spread.values[:, :pc_count]
reconstructed_scaled = scores_used @ loadings_used.T
reconstructed_spreads_3M = pd.DataFrame(reconstructed_scaled * data_std.values + data_mean.values, index=spreads_3M_df_clean.index, columns=spreads_3M_df_clean.columns)

historical_outrights_df, historical_spreads_3M_df, historical_butterflies_3M_df, historical_spreads_6M_df, historical_butterflies_6M_df, historical_spreads_12M_df, historical_butterflies_12M_df, historical_double_butterflies_3M_df, historical_double_butterflies_6M_df, historical_double_butterflies_12M_df, spreads_3M_df_no_prefix = reconstruct_prices_and_derivatives(
    analysis_curve_df, reconstructed_spreads_3M, spreads_3M_df_raw,
    spreads_6M_df, butterflies_3M_df, butterflies_6M_df,
    spreads_12M_df, butterflies_12M_df,
    double_butterflies_3M_df, double_butterflies_6M_df, double_butterflies_12M_df
)

# -----------------------------
# 5. Curve Snapshot Analysis (Outrights + derivatives)
# -----------------------------
st.header("5. Curve Snapshot Analysis: " + analysis_date.strftime('%Y-%m-%d'))

# Outright snapshot
try:
    curve_snapshot_original = historical_outrights_df.filter(regex='\(Original\)$').loc[[analysis_dt]].T
    curve_snapshot_pca = historical_outrights_df.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
    curve_snapshot_original.columns = ['Original']; curve_snapshot_original.index = curve_snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)
    curve_snapshot_pca.columns = ['PCA Fair']; curve_snapshot_pca.index = curve_snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)
    curve_comparison = pd.concat([curve_snapshot_original, curve_snapshot_pca], axis=1).dropna()
    if not curve_comparison.empty:
        fig_curve, ax_curve = plt.subplots(figsize=(15, 7))
        ax_curve.plot(curve_comparison.index, curve_comparison['Original'], label='Original Market Curve', marker='o', linestyle='-')
        ax_curve.plot(curve_comparison.index, curve_comparison['PCA Fair'], label=f'PCA Fair Curve ({pc_count} PCs)', marker='x', linestyle='--')
        mispricing = curve_comparison['Original'] - curve_comparison['PCA Fair']
        max_abs_mispricing = mispricing.abs().max()
        if max_abs_mispricing > 0:
            mispricing_contract = mispricing.abs().idxmax()
            mispricing_value = mispricing.loc[mispricing_contract] * 10000
            ax_curve.annotate(f"Mispricing: {mispricing_value:.2f} BPS", (mispricing_contract, curve_comparison.loc[mispricing_contract]['Original']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
        ax_curve.set_title('Market Price Curve vs. PCA Fair Price Curve'); ax_curve.set_xlabel('Contract Maturity'); ax_curve.set_ylabel('Price (100 - Rate)')
        ax_curve.legend(loc='upper right'); ax_curve.grid(True, linestyle=':', alpha=0.6); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        st.pyplot(fig_curve)
        detailed_comparison = curve_comparison.copy()
        detailed_comparison.index.name = 'Contract'
        detailed_comparison['Original Rate (%)'] = 100.0 - detailed_comparison['Original']
        detailed_comparison['PCA Fair Rate (%)'] = 100.0 - detailed_comparison['PCA Fair']
        detailed_comparison['Mispricing (BPS)'] = (detailed_comparison['Original'] - detailed_comparison['PCA Fair']) * 10000
        detailed_comparison = detailed_comparison.rename(columns={'Original': 'Original Price', 'PCA Fair': 'PCA Fair Price'})
        detailed_comparison = detailed_comparison[['Original Price', 'Original Rate (%)', 'PCA Fair Price', 'PCA Fair Rate (%)', 'Mispricing (BPS)']]
        st.dataframe(detailed_comparison.style.format({'Original Price': "{:.4f}", 'PCA Fair Price': "{:.4f}", 'Original Rate (%)': "{:.4f}", 'PCA Fair Rate (%)': "{:.4f}", 'Mispricing (BPS)': "{:.2f}"}), use_container_width=True)
    else:
        st.warning("No complete Outright Price data available for the selected analysis date.")
except KeyError:
    st.error("Selected analysis date not present in filtered price data for Outrights.")

# snapshot helper for other derivatives
def plot_snapshot(historical_df, derivative_type, analysis_dt, pc_count):
    if historical_df is None or historical_df.empty:
        st.info(f"Not enough contracts to calculate and plot {derivative_type} snapshot.")
        return
    try:
        snapshot_original = historical_df.filter(regex='\(Original\)$').loc[[analysis_dt]].T
        snapshot_pca = historical_df.filter(regex='\(PCA\)$').loc[[analysis_dt]].T
        snapshot_original.columns = ['Original']; snapshot_original.index = snapshot_original.index.str.replace(r'\s\(Original\)$', '', regex=True)
        snapshot_pca.columns = ['PCA Fair']; snapshot_pca.index = snapshot_pca.index.str.replace(r'\s\(PCA\)$', '', regex=True)
        comparison = pd.concat([snapshot_original, snapshot_pca], axis=1).dropna()
        if comparison.empty:
            st.warning(f"No complete {derivative_type} data available for {analysis_date.strftime('%Y-%m-%d')}.")
            return
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(comparison.index, comparison['Original'], label=f'Original Market {derivative_type}', marker='o', linestyle='-')
        ax.plot(comparison.index, comparison['PCA Fair'], label=f'PCA Fair {derivative_type} ({pc_count} PCs)', marker='x', linestyle='--')
        mispricing = comparison['Original'] - comparison['PCA Fair']
        max_abs_mispricing = mispricing.abs().max()
        if max_abs_mispricing > 0:
            mispricing_contract = mispricing.abs().idxmax()
            mispricing_value = mispricing.loc[mispricing_contract] * 10000
            ax.annotate(f"Mispricing: {mispricing_value:.2f} BPS", (mispricing_contract, comparison.loc[mispricing_contract]['Original']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
        ax.set_title(f'Market {derivative_type} vs. PCA Fair {derivative_type}'); ax.set_xlabel(f'{derivative_type} Contract'); ax.set_ylabel(f'{derivative_type} Value (Price Difference)'); ax.legend(loc='upper right'); ax.grid(True, linestyle=':', alpha=0.6); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        st.pyplot(fig)
        detailed = comparison.copy()
        detailed.index.name = f'{derivative_type} Contract'
        detailed['Mispricing (BPS)'] = (detailed['Original'] - detailed['PCA Fair']) * 10000
        st.markdown(f"###### {derivative_type} Mispricing")
        st.dataframe(detailed.style.format({'Original': "{:.4f}", 'PCA Fair': "{:.4f}", 'Mispricing (BPS)': "{:.2f}"}), use_container_width=True)
    except KeyError:
        st.error(f"The selected analysis date {analysis_date.strftime('%Y-%m-%d')} is not present in the data for {derivative_type}.")

# 3M snapshots
st.subheader("5.2 3M Spread Snapshot")
plot_snapshot(historical_spreads_3M_df, "3M Spread", analysis_dt, pc_count)
st.subheader("5.3 3M Butterfly (Fly) Snapshot")
plot_snapshot(historical_butterflies_3M_df, "3M Butterfly", analysis_dt, pc_count)
st.subheader("5.4 3M Double Butterfly (DBF) Snapshot")
plot_snapshot(historical_double_butterflies_3M_df, "3M Double Butterfly", analysis_dt, pc_count)

# 6M snapshots
st.subheader("5.5 6M Spread Snapshot")
plot_snapshot(historical_spreads_6M_df, "6M Spread", analysis_dt, pc_count)
st.subheader("5.6 6M Butterfly (Fly) Snapshot")
plot_snapshot(historical_butterflies_6M_df, "6M Butterfly", analysis_dt, pc_count)
st.subheader("5.7 6M Double Butterfly (DBF) Snapshot")
plot_snapshot(historical_double_butterflies_6M_df, "6M Double Butterfly", analysis_dt, pc_count)

# 12M snapshots
st.subheader("5.8 12M Spread Snapshot")
plot_snapshot(historical_spreads_12M_df, "12M Spread", analysis_dt, pc_count)
st.subheader("5.9 12M Butterfly (Fly) Snapshot")
plot_snapshot(historical_butterflies_12M_df, "12M Butterfly", analysis_dt, pc_count)
st.subheader("5.10 12M Double Butterfly (DBF) Snapshot")
plot_snapshot(historical_double_butterflies_12M_df, "12M Double Butterfly", analysis_dt, pc_count)

# -----------------------------
# 6. PCA-Based Hedging Strategy (3M spreads only)
# -----------------------------
st.header("6. PCA-Based Hedging Strategy (3M Spreads ONLY - Original Section)")
if spreads_3M_df_clean.shape[1] < 2:
    st.warning("Need at least two 3M spreads to analyze hedging.")
else:
    default_trade = spreads_3M_df_clean.columns[0]
    trade_selection_3m = st.selectbox("Select Trade Spread (Long 1 unit):", options=spreads_3M_df_clean.columns.tolist(), index=0, key='trade_spread_select_3m')
    best_hedge_data_3m, worst_hedge_data_3m, all_results_df_full_3m = calculate_best_and_worst_hedge_3M(trade_selection_3m, loadings_spread, eigenvalues, pc_count, spreads_3M_df_clean)
    if best_hedge_data_3m is not None:
        col_best, col_worst = st.columns(2)
        with col_best:
            st.success(f"Best Hedge for Long 1x {trade_selection_3m}")
            st.markdown(f"- Hedge Spread: **{best_hedge_data_3m['Hedge Spread']}**\n- Hedge Ratio: **{best_hedge_data_3m['Hedge Ratio (k*)']:.4f}**\n- Residual Volatility: **{best_hedge_data_3m['Residual Volatility (BPS)']:.2f} BPS**")
        with col_worst:
            st.error(f"Worst Hedge for Long 1x {trade_selection_3m}")
            st.markdown(f"- Hedge Spread: **{worst_hedge_data_3m['Hedge Spread']}**\n- Hedge Ratio: **{worst_hedge_data_3m['Hedge Ratio (k*)']:.4f}**\n- Residual Volatility: **{worst_hedge_data_3m['Residual Volatility (BPS)']:.2f} BPS**")
        st.markdown("###### Detailed Hedging Results (All 3M Spreads as Hedge Candidates)")
        all_results_df_full_3m = all_results_df_full_3m.sort_values(by='Residual Volatility (BPS)', ascending=True)
        st.dataframe(all_results_df_full_3m.style.format({'Hedge Ratio (k*)': "{:.4f}", 'Residual Volatility (BPS)': "{:.2f}"}), use_container_width=True)
    else:
        st.warning("3M Hedging calculation failed.")

# -----------------------------
# 7. Generalized Hedging (All derivatives)
# -----------------------------
st.header("7. PCA-Based Generalized Hedging Strategy (Minimum Variance)")
# Build all_derivatives_df already done earlier as all_derivatives_df
Sigma_Raw_df, all_derivatives_aligned, loadings_df_gen = calculate_derivatives_covariance_generalized(all_derivatives_df, scores, eigenvalues, pc_count)
if Sigma_Raw_df.empty or Sigma_Raw_df.shape[0] < 2:
    st.warning("Not enough data to calculate generalized hedging correlations.")
else:
    trade_selection_gen = st.selectbox("Select Trade Instrument (Long 1 unit):", options=Sigma_Raw_df.columns.tolist(), index=0, key='trade_instrument_select_gen')
    best_hedge_data_gen, worst_hedge_data_gen, all_results_df_full_gen = calculate_best_and_worst_hedge_generalized(trade_selection_gen, Sigma_Raw_df)
    if best_hedge_data_gen is not None:
        col_best_gen, col_worst_gen = st.columns(2)
        with col_best_gen:
            st.success(f"Best Hedge for Long 1x {trade_selection_gen}")
            st.markdown(f"- Hedge Instrument: **{best_hedge_data_gen['Hedge Instrument']}**\n- Hedge Ratio: **{best_hedge_data_gen['Hedge Ratio (k*)']:.4f}**\n- Residual Volatility: **{best_hedge_data_gen['Residual Volatility (BPS)']:.2f} BPS**")
        with col_worst_gen:
            st.error(f"Worst Hedge for Long 1x {trade_selection_gen}")
            st.markdown(f"- Hedge Instrument: **{worst_hedge_data_gen['Hedge Instrument']}**\n- Hedge Ratio: **{worst_hedge_data_gen['Hedge Ratio (k*)']:.4f}**\n- Residual Volatility: **{worst_hedge_data_gen['Residual Volatility (BPS)']:.2f} BPS**")
        st.markdown("###### Detailed Hedging Results (All Derivatives as Hedge Candidates - Sorted by Minimum Variance)")
        all_results_df_full_gen = all_results_df_full_gen.sort_values(by='Residual Volatility (BPS)', ascending=True)
        st.dataframe(all_results_df_full_gen.style.format({'Hedge Ratio (k*)': "{:.4f}", 'Residual Volatility (BPS)': "{:.2f}"}), use_container_width=True)
    else:
        st.warning("Generalized Minimum Variance Hedging calculation failed for selected trade.")

# -----------------------------
# 8. Factor Hedging (Sensitivity Hedging)
# -----------------------------
st.header("8. PCA-Based Factor Hedging Strategy (Sensitivity Hedging)")
factor_sensitivities_df = calculate_factor_sensitivities(loadings_df_gen, pc_count)
if factor_sensitivities_df.empty:
    st.error("Factor sensitivity data unavailable. Ensure generalized loadings calculated.")
else:
    col_trade_sel, col_factor_sel = st.columns(2)
    instrument_options = factor_sensitivities_df.index.tolist()
    factor_options = factor_sensitivities_df.columns.tolist()
    with col_trade_sel:
        trade_selection_factor = st.selectbox("Select Trade Instrument (Long 1 unit):", options=instrument_options, index=0, key='trade_instrument_factor_table')
    with col_factor_sel:
        factor_selection = st.selectbox("Select Factor to Neutralize:", options=factor_options, index=0, key='factor_select_table')
    factor_results_df, error_msg = calculate_all_factor_hedges(trade_selection_factor, factor_selection, factor_sensitivities_df, Sigma_Raw_df)
    if error_msg:
        st.error(f"Factor hedge calculation failed: {error_msg}")
    elif not factor_results_df.empty:
        factor_results_df_clean = factor_results_df.dropna(subset=['Residual Volatility (BPS)'])
        best_hedge_row = factor_results_df_clean.iloc[0]
        st.success(f"Best Suitable Hedge for Neutralizing {factor_selection} is {best_hedge_row['Hedge Instrument']}.")
        st.markdown(f"- Trade Sensitivity: {best_hedge_row['Trade Sensitivity']:.4f}\n- Best Hedge Ratio (k_factor): {best_hedge_row['Factor Hedge Ratio (k_factor)']:.4f}\n- Lowest Residual Volatility: {best_hedge_row['Residual Volatility (BPS)']:.2f} BPS")
        st.dataframe(factor_results_df_clean.style.format({'Trade Sensitivity': "{:.4f}", 'Hedge Sensitivity': "{:.4f}", 'Factor Hedge Ratio (k_factor)': "{:.4f}", 'Residual Volatility (BPS)': "{:.2f}"}), use_container_width=True)
    else:
        st.info("No factor-hedge results to display.")

# -----------------------------
# 9. Hedge Excess Return & Reward-to-Risk Dashboard
# -----------------------------
st.header("9. Hedge Excess Return & Reward-to-Risk Dashboard")
if 'historical_outrights_df' in locals() and historical_outrights_df is not None and not historical_outrights_df.empty:
    ranking_df, err = rank_all_hedge_excess_returns_with_risk(historical_outrights_df, all_results_df_full_gen if 'all_results_df_full_gen' in locals() else None)
    if err:
        st.warning(err)
    else:
        st.sidebar.header("📊 Hedge Filters")
        instr_filter = st.sidebar.multiselect("Instrument Type", ["Spread", "Fly", "Double Fly", "Other"], default=["Spread", "Fly", "Double Fly"])
        sort_choice = st.sidebar.radio("Sort by", ["Reward-to-Risk", "Excess Return (bps)", "Residual Volatility (BPS)"], index=0)
        if sort_choice not in ranking_df.columns:
            # ensure column exists
            sort_choice = "Reward-to-Risk" if "Reward-to-Risk" in ranking_df.columns else "Excess Return (bps)"
        ascending = sort_choice in ["Residual Volatility (BPS)", "Excess Return (bps)"]
        filtered_df = ranking_df[ranking_df["Instrument Type"].isin(instr_filter)]
        filtered_df = filtered_df.sort_values(by=sort_choice, ascending=ascending, na_position='last')
        st.markdown(f"Filtered by **{', '.join(instr_filter)}**, sorted by **{sort_choice}**.")
        st.dataframe(filtered_df.style.format({"Market Price": "{:.4f}", "PCA Fair Price": "{:.4f}", "Excess Return (bps)": "{:.2f}", "Excess Return ($)": "{:.0f}", "Residual Volatility (BPS)": "{:.2f}", "Reward-to-Risk": "{:.2f}"}), use_container_width=True)
        st.subheader("Top 5 Best Reward-to-Risk Hedges")
        st.dataframe(filtered_df.head(5).style.format({"Excess Return (bps)": "{:.2f}", "Residual Volatility (BPS)": "{:.2f}", "Reward-to-Risk": "{:.2f}"}))
        # Scatter risk-reward
        fig, ax = plt.subplots(figsize=(9, 6))
        types = filtered_df["Instrument Type"].astype("category")
        colors = types.cat.codes
        sc = ax.scatter(filtered_df["Residual Volatility (BPS)"], filtered_df["Excess Return (bps)"], c=colors, cmap='tab10', alpha=0.8)
        ax.set_xlabel("Residual Volatility (BPS)"); ax.set_ylabel("Excess Return (bps)"); ax.set_title("Risk – Reward Map of Hedges vs PCA Fair Value")
        # color legend
        handles = []
        labels = list(types.cat.categories)
        for i, lab in enumerate(labels):
            handles.append(plt.Line2D([], [], marker='o', color=plt.cm.tab10(i), linestyle='', label=lab))
        ax.legend(handles=handles, title="Instrument Type")
        st.pyplot(fig)
else:
    st.warning("Historical fair-value data not found. Run PCA analysis first.")
