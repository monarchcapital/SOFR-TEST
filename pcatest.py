# -*- coding: utf-8 -*-
# SOFR/Treasury Curve PCA Backtesting Engine (pcattest.py)
#
# This script performs a walk-forward validation (backtest) of the PCA-based
# yield curve forecasting models for the US SOFR/Treasury curve.
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
import datetime
from datetime import timedelta

# --- Setup ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="SOFR Curve Backtesting Engine")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------------
# CORE DATA PROCESSING AND PCA LOGIC
# ---------------------------------------------------------------------------------

def build_std_grid_by_rule(max_year=30.0):
    """
    Defines the standard set of tenors (TTM) for the US SOFR/Treasury curve.
    We use standard points: 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 30Y.
    """
    # Define standard US tenors in years (0.25=3M, 0.5=6M)
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
    
    # Filter the standard tenors to the chosen max_year
    std_grid_tenors = tenors[tenors <= max_year]
    
    # Create column names: convert back to standard format (e.g., 1.0 -> '1Y')
    std_grid_cols = []
    for t in std_grid_tenors:
        if t < 1.0:
             std_grid_cols.append(f"{int(t*12)}M")
        else:
            std_grid_cols.append(f"{int(t)}Y")
            
    return list(std_grid_tenors), std_grid_cols


def row_to_std_grid(row, std_tenors, date_col='Date'):
    """
    Converts a single day's raw SOFR/Treasury rates (columns are tenors) 
    into a standardized grid via linear interpolation.
    
    The function expects the column headers of the input data to be the tenors
    (e.g., '1Y', '2Y', '5Y').
    """
    raw_rates = []
    raw_tenors = []
    
    # Iterate through all columns, ignoring the date column
    for col in row.index:
        if col != date_col:
            rate = row[col]
            
            # Use the column name directly as the tenor (TTM)
            try:
                col_upper = col.upper().strip()
                tenor = None
                
                if 'M' in col_upper:
                    tenor = float(col_upper.replace('M', '')) / 12.0 # Months to Years
                elif 'Y' in col_upper:
                    tenor = float(col_upper.replace('Y', '')) # Years
                
                if tenor is not None and pd.notna(rate):
                    raw_rates.append(rate)
                    raw_tenors.append(tenor)
                    
            except ValueError:
                # Skip non-numeric/non-tenor columns
                continue

    if len(raw_rates) < 2 or max(raw_tenors) < max(std_tenors):
        # Cannot interpolate with less than 2 points or if max tenor is too short
        return np.full(len(std_tenors), np.nan)

    # Sort tenors and rates for interpolation
    sorted_pairs = sorted(zip(raw_tenors, raw_rates))
    raw_tenors, raw_rates = zip(*sorted_pairs)
    
    # Interpolation using linear method
    f = interp1d(
        raw_tenors, 
        raw_rates, 
        kind='linear',
        fill_value='extrapolate'
    )
    
    # Calculate the interpolated rates for the standard grid points
    interpolated_rates = f(std_tenors)

    return interpolated_rates

def calculate_curve_metrics(df_std, std_cols, spread_defs, fly_defs):
    """
    Calculates outright rates, spreads, and butterflies and concatenates them
    into a single matrix for PCA.
    """
    # 1. Outright Rates (Standard Grid)
    outright_df = df_std[std_cols].copy()
    
    # 2. Spreads (Slope)
    spread_cols = []
    spread_df = pd.DataFrame(index=df_std.index)
    for t_long, t_short in spread_defs:
        spread_col = f'{t_long}-{t_short}'
        spread_df[spread_col] = df_std[t_long] - df_std[t_short]
        spread_cols.append(spread_col)
    
    # 3. Butterflies (Flies/Curvature)
    fly_cols = []
    fly_df = pd.DataFrame(index=df_std.index)
    for t_short, t_mid, t_long in fly_defs:
        fly_col = f'{t_short}x{t_mid}x{t_long}'
        # Standard butterfly: 2 * Mid - Short - Long
        fly_df[fly_col] = (2 * df_std[t_mid]) - df_std[t_short] - df_std[t_long]
        fly_cols.append(fly_col)

    # Combine all metrics into a single matrix
    full_df = pd.concat([outright_df, spread_df, fly_df], axis=1)
    full_cols = std_cols + spread_cols + fly_cols
    
    return full_df, std_cols, spread_cols, fly_cols

# ---------------------------------------------------------------------------------
# PCA AND FORECASTING FUNCTIONS
# ---------------------------------------------------------------------------------

def fit_pca_and_var(df_train_diff, n_components, var_lag):
    """Fits PCA and VAR model on the training data."""
    # 1. Standardize (Center)
    scaler = StandardScaler(with_std=False)
    df_scaled = scaler.fit_transform(df_train_diff)
    
    # 2. PCA Decomposition
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    pc_df = pd.DataFrame(principal_components, index=df_train_diff.index)
    
    # 3. VAR Model on Principal Components
    var_model = VAR(pc_df)
    var_results = var_model.fit(var_lag)
    
    return scaler, pca, var_results, var_model.names

def forecast_next_day(scaler, pca, var_results, last_pc_vector, last_rate_vector, method='VAR'):
    """Forecasts the next day's curve vector (all metrics)."""
    
    # 1. Forecast Principal Components (VAR)
    if method == 'VAR':
        # VAR forecast requires the last vector of PCs
        pc_forecast = var_results.forecast(last_pc_vector.values.reshape(1, -1), steps=1)[0]
    else: # Default to simple average (mean)
        # Use the mean of the training PCs as the forecast
        pc_forecast = last_pc_vector * 0 # Simpler, less effective forecast placeholder
    
    # 2. Reconstruct (Inverse PCA) - Forecasted Curve Change
    # pc_forecast * pca.components_ + scaler.mean_
    forecast_diff_scaled = pc_forecast @ pca.components_
    forecast_diff = forecast_diff_scaled + scaler.mean_
    
    # 3. Integrate (Add to previous day's level) - Forecasted Curve Level
    forecast_level = last_rate_vector.values + forecast_diff
    
    return forecast_level


# ---------------------------------------------------------------------------------
# BACKTESTING LOOP
# ---------------------------------------------------------------------------------

def run_backtest(df_full, std_cols, spread_cols, fly_cols, **params):
    """Runs the rolling-window backtest."""
    
    window_size = params['window_size']
    n_components = params['n_components']
    var_lag = params['var_lag']
    start_date = params['start_date']
    end_date = params['end_date']

    # Filter data for the analysis period
    df_filtered = df_full.loc[start_date:end_date].copy()
    
    # Calculate daily changes (difference)
    df_diff = df_filtered.diff().dropna()
    
    # Adjust filtered data to align with diff (start one day later)
    df_filtered = df_filtered.loc[df_diff.index].copy()
    
    backtest_start_date = df_filtered.index[window_size]
    st.info(f"Starting backtest from {backtest_start_date.strftime('%Y-%m-%d')} using a {window_size}-day rolling window.")

    results_list = []
    
    # --- Rolling Window Loop ---
    for i in range(window_size, len(df_filtered) - 1):
        # Define Training Window
        train_end_index = i
        train_start_index = i - window_size
        
        df_train_diff = df_diff.iloc[train_start_index:train_end_index]
        last_train_level = df_filtered.iloc[train_end_index - 1] # Day t-1 level
        
        # Day to Forecast (Actual Day t)
        forecast_date = df_filtered.index[train_end_index]
        actual_level = df_filtered.iloc[train_end_index]
        
        # 1. Fit PCA and VAR/Forecasting Model
        try:
            scaler, pca, var_results, pc_names = fit_pca_and_var(df_train_diff, n_components, var_lag)
        except Exception as e:
            # st.warning(f"Skipping {forecast_date.strftime('%Y-%m-%d')} due to model fitting error: {e}")
            continue

        # 2. Prepare Last Principal Component Vector
        # Transform the last change vector (df_diff.iloc[train_end_index - 1]) into PCs
        last_diff_vector = df_diff.iloc[train_end_index - 1].values
        last_diff_scaled = scaler.transform(last_diff_vector.reshape(1, -1))
        last_pc_vector = pd.Series(pca.transform(last_diff_scaled)[0], index=pc_names)
        
        # 3. Forecast
        predicted_level = forecast_next_day(scaler, pca, var_results, last_pc_vector, last_train_level, method='VAR')

        # 4. Record Results
        result = {
            'Date': forecast_date.strftime('%Y-%m-%d'),
            'Predicted_Curve': predicted_level,
            'Actual_Curve': actual_level.values
        }
        results_list.append(result)
        
    return pd.DataFrame(results_list)

# ---------------------------------------------------------------------------------
# STREAMLIT UI AND EXECUTION
# ---------------------------------------------------------------------------------

def main():
    st.title("🏛️ SOFR/Treasury Curve PCA Backtesting Engine")
    st.markdown("---")

    # --- Sidebar for Parameters ---
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload SOFR/Treasury Data (CSV)", type="csv")
    
    # Default parameters based on Brazil DI analysis (adjust for US market norms)
    st.sidebar.header("Backtest Parameters")
    window_size = st.sidebar.slider("Rolling Window Size (Days)", min_value=120, max_value=500, value=252)
    n_components = st.sidebar.slider("PCA Components (Factors)", min_value=1, max_value=5, value=3)
    var_lag = st.sidebar.slider("VAR Model Lag (Days)", min_value=1, max_value=5, value=1)
    
    # Standard SOFR/Treasury tenors for backtesting grid
    std_tenors_val, std_cols = build_std_grid_by_rule()
    st.sidebar.markdown(f"**Standard Grid Points:** {', '.join(std_cols)}")
    
    # Define Spreads and Flies (standard US conventions)
    spread_defs = [('10Y', '2Y'), ('30Y', '10Y')]
    fly_defs = [('2Y', '5Y', '10Y'), ('5Y', '10Y', '30Y')]

    # --- Main Execution ---
    if uploaded_file is not None:
        try:
            data_df = pd.read_csv(uploaded_file)
            
            # --- Data Preprocessing ---
            # 1. Format Date Column
            date_col = data_df.columns[0]
            data_df[date_col] = pd.to_datetime(data_df[date_col])
            data_df = data_df.set_index(date_col).sort_index()

            # 2. Filter data for tenor columns
            rate_columns = [col for col in data_df.columns if any(s in col.upper() for s in ['M', 'Y'])]
            if not rate_columns:
                st.error("Could not find any columns representing SOFR/Treasury tenors (e.g., '1Y', '5Y', '3M'). Please check your file headers.")
                return

            st.sidebar.markdown("---")
            st.sidebar.subheader("Backtest Period")
            max_date = data_df.index.max()
            min_date = data_df.index.min()
            
            # Ensure backtest start is after the window size
            initial_start_date = min_date + timedelta(days=window_size + 10) 
            if initial_start_date >= max_date:
                 initial_start_date = max_date - timedelta(days=window_size + 10)
                 if initial_start_date <= min_date:
                     initial_start_date = min_date

            date_range = st.sidebar.date_input(
                "Select Start and End Dates",
                value=[initial_start_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) != 2:
                st.warning("Please select a start and an end date for the backtest.")
                return

            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])

            if start_date < min_date or end_date > max_date:
                st.error("Selected dates are outside the data range.")
                return
            
            # --- STEP 1: Interpolation to Standard Grid ---
            st.subheader("1. Standardizing Curve Data")
            
            # Apply interpolation to each row
            tqdm_text = st.empty()
            tqdm_text.text("Interpolating to standard grid...")
            
            df_std = data_df.apply(lambda row: row_to_std_grid(row, std_tenors_val, date_col=date_col), axis=1, result_type='expand')
            df_std.columns = std_cols
            df_std = df_std.dropna()
            
            tqdm_text.text(f"Interpolation complete. {len(df_std)} days of data available.")

            # --- STEP 2: Calculate Combined Metrics (Outright, Spread, Fly) ---
            st.subheader("2. Calculating PCA Input Metrics (Outright, Spread, Fly)")
            df_full, std_cols, spread_cols, fly_cols = calculate_curve_metrics(df_std, std_cols, spread_defs, fly_defs)
            
            total_cols = len(df_full.columns)
            if total_cols == 0:
                 st.error("No metrics could be calculated. Check tenor definitions.")
                 return

            st.info(f"Full PCA input matrix has {total_cols} dimensions: {len(std_cols)} Outrights, {len(spread_cols)} Spreads, {len(fly_cols)} Flies.")
            st.write(df_full.head())
            
            # --- STEP 3: Run Backtest ---
            st.subheader("3. Running Walk-Forward Backtest")
            
            params = {
                'window_size': window_size, 
                'n_components': n_components, 
                'var_lag': var_lag, 
                'start_date': start_date, 
                'end_date': end_date
            }
            
            if len(df_full.loc[start_date:end_date]) < window_size * 1.5:
                st.error(f"Not enough data for backtest. Need at least {window_size * 1.5} days of data. Found {len(df_full.loc[start_date:end_date])} days.")
                return
            
            results_df = run_backtest(df_full, std_cols, spread_cols, fly_cols, **params)
            
            if results_df.empty:
                st.error("Backtest failed to generate results. Check if the backtest period is large enough for the window size.")
                return

            results_df['Date'] = pd.to_datetime(results_df['Date'])
            results_df = results_df.set_index('Date').sort_index()

            st.success(f"Backtest completed successfully, generating {len(results_df)} daily forecasts.")

            # --- STEP 4: Analyze and Plot Results ---
            st.markdown("---")
            st.header("📊 Backtest Performance Analysis")

            all_cols = std_cols + spread_cols + fly_cols
            
            # Calculate daily prediction error (vector subtraction)
            results_df['Error_Vector'] = results_df.apply(
                lambda row: row['Predicted_Curve'] - row['Actual_Curve'], axis=1
            )
            
            # Calculate RMSE and MAE for each metric
            error_metrics = []
            for i, col in enumerate(all_cols):
                errors = results_df['Error_Vector'].apply(lambda x: x[i])
                rmse = np.sqrt(mean_squared_error(results_df['Actual_Curve'].apply(lambda x: x[i]), results_df['Predicted_Curve'].apply(lambda x: x[i])))
                mae = mean_absolute_error(results_df['Actual_Curve'].apply(lambda x: x[i]), results_df['Predicted_Curve'].apply(lambda x: x[i]))
                
                # Determine metric type for sorting/display
                metric_type = 'Outright'
                if col in spread_cols:
                    metric_type = 'Spread'
                elif col in fly_cols:
                    metric_type = 'Fly'
                
                error_metrics.append({
                    'Metric': col,
                    'Type': metric_type,
                    'RMSE (bps)': round(rmse * 10000, 2), # Convert to basis points
                    'MAE (bps)': round(mae * 10000, 2)
                })

            error_df = pd.DataFrame(error_metrics)
            
            # Sort by type and then RMSE
            error_df = error_df.sort_values(by=['Type', 'RMSE (bps)'], ascending=[False, True])
            
            st.subheader("Error Metrics (Basis Points)")
            st.dataframe(error_df.set_index('Metric'))

            # --- Plotting ---
            st.markdown("---")
            st.subheader("Predicted vs. Actual Curves (Outrights)")
            
            # Plotting the outright rates (the first N=len(std_cols) metrics)
            plot_cols = std_cols
            num_plots = len(plot_cols)
            
            rows = (num_plots + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows))
            axes = axes.flatten()

            for i, col in enumerate(plot_cols):
                ax = axes[i]
                actual_rates = results_df['Actual_Curve'].apply(lambda x: x[i])
                predicted_rates = results_df['Predicted_Curve'].apply(lambda x: x[i])
                
                actual_rates.plot(ax=ax, label='Actual', linewidth=1.5, color='blue')
                predicted_rates.plot(ax=ax, label='Predicted', linewidth=1.5, linestyle='--', color='red')
                
                ax.set_title(f'Forecast vs. Actual: {col}', fontsize=12)
                ax.set_ylabel('Rate (%)')
                ax.legend(loc='upper right')
            
            # Hide empty subplots
            for j in range(num_plots, len(axes)):
                fig.delaxes(axes[j])
                
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("Raw Curve Data (Spreads and Flies)")
            st.write("This table contains the raw predicted and actual spread and fly vectors for each day of the backtest.")
            
            raw_spreads_flies_df = pd.DataFrame(results_df.index).set_index('Date')
            
            # Get the start indices for spreads and flies within the full curve vector
            spreads_start_idx = len(std_cols)
            spreads_end_idx = spreads_start_idx + len(spread_cols)
            flies_start_idx = spreads_end_idx
            
            # Extract predicted and actual spreads and flies into their own columns
            for i, col in enumerate(spread_cols):
                raw_spreads_flies_df[f"Predicted_Spread_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[spreads_start_idx + i])
                raw_spreads_flies_df[f"Actual_Spread_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[spreads_start_idx + i])
                
            for i, col in enumerate(fly_cols):
                raw_spreads_flies_df[f"Predicted_Fly_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[flies_start_idx + i])
                raw_spreads_flies_df[f"Actual_Fly_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[flies_start_idx + i])

            st.dataframe(raw_spreads_flies_df)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.markdown("---")
            st.markdown("""
            **Troubleshooting:**
            1. Ensure your uploaded CSV has a **Date column** as the first column.
            2. Ensure subsequent columns are the **tenors** (e.g., `1Y`, `2Y`, `5Y`) and contain the yield/rate data.
            3. Check that the dates are within a reasonable range for the selected window size.
            """)

if __name__ == "__main__":
    main()
