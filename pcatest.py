import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from scipy.linalg import solve

# --- Configuration ---
st.set_page_config(layout="wide", page_title="SOFR Futures PCA & Advanced Hedging Analyzer")

# --- Helper Functions for Data Processing ---

def load_data(uploaded_file):
    """Loads CSV data into a DataFrame, adapting to price or expiry file formats."""
    if uploaded_file is None:
        return None
        
    try:
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue().decode("utf-8")
        uploaded_file.seek(0)
            
        # Case 1: Expiry File (e.g., MATURITY,DATE)
        if 'MATURITY,DATE' in file_content.split('\n')[0].upper():
            df = pd.read_csv(uploaded_file, sep=',')
            df = df.rename(columns={'MATURITY': 'Contract', 'DATE': 'ExpiryDate'})
            df = df.set_index('Contract')
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
            df.index.name = 'Contract'
            return df

        # Case 2: Price File (Date as index)
        df = pd.read_csv(
            uploaded_file, 
            index_col=0, 
            parse_dates=True, 
            na_values=['', 'NA', '#N/A', 'N/A', '#DIV/0!', '#REF!', '#VALUE!']
        )
        # Drop rows/columns that are all NaN/zero
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        # Only keep numeric columns (prices)
        df = df.select_dtypes(include=np.number)
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_analysis_contracts(price_df, expiry_df, analysis_date):
    """Filters prices to only include contracts that have not yet expired as of the analysis_date,
       and sorts them chronologically by expiry date."""
    if price_df is None or expiry_df is None:
        return pd.DataFrame(), []

    # Filter expiry data to contracts still active
    active_contracts = expiry_df[expiry_df['ExpiryDate'] >= analysis_date].copy()
    
    # Filter price data to only include active contracts
    price_df_filtered = price_df[active_contracts.index.intersection(price_df.columns)]
    
    # Sort contracts based on expiry date
    sorted_contracts = active_contracts.sort_values(by='ExpiryDate').index.tolist()
    
    # Ensure the final contract list only includes those present in price_df_filtered
    final_contracts = [c for c in sorted_contracts if c in price_df_filtered.columns]

    return price_df_filtered[final_contracts], final_contracts

def calculate_derivatives(df, sorted_contracts, derivative_type, gap_months):
    """Calculates spreads (C1-C2) or butterflies (C1-2*C2+C3) based on a contract gap."""
    df_out = pd.DataFrame(index=df.index)
    gap = gap_months // 3 # Futures are quarterly
    
    if gap < 1:
        return df_out

    if derivative_type == 'Spread':
        for i in range(len(sorted_contracts) - gap):
            c1, c2 = sorted_contracts[i], sorted_contracts[i + gap]
            col_name = f"{c1}-{c2}"
            df_out[col_name] = df[c1] - df[c2]
            
    elif derivative_type == 'Butterfly':
        for i in range(len(sorted_contracts) - 2 * gap):
            c1 = sorted_contracts[i]
            c2 = sorted_contracts[i + gap]
            c3 = sorted_contracts[i + 2 * gap]
            col_name = f"{c1}-2x{c2}+{c3}"
            df_out[col_name] = df[c1] - 2 * df[c2] + df[c3]
            
    return df_out

# --- PCA Core Functions ---

def perform_pca_on_prices(df, pc_count):
    """Performs PCA on the historical price levels."""
    
    if df.shape[0] < pc_count or df.shape[1] < pc_count:
        return None, None, None, None, None

    # 1. Prepare data: Center the prices (P - Mean(P))
    mean_prices = df.mean()
    centered_prices = df - mean_prices

    # 2. Perform PCA
    pca = PCA(n_components=min(df.shape[1], pc_count))
    pca.fit(centered_prices)

    # 3. Extract components
    loadings = pd.DataFrame(
        pca.components_.T, 
        index=df.columns, 
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    scores = pd.DataFrame(
        pca.transform(centered_prices), 
        index=df.index, 
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    explained_variance = pca.explained_variance_ratio_

    return pca, mean_prices, loadings, scores, explained_variance

def reconstruct_fair_curve_from_prices(mean_prices, loadings, scores, analysis_dt, pc_count):
    """Reconstructs the fair curve (outright prices) based on the top K factors."""
    try:
        # Get the PC Scores for the analysis date
        date_scores = scores.loc[analysis_dt, :].values
        
        # Use only the top K PCs
        loadings_k = loadings.iloc[:, :pc_count].values
        date_scores_k = date_scores[:pc_count]

        # Reconstruction: Mean + (Loadings_k @ Scores_k)
        # Loadings matrix (N contracts x K PCs) multiplied by Score vector (K PCs x 1)
        reconstruction = mean_prices.values + (loadings_k @ date_scores_k)
        
        fair_curve_prices = pd.Series(reconstruction, index=mean_prices.index)
        return fair_curve_prices
    
    except KeyError:
        # If the analysis date is not in the scores/price data
        return None
    except ValueError as e:
        # If matrix dimensions don't align (shouldn't happen if logic is correct)
        st.error(f"Error in reconstruction logic: {e}")
        return None


def calculate_snapshot_comparison(historical_df, fair_curve_prices, analysis_dt, is_outright=False):
    """Calculates the mispricing of all instruments (outright or derivative) for the analysis date."""
    
    if fair_curve_prices is None:
        return None, None

    # Get original market snapshot
    original_snapshot = historical_df.loc[analysis_dt]
    
    # 1. Calculate the PCA Fair Value for all instruments
    pca_fair_values = pd.Series(index=historical_df.columns)
    
    if is_outright:
        # For outrights, the fair price is the reconstructed price
        pca_fair_values = fair_curve_prices.loc[original_snapshot.index]
        mispricing = original_snapshot - pca_fair_values

    else:
        # For derivatives, re-apply the formula to the fair outright prices
        for col in original_snapshot.index:
            if '2x' in col: # Butterfly (C1 - 2*C2 + C3)
                parts = col.split('-')
                c1 = parts[0]
                c2 = parts[1].split('x')[1]
                c3 = parts[2].replace('+', '')
                
                if c1 in fair_curve_prices.index and c2 in fair_curve_prices.index and c3 in fair_curve_prices.index:
                    fair_val = fair_curve_prices[c1] - 2 * fair_curve_prices[c2] + fair_curve_prices[c3]
                    pca_fair_values[col] = fair_val
                else:
                    pca_fair_values[col] = np.nan

            else: # Spread (C1 - C2)
                c1, c2 = col.split('-')
                if c1 in fair_curve_prices.index and c2 in fair_curve_prices.index:
                    fair_val = fair_curve_prices[c1] - fair_curve_prices[c2]
                    pca_fair_values[col] = fair_val
                else:
                    pca_fair_values[col] = np.nan
        
        pca_fair_values = pca_fair_values.dropna()
        original_snapshot = original_snapshot.loc[pca_fair_values.index]
        mispricing = original_snapshot - pca_fair_values

    # 2. Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Original': original_snapshot,
        'PCA Fair': pca_fair_values,
    }).dropna()

    return comparison_df, mispricing

# --- Hedging Core Functions ---

def calculate_factor_sensitivities(all_instruments_df, loadings_df):
    """
    Calculates the Factor Sensitivity (Beta) for every instrument (outrights, spreads, flies)
    against each PC factor.
    
    Beta(D, PCk) = sum over contracts (Weight(D, Ci) * Loading(Ci, PCk))
    """
    sensitivities = pd.DataFrame(columns=loadings_df.columns)
    
    # 1. Handle Outrights
    for contract in loadings_df.index:
        sensitivities.loc[contract] = loadings_df.loc[contract]
        
    # 2. Handle Derivatives (Spreads and Flies)
    for instrument in all_instruments_df.columns:
        # Get the components and their weights
        weights = {}
        
        if '2x' in instrument: # Butterfly: C1 - 2*C2 + C3
            parts = instrument.split('-')
            c1 = parts[0]
            c2 = parts[1].split('x')[1]
            c3 = parts[2].replace('+', '')
            weights[c1] = 1
            weights[c2] = -2
            weights[c3] = 1
        else: # Spread: C1 - C2
            c1, c2 = instrument.split('-')
            weights[c1] = 1
            weights[c2] = -1

        # Calculate the net sensitivity for this instrument
        net_sensitivity = pd.Series(0.0, index=loadings_df.columns)
        
        for contract, weight in weights.items():
            if contract in loadings_df.index:
                net_sensitivity += weight * loadings_df.loc[contract]
        
        sensitivities.loc[instrument] = net_sensitivity

    return sensitivities

def calculate_mvhr(trade_instrument, sensitivities, scores, pc_count):
    """
    Calculates Minimum Variance Hedge Ratios (MVHR) for a trade against all hedge candidates.
    h* = Cov(T, H) / Var(H)
    """
    
    # Use only the selected PC factors (Betas)
    sensitivities_k = sensitivities.iloc[:, :pc_count]
    
    # Covariance Matrix of PC Scores (S)
    # The scores are already centered, so we can use cov() directly
    scores_k = scores.iloc[:, :pc_count]
    cov_s = scores_k.cov() * 252 # Annualize covariance (assuming daily data)
    
    # Get the Beta vector for the Trade (T)
    beta_t = sensitivities_k.loc[trade_instrument].values
    
    # Initial Volatility of the Trade (Var(T))
    var_t = beta_t @ cov_s @ beta_t.T
    
    results = []
    
    # Iterate over all possible hedge candidates (excluding the trade itself)
    hedge_candidates = [i for i in sensitivities_k.index if i != trade_instrument]
    
    for hedge_instrument in hedge_candidates:
        beta_h = sensitivities_k.loc[hedge_instrument].values
        
        # Variance of the Hedge (Var(H))
        var_h = beta_h @ cov_s @ beta_h.T
        
        # Covariance between Trade and Hedge (Cov(T, H))
        cov_th = beta_t @ cov_s @ beta_h.T
        
        # MVHR (h*)
        if var_h > 1e-9: # Avoid division by zero
            mvhr = cov_th / var_h
        else:
            mvhr = 0 
        
        # Residual Variance of the Hedged Portfolio (Var(T - h*H))
        residual_var = var_t - mvhr * cov_th
        
        # Ensure residual variance is non-negative due to floating point arithmetic
        residual_var = max(0, residual_var) 
        
        vol_t = np.sqrt(var_t) * 10000 # Convert to BPS volatility
        vol_residual = np.sqrt(residual_var) * 10000 # Convert to BPS volatility
        
        # Risk Reduction
        risk_reduction = (1 - (vol_residual / vol_t)) * 100 if vol_t > 0 else 0
        
        results.append({
            'Hedge Candidate': hedge_instrument,
            'Trade Volatility (BPS)': vol_t,
            'MVHR (Units of Hedge/Unit of Trade)': mvhr,
            'Residual Volatility (BPS)': vol_residual,
            'Risk Reduction (%)': risk_reduction
        })
        
    results_df = pd.DataFrame(results)
    
    # Sort by risk reduction (descending)
    return results_df.sort_values(by='Risk Reduction (%)', ascending=False)

def solve_scenario_hedge(trade_instrument, hedge1, hedge2, sensitivities):
    """
    Solves for the hedge weights h1 and h2 required to neutralize a trade
    to PC1 (Level) and PC2 (Slope) factors.
    
    A * H = B  =>  H = inv(A) * B
    """
    try:
        # Get Betas for PC1 and PC2 for all three instruments
        beta_t = sensitivities.loc[trade_instrument, ['PC1', 'PC2']].values
        beta_h1 = sensitivities.loc[hedge1, ['PC1', 'PC2']].values
        beta_h2 = sensitivities.loc[hedge2, ['PC1', 'PC2']].values

        # A is the matrix of hedge betas (2x2)
        A = np.array([beta_h1, beta_h2]).T 
        
        # B is the vector of trade betas (2x1)
        B = beta_t

        # Solve for H (h1, h2)
        H = solve(A, B)
        h1, h2 = H[0], H[1]
        
        # Calculate the residual PC3 exposure
        if 'PC3' in sensitivities.columns:
            beta_t_pc3 = sensitivities.loc[trade_instrument, 'PC3']
            beta_h1_pc3 = sensitivities.loc[hedge1, 'PC3']
            beta_h2_pc3 = sensitivities.loc[hedge2, 'PC3']
            residual_pc3 = beta_t_pc3 - h1 * beta_h1_pc3 - h2 * beta_h2_pc3
        else:
            residual_pc3 = 0.0

        return h1, h2, residual_pc3
    except np.linalg.LinAlgError:
        return None, None, None # Singular matrix (hedges are too correlated/identical)
    except KeyError:
        return None, None, None # PC1 or PC2 not available

# --- Streamlit Application Layout ---

st.title("🏛️ SOFR Futures PCA & Advanced Hedging Analyzer")
st.markdown("Use historical price data to decompose curve movements into **Principal Components (Level, Slope, Curve)** and calculate **optimal hedge ratios** based on Minimum Variance and specific market scenarios.")

# --- 1. Data Upload and Parameters ---
with st.sidebar:
    st.header("1. Data & Parameters")
    
    price_file = st.file_uploader("Upload Historical Price Data (e.g., `sofr rates.csv`)", type="csv")
    expiry_file = st.file_uploader("Upload Expiry Data (e.g., `EXPIRY (2).csv`)", type="csv")
    
    price_df = load_data(price_file)
    expiry_df = load_data(expiry_file)

    if price_df is not None:
        # Safely determine min/max dates
        try:
            min_date = price_df.index.min().to_pydatetime().date()
            max_date = price_df.index.max().to_pydatetime().date()
        except AttributeError:
             # Handle case where index might not be datetime objects after parsing
             min_date = date(2020, 1, 1)
             max_date = date.today()
    else:
        min_date = date(2020, 1, 1)
        max_date = date.today()

    st.subheader("Time Range")
    # Date Pickers for historical range (PCA calibration)
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("PCA Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col_end:
        end_date = st.date_input("PCA End Date", value=max_date, min_value=min_date, max_value=max_date)
        
    # Single date for market snapshot analysis
    analysis_date = st.date_input("Curve Analysis Date", value=max_date, min_value=start_date, max_value=max_date)
    analysis_dt = datetime.combine(analysis_date, datetime.min.time())

    # PCA Parameters
    st.subheader("PCA Settings")
    pc_count = st.slider("Number of Principal Components (K)", min_value=2, max_value=6, value=3)

# --- Main Logic Execution ---

if price_df is not None and expiry_df is not None:
    # 1. Filter Data based on time range and expiry
    price_df_filtered, sorted_contracts = get_analysis_contracts(price_df, expiry_df, analysis_date)
    
    # Ensure all required contracts are available up to the analysis date
    historical_price_df = price_df_filtered.loc[start_date:end_date]
    
    if historical_price_df.empty:
        st.error("No historical data found in the selected date range and/or contracts have all expired. Please adjust dates.")
        st.stop()
        
    # 2. Calculate Derivatives
    # Fixed 3M gap for spreads/butterflies for simplicity
    historical_spreads_df = calculate_derivatives(historical_price_df, sorted_contracts, 'Spread', 3)
    historical_butterflies_df = calculate_derivatives(historical_price_df, sorted_contracts, 'Butterfly', 3)
    
    # Consolidate all instruments for sensitivity calculation
    all_instruments_df = pd.concat([historical_price_df, historical_spreads_df, historical_butterflies_df], axis=1).dropna(axis=1, how='all')

    # 3. Perform PCA
    pca_result = perform_pca_on_prices(historical_price_df, pc_count)

    if pca_result[0] is not None:
        pca, mean_prices, loadings, scores, explained_variance = pca_result
        fair_curve_prices = reconstruct_fair_curve_from_prices(mean_prices, loadings, scores, analysis_dt, pc_count)

        # --- 2. PCA Results ---
        st.header("2. PCA Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Explained Variance")
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                'Variance Explained (%)': (explained_variance * 100).round(2),
                'Cumulative (%)': (np.cumsum(explained_variance) * 100).round(2)
            })
            st.dataframe(variance_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("PC Loadings (Outright Prices)")
            st.markdown("Sensitivities of each contract's price to a 1-unit move in the PC factor.")
            
            fig_loadings, ax_loadings = plt.subplots(figsize=(10, 5))
            loadings.iloc[:, :pc_count].plot(kind='bar', ax=ax_loadings)
            ax_loadings.set_title("PC Loadings by Contract")
            ax_loadings.set_ylabel("Loading (Price Sensitivity)")
            ax_loadings.set_xlabel("Futures Contract")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            st.pyplot(fig_loadings)


        # --- 3. Mispricing Analysis ---
        st.header(f"3. Market Mispricing Snapshot ({analysis_date.strftime('%Y-%m-%d')})")
        
        # 3.1 Outright Mispricing
        st.subheader("3.1 Outright Contract Mispricing (P - PCA Fair)")
        try:
            comparison_outright, mispricing_outright = calculate_snapshot_comparison(historical_price_df, fair_curve_prices, analysis_dt, is_outright=True)
            
            fig_outright, ax_outright = plt.subplots(figsize=(12, 5))
            mispricing_outright.mul(10000).plot(kind='bar', ax=ax_outright, 
                                                color=np.where(mispricing_outright.mul(10000) > 0, 'green', 'red'))
            ax_outright.set_title(f"Outright Mispricing (Actual - PCA Fair) in BPS ({pc_count} Factors)")
            ax_outright.set_ylabel("Mispricing (BPS)")
            ax_outright.set_xlabel("Futures Contract")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            st.pyplot(fig_outright)
            
            # Detailed Table
            detailed_comparison = comparison_outright.copy()
            detailed_comparison.columns = ['Original Price', 'PCA Fair Price']
            detailed_comparison['Original Rate (%)'] = (100 - detailed_comparison['Original Price'])
            detailed_comparison['PCA Fair Rate (%)'] = (100 - detailed_comparison['PCA Fair Price'])
            detailed_comparison['Mispricing (BPS)'] = mispricing_outright * 10000
            st.dataframe(
                detailed_comparison.style.format({
                    'Original Price': "{:.4f}", 'PCA Fair Price': "{:.4f}",
                    'Original Rate (%)': "{:.4f}", 'PCA Fair Rate (%)': "{:.4f}",
                    'Mispricing (BPS)': "{:.2f}"
                }),
                use_container_width=True
            )
            
        except KeyError:
             st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the price data.")
        
        
        # 3.2 Spread Mispricing
        st.subheader("3.2 Spread Mispricing (3M Gap)")
        if not historical_spreads_df.empty:
            try:
                comparison_spread, mispricing_spread = calculate_snapshot_comparison(historical_spreads_df, fair_curve_prices, analysis_dt, is_outright=False)
                fig_spread, ax_spread = plt.subplots(figsize=(12, 5))
                mispricing_spread.mul(10000).plot(kind='bar', ax=ax_spread, 
                                                  color=np.where(mispricing_spread.mul(10000) > 0, 'green', 'red'))
                ax_spread.set_title(f"Spread Mispricing (Actual - PCA Fair) in BPS ({pc_count} Factors)")
                ax_spread.set_ylabel("Mispricing (BPS)")
                ax_spread.set_xlabel("Spread Contract (C1-C2)")
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig_spread)
                
                detailed_comparison_spread = comparison_spread.copy()
                detailed_comparison_spread['Mispricing (BPS)'] = mispricing_spread * 10000
                st.dataframe(
                    detailed_comparison_spread.style.format({'Original': "{:.4f}", 'PCA Fair': "{:.4f}", 'Mispricing (BPS)': "{:.2f}"}),
                    use_container_width=True
                )
            except KeyError:
                st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for Spreads.")
        else:
            st.info("Not enough contracts (need 2 or more) to calculate spreads.")
            
        # 3.3 Butterfly Mispricing
        st.subheader("3.3 Butterfly Mispricing (3M Gap)")
        if not historical_butterflies_df.empty:
            try:
                comparison_fly, mispricing_fly = calculate_snapshot_comparison(historical_butterflies_df, fair_curve_prices, analysis_dt, is_outright=False)
                fig_fly, ax_fly = plt.subplots(figsize=(12, 5))
                mispricing_fly.mul(10000).plot(kind='bar', ax=ax_fly, 
                                               color=np.where(mispricing_fly.mul(10000) > 0, 'green', 'red'))
                ax_fly.set_title(f"Butterfly Mispricing (Actual - PCA Fair) in BPS ({pc_count} Factors)")
                ax_fly.set_ylabel("Mispricing (BPS)")
                ax_fly.set_xlabel("Butterfly Contract (C1-2xC2+C3)")
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig_fly)
                
                detailed_comparison_fly = comparison_fly.copy()
                detailed_comparison_fly['Mispricing (BPS)'] = mispricing_fly * 10000
                st.dataframe(
                    detailed_comparison_fly.style.format({'Original': "{:.4f}", 'PCA Fair': "{:.4f}", 'Mispricing (BPS)': "{:.2f}"}),
                    use_container_width=True
                )
            except KeyError:
                st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for Butterflies.")
        else:
            st.info("Not enough contracts (need 3 or more) to calculate butterflies.")


        # --- 4. Advanced Hedging Analysis ---
        st.header("4. Advanced Hedging Analysis")
        st.markdown("Select a trade to determine the optimal hedge. Results are based on the **Factor Sensitivities** derived from the top K PCs.")

        # Calculate all factor sensitivities once
        factor_sensitivities = calculate_factor_sensitivities(all_instruments_df, loadings)
        
        # UI for Trade Selection
        all_instruments = factor_sensitivities.index.tolist()
        trade_instrument = st.selectbox("Select Your Trade Instrument (T)", all_instruments)
        trade_direction = st.radio("Trade Direction of T", ['Buy (Long 1 unit)', 'Sell (Short 1 unit)'])
        is_long = (trade_direction == 'Buy (Long 1 unit)')
        
        
        st.subheader("4.1 Factor Sensitivity (Betas) of Selected Trade")
        st.dataframe(
            factor_sensitivities.loc[[trade_instrument]].T.style.format({f'PC{i+1}': "{:.4f}" for i in range(pc_count)}), 
            use_container_width=True
        )


        # --- 4.2 Minimum Variance Hedging (MVH) ---
        st.subheader("4.2 Minimum Variance Hedging (MVH)")
        st.markdown(f"Optimal hedge for **{trade_instrument}** to minimize overall volatility (factor risk).")
        
        mvhr_results = calculate_mvhr(trade_instrument, factor_sensitivities, scores, pc_count)
        best_hedge = mvhr_results.iloc[0]

        if not mvhr_results.empty:
            
            st.markdown(f"#### Best Hedge Recommendation (Risk Reduction: **{best_hedge['Risk Reduction (%)']:.2f}%**)")
            
            h_star = best_hedge['MVHR (Units of Hedge/Unit of Trade)']
            
            # Determine Final Action (Long/Short) based on trade direction and MVHR sign
            if is_long:
                # Long trade: hedge (H) takes opposite sign of MVHR
                action = "Short" if h_star > 0 else "Long"
            else: # is_short
                # Short trade: hedge (H) takes same sign as MVHR
                action = "Long" if h_star > 0 else "Short"
            
            st.success(f"To hedge 1 unit of **{trade_instrument}**, take a position of **{action} {abs(h_star):.2f} units** of the **{best_hedge['Hedge Candidate']}**.")
            st.info(f"The hedged portfolio volatility is reduced from **{best_hedge['Trade Volatility (BPS)']:.2f} BPS** to **{best_hedge['Residual Volatility (BPS)']:.2f} BPS**.")
            
            with st.expander("Show All Hedge Candidates Ranked by Risk Reduction"):
                st.dataframe(
                    mvhr_results.style.format({
                        'Trade Volatility (BPS)': "{:.2f}",
                        'MVHR (Units of Hedge/Unit of Trade)': "{:.4f}",
                        'Residual Volatility (BPS)': "{:.2f}",
                        'Risk Reduction (%)': "{:.2f}"
                    }),
                    use_container_width=True
                )
        
        # --- 4.3 Scenario-Based Hedging (PC1/PC2 Neutral) ---
        if pc_count >= 2:
            st.subheader("4.3 Scenario-Based Hedging (PC1/PC2 Neutral)")
            st.markdown("Creates a hedge that is perfectly neutral to a **Parallel Shift (PC1 - Level)** and **Steepening/Flattening (PC2 - Slope)**.")

            # Select two hedge candidates for the 2x2 system
            hedge_candidates_2x2 = [i for i in all_instruments if i != trade_instrument]
            
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                # Find the index of the best MVH candidate to use as a default
                best_hedge_name = best_hedge['Hedge Candidate'] if not mvhr_results.empty else hedge_candidates_2x2[0]
                default_h1_idx = hedge_candidates_2x2.index(best_hedge_name) if best_hedge_name in hedge_candidates_2x2 else 0
                hedge1 = st.selectbox("Select Hedge Candidate 1 (H1)", hedge_candidates_2x2, index=default_h1_idx)
            
            with col_h2:
                # Ensure H2 is not the same as H1
                default_h2_idx = (default_h1_idx + 1) % len(hedge_candidates_2x2)
                if len(hedge_candidates_2x2) > 1 and hedge1 == hedge_candidates_2x2[default_h2_idx]:
                    default_h2_idx = (default_h2_idx + 1) % len(hedge_candidates_2x2)
                
                # Ensure index is within bounds for the selectbox
                default_h2_idx = min(default_h2_idx, len(hedge_candidates_2x2) - 1) if len(hedge_candidates_2x2) > 0 else 0
                
                hedge2 = st.selectbox("Select Hedge Candidate 2 (H2)", hedge_candidates_2x2, index=default_h2_idx)

            if hedge1 != hedge2:
                h1, h2, residual_pc3 = solve_scenario_hedge(trade_instrument, hedge1, hedge2, factor_sensitivities)
                
                if h1 is not None:
                    
                    st.markdown("##### Required Hedge Weights (Hedge 1 & 2 per 1 Unit of Trade)")

                    # Determine Final Action (Long/Short) for H1 and H2 based on trade direction
                    
                    # For a Long Trade: Portfolio is T - h1*H1 - h2*H2. Weights h1, h2 are required to zero out betas.
                    # The action taken in H1/H2 is short if h > 0, long if h < 0.
                    # For a Short Trade: Portfolio is -T + h1*H1 + h2*H2. The weights are mathematically h1, h2 but the sign of the trade is reversed.
                    # To maintain the factor neutrality relative to the market move, the signs of h1 and h2 must be flipped
                    
                    # Simplified logic: The calculated h1, h2 are for the Long position (T).
                    # If trade is Long: Action is opposite sign of h
                    # If trade is Short: Action is same sign of h (reverse of the reversal)
                    
                    if is_long:
                        action_h1 = "Short" if h1 > 0 else "Long"
                        action_h2 = "Short" if h2 > 0 else "Long"
                    else:
                        action_h1 = "Long" if h1 > 0 else "Short"
                        action_h2 = "Long" if h2 > 0 else "Short"

                    
                    st.dataframe(
                        pd.DataFrame({
                            'Hedge Instrument': [hedge1, hedge2],
                            'Required Units': [f"{abs(h1):.4f}", f"{abs(h2):.4f}"],
                            'Action': [action_h1, action_h2]
                        }, index=['H1', 'H2']),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    st.info(f"The resulting portfolio is neutral to PC1 (Level) and PC2 (Slope). The residual exposure to PC3 (Curve) is **{residual_pc3:.4f}**.")
                    
                    if not is_long:
                        st.warning("Note: Since your trade direction is **Short**, the hedge actions shown maintain the factor neutrality. Mathematically, the weights were calculated for the **Long** trade, but the actions have been **reversed** to match the Short position's factor exposures.")

                
                else:
                    st.warning("Could not solve the hedge system. This typically happens if the two hedge candidates are too correlated (e.g., adjacent outright contracts) or PC1/PC2 factors are not available.")
            else:
                st.error("Please select two different hedge candidates.")
            
        else:
            st.info("Requires at least 2 Principal Components to calculate Level/Slope Neutrality.")


    else:
        st.error("PCA failed. Please upload both Historical Price Data and Expiry Data and check your PCA parameters and date range.")

else:
    st.info("Please upload both Historical Price Data and Expiry Data files to begin the analysis.")
