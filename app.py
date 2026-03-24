import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

from data import fetch_data, calculate_returns, calculate_portfolio_returns, NIFTY_50_TICKERS
from scipy.stats import norm

def calculate_historical_var(returns, confidence_level=0.95):
    """
    Historical Value at Risk (VaR).
    Calculates the percentile of actual historical returns.
    """
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_parametric_var(returns, confidence_level=0.95):
    """
    Parametric (Variance-Covariance) Value at Risk.
    Assuming normal distribution of returns.
    """
    mean = np.mean(returns)
    std_dev = np.std(returns)
    # The normal percent point function
    return norm.ppf(1 - confidence_level, loc=mean, scale=std_dev)

def simulate_monte_carlo_var(returns, confidence_level=0.95, num_simulations=10000, days=1):
    """
    Monte Carlo Simulation for Value at Risk (1-day).
    Simulates thousands of price/return paths assuming normality.
    For more complex distributions, historical covariance would be used.
    Here we simulate returns directly using the historical mean and std dev.
    """
    if len(returns) == 0:
        return np.nan
        
    mean = np.mean(returns)
    std_dev = np.std(returns)
    
    simulated_returns = np.random.normal(mean, std_dev, num_simulations)
    
    return np.percentile(simulated_returns, 100 * (1 - confidence_level))

def calculate_cvar(returns, confidence_level=0.95):
    """
    Conditional Value at Risk (CVaR) or Expected Shortfall.
    Provides the expected loss given that the loss exceeds the VaR threshold.
    """
    var_threshold = calculate_historical_var(returns, confidence_level)
    tail_losses = returns[returns <= var_threshold]
    
    if len(tail_losses) == 0:
        return np.nan
        
    return tail_losses.mean()

def run_stress_test(portfolio_returns, initial_investment, returns=None, weights=None, scenarios=None):
    """
    Provide a more comprehensive stress testing suite.
    1. Historical Drawdowns & Worst Periods (from the actual data)
    2. Portfolio-Specific Idiosyncratic Shocks
    """
    historical_results = {}
    
    if isinstance(portfolio_returns, pd.Series) and not portfolio_returns.empty:
        worst_day_ret = portfolio_returns.min()
        worst_day_date = portfolio_returns.idxmin().strftime('%Y-%m-%d')
        historical_results['Historical Worst Day'] = (worst_day_ret * initial_investment, worst_day_date)
        
        if len(portfolio_returns) >= 5:
            rolling_5 = (portfolio_returns + 1).rolling(5).apply(np.prod, raw=True) - 1
            worst_week_ret = rolling_5.min()
            worst_week_date = rolling_5.idxmin().strftime('%Y-%m-%d')
            historical_results['Worst Week (5-day)'] = (worst_week_ret * initial_investment, worst_week_date)
            
        if len(portfolio_returns) >= 21:
            rolling_21 = (portfolio_returns + 1).rolling(21).apply(np.prod, raw=True) - 1
            worst_month_ret = rolling_21.min()
            worst_month_date = rolling_21.idxmin().strftime('%Y-%m-%d')
            historical_results['Worst Month (21-day)'] = (worst_month_ret * initial_investment, worst_month_date)
            
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        mdd_date = drawdown.idxmin().strftime('%Y-%m-%d')
        historical_results['Max Drawdown'] = (max_drawdown * initial_investment, mdd_date)

    hypothetical_results = {}
    if returns is not None and weights is not None and len(weights) == len(returns.columns):
        tickers = returns.columns
        weighted_assets = sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)
        top_assets = weighted_assets[:min(3, len(weighted_assets))]
        
        for ticker, weight in top_assets:
            scenario_name = f"Isolation Shock: {ticker} drops 20%"
            impact = initial_investment * weight * -0.20
            hypothetical_results[scenario_name] = (impact, "N/A")
            
        worst_days = returns.min()
        extreme_crash_impact = np.sum(worst_days.values * weights) * initial_investment
        hypothetical_results['Concurrent Asset Worst Days'] = (extreme_crash_impact, "Various")
    else:
        if scenarios is None:
            scenarios = {
                'Black Monday (1987) Shock': -0.226,
                'COVID-19 Crash (Daily)': -0.12,
                'Global Fin. Crisis (Daily)': -0.09,
                'Bull Market (+10%)': 0.10
            }
        
        for scenario_name, shock_pct in scenarios.items():
            impact = initial_investment * shock_pct
            hypothetical_results[scenario_name] = (impact, "N/A")
            
    return historical_results, hypothetical_results

st.set_page_config(page_title="Market Risk Analysis", layout="wide")

st.title("📈 Market Risk Analysis Dashboard")
st.markdown("Analyze VaR, CVaR, Monte Carlo simulations and conduct Stress Testing for NIFTY 50 portfolios.")

st.sidebar.header("Portfolio Configuration")
selected_tickers = st.sidebar.multiselect(
    "Select NIFTY 50 Assets", 
    NIFTY_50_TICKERS, 
    default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
)

if not selected_tickers:
    st.warning("Please select at least one asset to proceed.")
    st.stop()

st.sidebar.subheader("Weights")
weights_input = []
for ticker in selected_tickers:
    w = st.sidebar.number_input(
        f"{ticker} Weight", 
        value=1.0/len(selected_tickers), 
        min_value=0.0, 
        max_value=1.0, 
        step=0.05
    )
    weights_input.append(w)

# Normalize weights
weights = np.array(weights_input)
if np.sum(weights) == 0:
    st.sidebar.error("Weights cannot sum to zero.")
    st.stop()
weights = weights / np.sum(weights)

initial_investment = st.sidebar.number_input("Initial Investment (INR)", value=1000000, step=50000)
confidence_level = st.sidebar.slider("Confidence Level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)

years_map = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
time_horizon_str = st.sidebar.selectbox("Historical Data Range", list(years_map.keys()), index=2)
start_date = datetime.date.today() - datetime.timedelta(days=years_map[time_horizon_str])

st.sidebar.markdown("---")
if st.sidebar.button("Calculate Risk Metrics"):
    with st.spinner("Fetching data and computing risk models..."):
        price_data = fetch_data(selected_tickers, start_date=start_date)
        if price_data.empty:
            st.error("Failed to fetch data. Please check your internet connection or the selected tickers.")
            st.stop()
            
        returns = calculate_returns(price_data)
        port_returns = calculate_portfolio_returns(returns, weights)
        
        # Risk Computations
        hist_var = calculate_historical_var(port_returns, confidence_level)
        param_var = calculate_parametric_var(port_returns, confidence_level)
        cvar = calculate_cvar(port_returns, confidence_level)
        mc_var = simulate_monte_carlo_var(port_returns, confidence_level, num_simulations=10000)
        
        # UI Presentation
        st.subheader("1. Risk Summary Report (Daily Horizon)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Historical VaR", f"{hist_var*100:.2f}%", f"₹{hist_var * initial_investment:,.2f}", delta_color="inverse")
        col2.metric("Parametric VaR", f"{param_var*100:.2f}%", f"₹{param_var * initial_investment:,.2f}", delta_color="inverse")
        col3.metric("Monte Carlo VaR", f"{mc_var*100:.2f}%", f"₹{mc_var * initial_investment:,.2f}", delta_color="inverse")
        col4.metric("CVaR (Expected Shortfall)", f"{cvar*100:.2f}%", f"₹{cvar * initial_investment:,.2f}", delta_color="inverse")
        
        st.markdown(f"*At a {confidence_level*100}% confidence level, the maximum expected daily loss is represented by the VaR. The CVaR represents the average loss in the worst {(1-confidence_level)*100:.0f}% of cases.*")

        st.markdown("---")
        st.subheader("2. Return Distributions & Risk Thresholds")
        
        # Main Hist
        fig = px.histogram(port_returns, nbins=60, title="Distribution of Portfolio Returns", opacity=0.7)
        fig.add_vline(x=hist_var, line_dash="dash", line_color="red", annotation_text=f"Hist VaR: {hist_var*100:.2f}%")
        fig.add_vline(x=param_var, line_dash="dot", line_color="orange", annotation_text=f"Param VaR: {param_var*100:.2f}%")
        fig.add_vline(x=cvar, line_dash="dashdot", line_color="black", annotation_text=f"CVaR: {cvar*100:.2f}%")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("3. Monte Carlo Simulation (10,000 Paths)")
        
        mean_ret = np.mean(port_returns)
        std_ret = np.std(port_returns)
        simulated_returns = np.random.normal(mean_ret, std_ret, 10000)
        
        fig_mc = px.histogram(simulated_returns, nbins=60, title="Monte Carlo Simulated Returns", opacity=0.7, color_discrete_sequence=['green'])
        fig_mc.add_vline(x=mc_var, line_dash="dash", line_color="red", annotation_text=f"MC VaR: {mc_var*100:.2f}%")
        st.plotly_chart(fig_mc, use_container_width=True)

        st.markdown("---")
        st.subheader("4. Portfolio-Specific Stress Testing")
        st.write("Evaluate portfolio performance under historical worst-case data and asset-specific idiosyncratic shocks.")
        
        hist_results, hypo_results = run_stress_test(port_returns, initial_investment, returns=returns, weights=weights)
        
        col_hist, col_hypo = st.columns(2)
        
        with col_hist:
            st.markdown("**Data-Driven Historical Worst Cases**")
            if hist_results:
                hist_df = pd.DataFrame({
                    "Scenario": list(hist_results.keys()), 
                    "Impact (INR)": [v[0] for v in hist_results.values()],
                    "Date": [v[1] for v in hist_results.values()]
                })
                fig_hist = px.bar(
                    hist_df, x="Impact (INR)", y="Scenario", orientation='h', 
                    title="Historical Stress Impacts",
                    labels={'Impact (INR)': 'Impact', 'Scenario': ''},
                    color="Impact (INR)", color_continuous_scale=[(0, "red"), (1, "gray")]
                )
                fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_hist, use_container_width=True)
                
                hist_df['Impact (INR)'] = hist_df['Impact (INR)'].apply(lambda x: f"₹{x:,.2f}")
                st.dataframe(hist_df, hide_index=True, use_container_width=True)
            else:
                st.info("Not enough data to compute historical worst cases.")

        with col_hypo:
            st.markdown("**Asset-Specific & Idiosyncratic Shocks**")
            hypo_df = pd.DataFrame({
                "Scenario": list(hypo_results.keys()), 
                "Impact (INR)": [v[0] for v in hypo_results.values()],
                "Date": [v[1] for v in hypo_results.values()]
            })
            fig_hypo = px.bar(
                hypo_df, x="Impact (INR)", y="Scenario", orientation='h', 
                title="Idiosyncratic Shock Impacts",
                labels={'Impact (INR)': 'Impact', 'Scenario': ''},
                color="Impact (INR)", color_continuous_scale=[(0, "red"), (0.5, "gray"), (1, "green")]
            )
            fig_hypo.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_hypo, use_container_width=True)
            
            hypo_df['Impact (INR)'] = hypo_df['Impact (INR)'].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(hypo_df, hide_index=True, use_container_width=True)
else:
    st.info("Configure your portfolio on the sidebar and click 'Calculate Risk Metrics'.")
