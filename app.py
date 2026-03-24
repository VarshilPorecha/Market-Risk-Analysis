import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

from data import fetch_data, calculate_returns, calculate_portfolio_returns, NIFTY_50_TICKERS
from risk_models import calculate_historical_var, calculate_parametric_var, simulate_monte_carlo_var, calculate_cvar, run_stress_test

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
        st.subheader("4. Historical Stress Testing")
        stress_results = run_stress_test(port_returns, initial_investment)
        
        scenarios = list(stress_results.keys())
        impacts = list(stress_results.values())
        
        stress_df = pd.DataFrame({"Scenario": scenarios, "Impact (INR)": impacts})
        
        fig_stress = px.bar(
            x=impacts, 
            y=scenarios, 
            orientation='h', 
            title="Portfolio Impact under Stress Scenarios",
            labels={'x': 'Impact (INR)', 'y': 'Scenario'},
            color=impacts,
            color_continuous_scale=[(0, "red"), (0.5, "gray"), (1, "green")]
        )
        st.plotly_chart(fig_stress, use_container_width=True)
        
        stress_df['Impact (INR)'] = stress_df['Impact (INR)'].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(stress_df, use_container_width=True)
else:
    st.info("Configure your portfolio on the sidebar and click 'Calculate Risk Metrics'.")
