import numpy as np
import pandas as pd
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
    
    # Simulate single-day returns
    # Over multi-days we would sum the simulated returns or simulate geometric Brownian motion
    simulated_returns = np.random.normal(mean, std_dev, num_simulations)
    
    # VaR is the specified quantile of the simulated distribution
    return np.percentile(simulated_returns, 100 * (1 - confidence_level))

def calculate_cvar(returns, confidence_level=0.95):
    """
    Conditional Value at Risk (CVaR) or Expected Shortfall.
    Provides the expected loss given that the loss exceeds the VaR threshold.
    """
    var_threshold = calculate_historical_var(returns, confidence_level)
    # Filter returns that are worse than the VaR threshold
    tail_losses = returns[returns <= var_threshold]
    
    if len(tail_losses) == 0:
        return np.nan
        
    return tail_losses.mean()

def run_stress_test(portfolio_returns, initial_investment, scenarios=None):
    """
    Apply extreme hypothetical scenarios to the portfolio.
    Returns the monetary impact of these shocks.
    """
    if scenarios is None:
        scenarios = {
            'Black Monday (1987) Shock': -0.226,
            'COVID-19 Crash (Daily)': -0.12,
            'Global Financial Crisis (Daily)': -0.09,
            '+10% Bull Market Rally': 0.10
        }
        
    results = {}
    for scenario_name, shock_pct in scenarios.items():
        impact = initial_investment * shock_pct
        results[scenario_name] = impact
        
    return results
