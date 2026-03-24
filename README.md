# Market Risk Analysis 📈

An end-to-end Market Risk Analysis application built with Python, Streamlit, and Jupyter Notebooks. It calculates Historical Value at Risk (VaR), Parametric VaR, Conditional VaR (Expected Shortfall), and executes a robust Monte Carlo simulation (for 100,000 paths) on custom NIFTY 50 portfolios.

## Key Features
- **Live Data Ingestion**: Dynamically fetches adjusted historical closing data via `yfinance`.
- **Value at Risk (VaR)**: Fully models Parametric and Historical VaR thresholds using `scipy.stats`.
- **Monte Carlo Simulator**: Simulates 100,000 distribution paths to calculate Monte Carlo VaR seamlessly.
- **Historical Stress Testing**: Measures portfolio impact under theoretical negative shocks (e.g. 1987 Black Monday, COVID Crash, Global Financial Crisis).
- **Interactive Streamlit Dashboard**: Provides an interactive risk UI (`app.py`) for on-the-fly metric calculation and scenario analysis.
- **Self-Contained Notebook**: A fully self-contained standalone Jupyter visualizer (`Market_Risk_Analysis.ipynb`) featuring elegant Seaborn and Matplotlib histograms.

## Installation
Ensure you have Python 3 installed. Then, run:
```bash
pip install -r requirements.txt
```

## Running the Dashboard
```bash
streamlit run app.py
```

## License
MIT
