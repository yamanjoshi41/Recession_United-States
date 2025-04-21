US Recession Indicator Analysis (2000–2024)

This project analyzes quarterly US recession data (JHDUSRGDPBR) from 2000 to 2024, where 1 indicates a recession and 0 indicates no recession. The analysis includes descriptive statistics, time-series trends, Markov transition probabilities, and simulated correlations with economic indicators.

📊 Key Findings

1. Recession Patterns

Frequency: 3 major recessions were identified (2001, 2008, 2020).
Duration:
2001: 181 days
2008: 548 days (Great Recession)
2020: 91 days (COVID-19)
Time in Recession: 12.12% of the observed period.
Recovery Period: Average of 3,424 days between recessions.

2. Statistical Insights

Persistence: Strong autocorrelation (0.72) between consecutive quarters, indicating recessions tend to persist.
Seasonality: Recessions were most frequent in Q1 and Q2 (4 occurrences each).
Markov Transitions:
Non-recession → Recession: 52% probability
Recession → Non-recession: 73% probability


🔍 Next Steps

Integrate Real Economic Data: Replace simulated data with actual GDP, unemployment, and stock market indices for robust correlation analysis.
Leading Indicator Analysis: Explore advanced time-series models (e.g., ARIMA, VAR) to predict recessions using external economic factors.
Expand Timeframe: Extend analysis to include earlier recessions (e.g., 1990s) for broader insights.

Data Source:

Primary Dataset: JHDUSRGDPBR.csv (Federal Reserve Economic Data (FRED)).
