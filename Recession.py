#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm


# In[2]:


df_recession = df_recession = pd.read_csv('JHDUSRGDPBR.csv')
display(df_recession.head())


# In[4]:


# Check data types
print(df_recession.dtypes)


# In[5]:


# Determine the time range
print(f"Time range: {df_recession['observation_date'].min()} to {df_recession['observation_date'].max()}")


# In[7]:


# Check for missing values
print(df_recession.isnull().sum())


# In[8]:


# Analyze the distribution of the recession indicator
print(df_recession['JHDUSRGDPBR'].value_counts(normalize=True))


# In[9]:


# Visualize the distribution
plt.figure(figsize=(8, 6))
df_recession['JHDUSRGDPBR'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Recession Indicator')
plt.xlabel('Recession Indicator (0 = No Recession, 1 = Recession)')
plt.ylabel('Frequency')
plt.show()


# In[10]:


# Summarize findings
print("\nSummary:")
print("The dataset spans from", df_recession['observation_date'].min(), "to", df_recession['observation_date'].max())
print("The mean value of recession indicator is", df_recession['JHDUSRGDPBR'].mean())
print("The percentage of time the economy was in recession is", df_recession['JHDUSRGDPBR'].value_counts(normalize=True)[1] * 100, "%")


# In[11]:


# Convert 'observation_date' to datetime
df_recession['observation_date'] = pd.to_datetime(df_recession['observation_date'], format='%Y-%m-%d')


# In[13]:


# Create 'quarter' column
df_recession['quarter'] = df_recession['observation_date'].dt.quarter


# In[14]:


df_recession


# In[19]:


# Identify recession start and end dates
df_recession['recession_start'] = pd.NaT
df_recession['recession_end'] = pd.NaT

in_recession = False
start_date = None

for i in range(len(df_recession)):
    if df_recession['JHDUSRGDPBR'][i] == 1 and not in_recession:
        in_recession = True
        start_date = df_recession['observation_date'][i]
    elif df_recession['JHDUSRGDPBR'][i] == 0 and in_recession:
        in_recession = False
        end_date = df_recession['observation_date'][i-1]
        df_recession.loc[(df_recession['observation_date'] >= start_date) & (df_recession['observation_date'] <= end_date), 'recession_start'] = start_date
        df_recession.loc[(df_recession['observation_date'] >= start_date) & (df_recession['observation_date'] <= end_date), 'recession_end'] = end_date
    elif i == len(df_recession)-1 and in_recession:
        in_recession = False
        end_date = df_recession['observation_date'][i]
        df_recession.loc[(df_recession['observation_date'] >= start_date) & (df_recession['observation_date'] <= end_date), 'recession_start'] = start_date
        df_recession.loc[(df_recession['observation_date'] >= start_date) & (df_recession['observation_date'] <= end_date), 'recession_end'] = end_date


# In[17]:


display(df_recession.head())


# In[20]:


# Calculate the frequency of recessions
recession_start_dates = df_recession[df_recession['recession_start'].notna()]['recession_start'].unique()
recession_frequency = len(recession_start_dates)
print(f"Recession Frequency: {recession_frequency}")


# In[21]:


# Calculate the duration of each recession
df_recession['recession_duration'] = df_recession['recession_end'] - df_recession['recession_start']
recession_durations = df_recession[df_recession['recession_duration'].notna()]['recession_duration'].unique()
print("\nRecession Durations (in days):")
print(recession_durations)


# In[22]:


# Calculate the total duration of all recessions
total_recession_duration = df_recession['recession_duration'].sum()
print(f"\nTotal Recession Duration: {total_recession_duration}")


# In[23]:


# Calculate the percentage of time in recession
total_observations = len(df_recession)
recession_observations = df_recession['JHDUSRGDPBR'].sum()
percentage_in_recession = (recession_observations / total_observations) * 100
print(f"\nPercentage of Time in Recession: {percentage_in_recession:.2f}%")


# In[24]:


# Calculate the average duration between recessions (recovery periods)
recession_end_dates = df_recession[df_recession['recession_end'].notna()]['recession_end'].unique()
recovery_periods = pd.Series(recession_end_dates[1:]).reset_index(drop=True) - pd.Series(recession_end_dates[:-1]).reset_index(drop=True)
avg_recovery_period = recovery_periods.mean()
print(f"\nAverage Recovery Period: {avg_recovery_period}")


# In[25]:


# Analyze the distribution of recessions across quarters
recession_quarters = df_recession[df_recession['recession_start'].notna()]['quarter'].value_counts()
print(f"\nRecession Distribution by Quarter:\n{recession_quarters}")


# In[26]:


# 1. Time series plot with shaded recession periods
plt.figure(figsize=(12, 6))
plt.plot(df_recession['observation_date'], df_recession['JHDUSRGDPBR'], color='blue')
for i in range(len(df_recession)):
    if pd.notna(df_recession['recession_start'][i]):
        plt.axvspan(df_recession['recession_start'][i], df_recession['recession_end'][i], color='red', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Recession Indicator')
plt.title('US Recession Periods')
plt.grid(True)
plt.show()


# In[27]:


# 2. Bar chart of recession distribution across quarters
plt.figure(figsize=(8, 6))
recession_quarters = df_recession[df_recession['recession_start'].notna()]['quarter'].value_counts()
recession_quarters.plot(kind='bar', color='skyblue')
plt.xlabel('Quarter')
plt.ylabel('Number of Recessions')
plt.title('Distribution of Recessions Across Quarters')
plt.xticks(rotation=0)
plt.show()


# In[28]:


# 3. Histogram of recession durations
plt.figure(figsize=(8, 6))
recession_durations = df_recession[df_recession['recession_duration'].notna()]['recession_duration'].dt.days
plt.hist(recession_durations, bins=5, color='salmon', edgecolor='black')
plt.xlabel('Recession Duration (Days)')
plt.ylabel('Frequency')
plt.title('Histogram of Recession Durations')
plt.show()


# In[29]:


get_ipython().system('pip install hmmlearn')

# Prepare the data for the Markov chain model
recession_sequence = df_recession['JHDUSRGDPBR'].values.reshape(-1, 1)

# Initialize and train the Markov chain model
model = hmm.MultinomialHMM(n_components=2, n_iter=100, random_state=0) # Two states: recession and no recession
model.fit(recession_sequence)

# Extract transition probabilities
transition_matrix = model.transmat_
print("Transition Matrix:")
print(transition_matrix)


# In[30]:


# Create lagged versions of the 'JHDUSRGDPBR' column
for i in range(1, 5):
    df_recession[f'JHDUSRGDPBR_lag_{i}'] = df_recession['JHDUSRGDPBR'].shift(i).fillna(0)


# In[31]:


# Calculate the correlation between the original 'JHDUSRGDPBR' and its lagged versions
correlations = {}
for i in range(1, 5):
    correlation = df_recession['JHDUSRGDPBR'].corr(df_recession[f'JHDUSRGDPBR_lag_{i}'])
    correlations[f'lag_{i}'] = correlation
print("Correlation between original 'JHDUSRGDPBR' and lagged versions:")
print(correlations)


# In[32]:


# Analyze the correlations
print("\nAnalysis:")
if correlations['lag_1'] > 0.5:
    print("There is a moderate positive correlation between the current and previous quarter's recession indicator.")
elif correlations['lag_1'] < -0.5:
    print("There is a moderate negative correlation between the current and previous quarter's recession indicator.")
else:
    print("The correlation between the current and previous quarter's recession indicator is not strong.")

for lag, corr in correlations.items():
    if abs(corr) > 0.3:
        print(f"A moderate correlation ({corr:.2f}) exists for {lag}.")

if all(abs(corr) < 0.2 for corr in correlations.values()):
    print("No strong correlations were found with any of the lags. Recessions may not exhibit predictable short-term patterns.")


# In[33]:


data_gdp = {'observation_date': pd.to_datetime(df_recession['observation_date']),
            'GDP_growth': np.random.rand(len(df_recession)) * 0.1}  # Placeholder GDP growth data
df_gdp = pd.DataFrame(data_gdp).set_index('observation_date')


# In[34]:


# Unemployment rate
data_unemployment = {'observation_date': pd.to_datetime(df_recession['observation_date']),
                     'Unemployment_rate': np.random.rand(len(df_recession)) * 0.05 + 0.04}  # Placeholder Unemployment data
df_unemployment = pd.DataFrame(data_unemployment).set_index('observation_date')


# In[35]:


# Stock market trends (e.g., S&P 500)
data_stock = {'observation_date': pd.to_datetime(df_recession['observation_date']),
              'S&P_500': np.random.rand(len(df_recession)) * 100 + 3000}  # Placeholder S&P500 data
df_stock = pd.DataFrame(data_stock).set_index('observation_date')


# In[36]:


# Merge dataframes
df_recession = df_recession.set_index('observation_date')
df_merged = pd.merge(df_recession, df_gdp, left_index=True, right_index=True, how='left')
df_merged = pd.merge(df_merged, df_unemployment, left_index=True, right_index=True, how='left')
df_merged = pd.merge(df_merged, df_stock, left_index=True, right_index=True, how='left')


# In[37]:


# Handle missing values (if any) using forward fill
df_merged.fillna(method='ffill', inplace=True)

display(df_merged.head())


# In[38]:


# Calculate the correlation between 'JHDUSRGDPBR' and each economic indicator
correlation_gdp = df_merged['JHDUSRGDPBR'].corr(df_merged['GDP_growth'])
correlation_unemployment = df_merged['JHDUSRGDPBR'].corr(df_merged['Unemployment_rate'])
correlation_sp500 = df_merged['JHDUSRGDPBR'].corr(df_merged['S&P_500'])


# In[39]:


# Store the correlations in a dictionary
correlations = {
    'GDP_growth': correlation_gdp,
    'Unemployment_rate': correlation_unemployment,
    'S&P_500': correlation_sp500
}

# Print the correlation coefficients
print("Correlation with Economic Indicators:")
for indicator, correlation in correlations.items():
    print(f"{indicator}: {correlation}")


# In[40]:


# Interpretation
print("\nInterpretation:")
for indicator, correlation in correlations.items():
    print(f"- {indicator}: {correlation:.2f}")
    if abs(correlation) > 0.3:
        if correlation > 0:
            print(f"  A moderate positive correlation suggests that when recessions increase, {indicator} tends to increase.")
        else:
            print(f"  A moderate negative correlation suggests that when recessions increase, {indicator} tends to decrease.")
    else:
        print(f"  The correlation between recession indicator and {indicator} is not strong.")


# In[41]:


# 1. Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
indicators = ['GDP_growth', 'Unemployment_rate', 'S&P_500']
colors = ['blue', 'red']
for i, indicator in enumerate(indicators):
    for recession_state, color in zip([0, 1], colors):
        axes[i].scatter(df_merged[df_merged['JHDUSRGDPBR'] == recession_state][indicator],
                        df_merged[df_merged['JHDUSRGDPBR'] == recession_state]['JHDUSRGDPBR'],
                        label=f"Recession = {recession_state}", color=color)
    axes[i].set_xlabel(indicator)
    axes[i].set_ylabel('Recession Indicator')
    axes[i].set_title(f'Recession vs. {indicator}')
    axes[i].legend()
plt.tight_layout()
plt.show()


# In[42]:


# 2. Line plots with shaded regions
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
indicators = ['GDP_growth', 'Unemployment_rate', 'S&P_500']
for i, indicator in enumerate(indicators):
  axes[i].plot(df_merged.index, df_merged[indicator], color='blue')
  for start, end in zip(df_merged['recession_start'].dropna(), df_merged['recession_end'].dropna()):
    axes[i].axvspan(start, end, color='red', alpha=0.3)
  axes[i].set_xlabel('Date')
  axes[i].set_ylabel(indicator)
  axes[i].set_title(f'{indicator} Over Time')

plt.tight_layout()
plt.show()


# In[43]:


# Identify recession periods
recession_periods = df_merged[df_merged['JHDUSRGDPBR'] == 1].groupby('recession_start')


# In[46]:


# Summarize major recession events
print("Major Recession Events:")
for start_date, group in recession_periods:
    end_date = group['recession_end'].iloc[0]
    duration = end_date - start_date
    print(f"- Start Date: {start_date.date()}")
    print(f"  End Date: {end_date.date()}")
    print(f"  Duration: {duration.days} days")
# Check for 2008 financial crisis and COVID-19 recession (approximate dates)
    if start_date.year == 2008:  # Note: this is an approximation
      print("  This period may correspond to the 2008 financial crisis.")
      print(f"  Simulated GDP Growth: {group['GDP_growth'].mean():.2f}")
      print(f"  Simulated Unemployment Rate: {group['Unemployment_rate'].mean():.2f}")
      print(f"  Simulated S&P 500: {group['S&P_500'].mean():.2f}")
    elif start_date.year == 2020: # Note: this is an approximation
      print("  This period may correspond to the COVID-19 recession.")
      print(f"  Simulated GDP Growth: {group['GDP_growth'].mean():.2f}")
      print(f"  Simulated Unemployment Rate: {group['Unemployment_rate'].mean():.2f}")
      print(f"  Simulated S&P 500: {group['S&P_500'].mean():.2f}")

