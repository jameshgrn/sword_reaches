import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from matplotlib.colors import LogNorm

def read_csv_with_fallback(primary_path, fallback_path):
    try:
        return pd.read_csv(primary_path)
    except FileNotFoundError:
        return pd.read_csv(fallback_path)

def calculate_average_correlation(df, min_bin_size, max_bin_size, step):
    min_bin_size_meters = min_bin_size * 1000
    max_bin_size_meters = max_bin_size * 1000
    step_meters = step * 1000
    width = df['width'].mean()
    bin_sizes_non_dim = np.arange(min_bin_size_meters / width, max_bin_size_meters / width + step_meters / width, step_meters / width)
    
    average_correlations = []
    counts = []
    for bin_size_non_dim in bin_sizes_non_dim:
        bin_size_meters = bin_size_non_dim * width
        correlations = []
        bin_count = 0
        for start_dist in np.arange(df['dist_out'].min(), df['dist_out'].max(), bin_size_meters):
            end_dist = start_dist + bin_size_meters
            segment_df = df[(df['dist_out'] >= start_dist) & (df['dist_out'] < end_dist)]
            if len(segment_df) > 1:
                correlation = segment_df['superelevation_mean'].corr(segment_df['gamma_mean'])
                if not np.isnan(correlation):
                    correlations.append(correlation)
                    bin_count += len(segment_df)
        if correlations:
            average_correlation = np.mean(correlations)
            average_correlations.append((bin_size_non_dim, average_correlation))
            counts.append(bin_count)
    return average_correlations, counts

def plot_correlations_and_counts(average_correlations, counts):
    bin_sizes_non_dim, avg_corrs = zip(*average_correlations)
    plt.scatter(bin_sizes_non_dim, avg_corrs, marker='o', s=20)
    plt.xlabel('Non-dimensional Reach Size')
    plt.ylabel('Average Correlation Coefficient')
    plt.title('Average Correlation by Non-dimensional Bin Size')
    plt.show()

def plot_custom_lags_acf_pacf(series, title, max_lag):
    #plt.figure(figsize=(10, 6))
    if 'ACF' in title:
        plot_acf(series, lags=max_lag, title=title,)
    else:
        plot_pacf(series, lags=max_lag, title=title)
    plt.xlabel('Lag')
    plt.tight_layout()
    plt.show()

# Example usage
name = 'MALAWI_2014'
river_df = read_csv_with_fallback(f'data/{name}_output_corrected.csv', f'data/{name}_output.csv')
average_correlations, counts = calculate_average_correlation(river_df, 0.1, 400, 1)
plot_correlations_and_counts(average_correlations, counts)

df_sorted = river_df.sort_values(by='dist_out')
max_lag = len(df_sorted) - 1
plot_custom_lags_acf_pacf(df_sorted['lambda'], 'Lambda ACF', max_lag)
plot_custom_lags_acf_pacf(df_sorted['lambda'], 'Lambda PACF', max_lag)

def find_last_significant_lag(series, max_lag):
    acf_values, confint = acf(series, nlags=max_lag, fft=True, alpha=0.05)
    last_significant_lag = None
    for lag in range(1, len(acf_values)):  # Skip lag 0
        if confint[lag][0] > 0 or confint[lag][1] < 0:
            last_significant_lag = lag  # This lag is significant
    return last_significant_lag

# Example usage
last_significant_lag = find_last_significant_lag(df_sorted['lambda'], max_lag)
channel_width_median = df_sorted['meand_len'].median()

if last_significant_lag is not None:
    print(f"Last significant lag outside the 95% CI: {last_significant_lag}")
    print(f"Median channel width: {channel_width_median} meters")
    
    # Calculate the actual distance corresponding to the last significant lag
    # by summing the differences in 'dist_out' up to that lag
    actual_distance = df_sorted['dist_out'].diff().iloc[1:last_significant_lag+1].sum()
    print(f"Actual distance corresponding to the last significant lag: {actual_distance:.2f} meters")
    # Calculate the number of channel widths corresponding to the actual distance
    channel_widths_at_lag = actual_distance / channel_width_median
    print(f"Last significant lag corresponds to approximately {actual_distance:.2f} meters, which is {channel_widths_at_lag:.2f} channel widths.")
else:
    print("No significant lag found outside the 95% CI.")

#%%
import seaborn as sns
df = pd.read_csv('data/MALAWI_2014_output_corrected.csv')
sns.scatterplot(x='dist_out', y='lambda', hue='width', data=df)
plt.yscale('log')

plt.show()
# %%
