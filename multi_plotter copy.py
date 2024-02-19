#%%
import pandas as pd
import matplotlib.pyplot as plt
from utils import compute_along_track_distance
import geopandas as gpd
import matplotlib
#matplotlib.use('Qt5Agg')
from sqlalchemy import create_engine
from utils import compute_along_track_distance
import warnings
warnings.filterwarnings('ignore')
import os
from shapely import wkb
import numpy as np
import seaborn as sns
from format_funcs import process_data
from statsmodels.nonparametric.smoothers_lowess import lowess

sns.set_context('paper', font_scale = 1.1)
sns.set_style('whitegrid')


# Example usage with a dictionary pre-filled with your data:
data_dict = {
    "B14": {
        "avulsion_lines": [3644.767, 3640.772, 3621.538, 3630.944, 3596.765, 3582.564, 3607.758],  # Converted from avulsion_distances
        "crevasse_splay_lines": [3647.977, 3639.155],  # Converted from crevasse_splay_distances
        "avulsion_belt": (3499.215, 3641.982),  # Converted from the range provided

    },
    "B1": {
        "avulsion_lines": [4403.519],  # Converted from avulsion_distances
        "crevasse_splay_lines": [],  # No crevasse splay distances for B1
        "avulsion_belt": (4434.684, 4395.922)  # Converted from the range provided
    },
    "VENEZ_2023": {
        "avulsion_lines": [],  # Converted from avulsion_distances
        "crevasse_splay_lines": [177.700, 147.912],  # Converted from crevasse_splay_distances
        "avulsion_belt": (178.085, 146.912)  # Example avulsion belt range
    },
    "VENEZ_2023_W": {
        "avulsion_lines": [],  # Converted from avulsion_distances
        "crevasse_splay_lines": [444.137, 475.626],  # Converted from crevasse_splay_distances
        "avulsion_belt": (480, 440)  # Example avulsion belt range
    },
    "VENEZ_2022_N": {
        "avulsion_lines": [18.604],  # Converted from avulsion_distances
        "crevasse_splay_lines": [8.969, 6.5],  # No crevasse splay distances for MAHAJAMBA
        "avulsion_belt": (18, 8)  # Converted from the range provided
    },
    "ARG_LAKE": {
        "avulsion_lines": [235.852, 204.908, 190.422, 170.924, 59.082],  # Converted from avulsion_distances
        "crevasse_splay_lines": [],  # Converted from crevasse_splay_distances
        "avulsion_belt": (255.346, 89.952)  # Example avulsion belt range
    },
    "SULENGGUOLE": {
        "avulsion_lines": [139.661, 125.459, 94.885, 93.486, 64.169, 49.354, 4.008],  # Converted from avulsion_distances
        "crevasse_splay_lines": [148.089, 136.663, 133.673, 118.248, 68.395],  # Converted from crevasse_splay_distances
        "avulsion_belt": (189.890, .2)  # Example avulsion belt range
    },
    # "V7": {
    #     "avulsion_lines": [2101.933, 2106.922],  # Converted from avulsion_distances
    #     "crevasse_splay_lines": [],  # Converted from crevasse_splay_distances
    #     "avulsion_belt": (2127.714, 2083.699)  # Example avulsion belt range
    # },
    "V11": {
        "avulsion_lines": [1869.058, 1865.705],  # Converted from avulsion_distances
        "crevasse_splay_lines": [1888.197],  # Converted from crevasse_splay_distances (SMALL SPLAY)
        "avulsion_belt": (1872.683, 1860.060)  # Example avulsion belt range
    },
    # "TRINITY": {
    #     "avulsion_lines": [50.000],  # Converted from avulsion_distances
    #     "crevasse_splay_lines": [],  # Converted from crevasse_splay_distances
    #     "avulsion_belt": (55.000, 45.000)  # Example avulsion belt range
    # },
}


#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from format_funcs import process_data
import binsreg
import numpy as np  # Ensure numpy is imported for numerical operations
sns.set_context('paper', font_scale = 1.5)
sns.set_style('white')
plt.ioff()  # Ensure interactive mode is off

def binscatter(**kwargs):
    # Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    # Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(kwargs['data'][kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est

def plot_binscatter(data_dict, max_gamma=1000, max_superelevation=500):
    num_plots = len(data_dict)
    num_columns = min(num_plots, 4)
    num_rows = (num_plots + num_columns - 1) // num_columns
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(3*num_columns, 3*num_rows), squeeze=False)
    
    for idx, (name, details) in enumerate(data_dict.items()):
        ax = axs.flatten()[idx]
        print(f'Processing {name}...')
        df = pd.read_csv(f'data/{name}_output_corrected.csv')
        df['dist_out'] = df['dist_out'] / 1000  # Convert 'dist_out' from meters to kilometers
        df_est = binscatter(x='dist_out', y='lambda', data=df, ci=(3,3), noplot=True)
        
        min_threshold = 0.05

        # Ensure 'lambda' and 'ci_l' values are above the threshold before calculating error bars
        df_est['lambda'] = df_est['lambda'].clip(lower=min_threshold)
        df_est['ci_l'] = df_est['ci_l'].clip(lower=min_threshold)
        df_est['ci_r'] = df_est['ci_r'].clip(lower=min_threshold)

        # Calculate the error bars in the original scale
        df_est['error_lower'] = df_est['lambda'] - df_est['ci_l']
        df_est['error_upper'] = df_est['ci_r'] - df_est['lambda']

        # Ensure errors are positive
        df_est['error_lower'] = np.abs(df_est['error_lower'])
        df_est['error_upper'] = np.abs(df_est['error_upper'])

        # Create an array with the lower and upper error margins
        errors = np.array([df_est['error_lower'], df_est['error_upper']])

        # Plot binned scatterplot
        sns.scatterplot(x='dist_out', y='lambda', data=df_est, ax=ax, s=180, color='#26C6DA', alpha=0.7, edgecolor='black', marker='D', zorder=0)

        # Use ax.errorbar to plot the error bars, passing the errors array directly
        ax.errorbar(df_est['dist_out'], df_est['lambda'], yerr=errors, fmt='none', ecolor='k', elinewidth=1, capsize=3, alpha=0.5)

        ax.set_xlabel('Distance along reach (km)')
        ax.set_ylabel(r'$\Lambda$', rotation=0, labelpad=5)
        # ax.set_ylim(.1, 400)  # Set limits in log scale
        if name == 'ARG_LAKE':
            ax.set_yscale('log')
        # ax.set_yscale('log')
        ax.invert_xaxis()  # Reverse the x-axis
        ax.set_title(name)  # Set the title of the plot to the name

        # Add horizontal grid lines, light and dashed
        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
        ax.yaxis.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')

        # # Set y tick labels to non-scientific notation
        # ax.set_yticks([0.1, 1, 10, 100], minor=False)
        # ax.set_yticklabels(['0.1', '1', '10', '100'])

        # Add minor tick marks
        ax.minorticks_on()

        # Fill the area between the start and end of the avulsion belt across the entire y-axis
        ax.fill_betweenx(y=[0, 1], x1=details['avulsion_belt'][0], x2=details['avulsion_belt'][1], color='gray', alpha=0.3, transform=ax.get_xaxis_transform())

        # Plot the avulsion_dist_out as vertical black dashed lines behind the data
        for dist_out in details.get('avulsion_lines', []):
            ax.axvline(x=dist_out, color='k', linestyle='-.', zorder=1)

        # Plot the crevasse_splay_dist_out as vertical dark blue dotted lines behind the data
        for dist_out in details.get('crevasse_splay_lines', []):
            ax.axvline(x=dist_out, color='k', linestyle=':', zorder=1)
        
        x_start, x_end = ax.get_xlim()
        # Change the background shading color to a more complementary color than red
        ax.fill_between([x_start, x_end], y1=2, y2=ax.get_ylim()[1], alpha=0.2, color='#B0E0E6', zorder=0)  # Using PowderBlue for a softer appearance
        ax.axhline(y=2, color='gray', linestyle='--')

    plt.tight_layout()
    #plt.savefig('/Users/jakegearon/CursorProjects/RORA_followup/lambda_binscatter.png', dpi=300)
    plt.show()

# Example usage with your data_dict
plot_binscatter(data_dict)
# %%
import pandas as pd
import numpy as np

def extract_and_analyze_lambda_df(data_dict, max_gamma=1000, max_superelevation=1000):
    all_lambda_values = []  # To store lambda values across all names for overall mean calculation
    results_list = []  # To store intermediate results for DataFrame conversion

    for name, details in data_dict.items():
        print(f'Processing {name}...')
        # Process data and perform bin scatter analysis
        df = process_data(f'data/{name}_output.csv', max_gamma=max_gamma, max_superelevation=max_superelevation)
        df = df[df['lambda'] > 0.001]
        df = df[df['lambda'] < 1000]
        df['dist_out'] = df['dist_out'] / 1000  # Convert 'dist_out' from meters to kilometers

        name_lambda_values = []  # To store lambda values for the current name

        # Process avulsion lines
        for line in details.get('avulsion_lines', []):
            closest_dists = df.iloc[(df['dist_out'] - line).abs().argsort()[:9]]
            lambda_values = closest_dists['lambda'].values
            for lambda_value in lambda_values:
                name_lambda_values.append(lambda_value)
                all_lambda_values.append(lambda_value)
                results_list.append({"Name": name, "Line Type": "Avulsion", "Distance": line, "Lambda Value": lambda_value})

        # Process crevasse splay lines
        for line in details.get('crevasse_splay_lines', []):
            closest_dists = df.iloc[(df['dist_out'] - line).abs().argsort()[:9]]
            lambda_values = closest_dists['lambda'].values
            for lambda_value in lambda_values:
                name_lambda_values.append(lambda_value)
                all_lambda_values.append(lambda_value)
                results_list.append({"Name": name, "Line Type": "Crevasse Splay", "Distance": line, "Lambda Value": lambda_value})

        # Calculate mean lambda value for the current name and add to results
        if name_lambda_values:
            mean_lambda = np.mean(name_lambda_values)
            for item in results_list:
                if item["Name"] == name:
                    item["Mean Lambda per Name"] = mean_lambda

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Calculate and print the overall mean lambda value
    if all_lambda_values:
        overall_mean_lambda = np.mean(all_lambda_values)
        print(f"Overall mean lambda value: {overall_mean_lambda}")
    else:
        print("No lambda values found.")
        overall_mean_lambda = None

    return results_df, overall_mean_lambda

# Example usage with your data_dict
results_df, overall_mean_lambda = extract_and_analyze_lambda_df(data_dict)
print(results_df)
print(f"Overall Mean Lambda: {overall_mean_lambda}")
#%%
# Initialize empty DataFrames to concatenate all processed data and estimated data
large_df = pd.DataFrame()
large_df_est = pd.DataFrame()

for name, details in data_dict.items():
    df = process_data(f'data/{name}_output.csv', max_gamma=1000, max_superelevation=1000)
    df = df[(df['lambda'] > 0) & (df['lambda'] < 1000)]
    df['dist_out'] = df['dist_out'] / 1000  # Convert 'dist_out' from meters to kilometers
    df_est = binscatter(x='dist_out', y='lambda', data=df, ci=(3,3), noplot=True)
    
    # Concatenate the current df and df_est to the large DataFrames
    large_df = pd.concat([large_df, df], ignore_index=True)
    large_df = large_df.sample(len(results_df), random_state=1997, replace=True)
    large_df_est = pd.concat([large_df_est, df_est], ignore_index=True)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming large_df['lambda'] is your original large dataset
# And results_df['Lambda Value'] contains the specific lambda values of interest

plt.figure(figsize=(6, 3))

# Log-transform the lambda values for plotting
log_large_df_lambda = np.log(large_df['lambda'])
log_results_df_lambda = results_df['Lambda Value'].apply(np.log)

# Plot the histograms for the original and specific lambda values
sns.histplot(log_large_df_lambda, kde=False, color="blue", label="original", alpha=.5, edgecolor="black")
sns.histplot(log_results_df_lambda, kde=False, color="orange", alpha=.5, label="collocated", edgecolor="black")

# Calculate and plot the mean values for each dataset
mean_log_large_df_lambda = np.mean(log_large_df_lambda)
mean_log_results_df_lambda = np.mean(log_results_df_lambda)

# Add grid, labels, legend, and title
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper right')
plt.axvline(x=np.log(2), color='black', linestyle='-', lw=3)
plt.text(np.log(2)-.55, plt.gca().get_ylim()[1]/2, r'$\Lambda$ = 2', rotation=90, verticalalignment='center')
plt.xlabel(r'Log($\Lambda$)', labelpad=5)
plt.ylabel('Count', labelpad=5)
# Save and show the plot
plt.savefig('/Users/jakegearon/CursorProjects/RORA_followup/lambda_comparison.png', dpi=300)
plt.show()


# %%
from scipy import stats

log_original = np.log(large_df['lambda'])
# Assuming log_original and log_bootstrapped are your log-transformed datasets
U_statistic, p_value = stats.mannwhitneyu(log_original, log_results_df_lambda, alternative='two-sided')

print(f"Mann-Whitney U Statistic: {U_statistic}")
print(f"P-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("There is a significant difference between the two distributions.")
else:
    print("There is no significant difference between the two distributions.")
# %%
# Assuming large_df['lambda'] and bootstrapped_distribution are defined
# Log-transform both distributions since the plotting was done on the log scale


# Perform the Kolmogorov-Smirnov test
ks_statistic, p_value = stats.ks_2samp(log_original, log_results_df_lambda)

print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("The two distributions are significantly different.")
else:
    print("The two distributions are not significantly different.")





# %%
from scipy.stats import shapiro, mannwhitneyu
import numpy as np

def perform_shapiro_test(data):
    stat, p_value = shapiro(data)
    return f"Shapiro-Wilk Test: Stat={stat}, p={p_value}"

log_large_df_lambda = np.log(large_df['lambda'])  # Assuming large_df['lambda'] is defined
log_results_df_lambda = np.log(results_df['Lambda Value'])  # Assuming results_df['lambda'] is defined

print(perform_shapiro_test(log_large_df_lambda))
print(perform_shapiro_test(log_results_df_lambda))

def permutation_test(x, y, num_permutations=10000):
    combined = np.concatenate([x, y])
    obs_diff = np.mean(x) - np.mean(y)
    perm_diffs = [np.mean(np.random.permutation(combined)[:len(x)]) - np.mean(np.random.permutation(combined)[len(x):]) for _ in range(num_permutations)]
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return p_value

p_value_perm_test = permutation_test(log_large_df_lambda, log_results_df_lambda)
print(f"P-value from permutation test: {p_value_perm_test}")

def calculate_effect_size(x, y):
    pooled_std = np.sqrt(((len(x) - 1) * np.var(x) + (len(y) - 1) * np.var(y)) / (len(x) + len(y) - 2))
    effect_size = (np.mean(x) - np.mean(y)) / pooled_std
    return effect_size

effect_size = calculate_effect_size(log_large_df_lambda, log_results_df_lambda)
print(f"Effect size: {effect_size}")

def bootstrap_ci_effect_size(x, y, num_bootstraps=10000, ci=95):
    bootstrapped_effect_sizes = []
    pooled_std = np.sqrt(((len(x) - 1) * np.var(x) + (len(y) - 1) * np.var(y)) / (len(x) + len(y) - 2))
    for _ in range(num_bootstraps):
        boot_x = np.random.choice(x, size=len(x), replace=True)
        boot_y = np.random.choice(y, size=len(y), replace=True)
        boot_diff = np.mean(boot_x) - np.mean(boot_y)
        boot_effect_size = boot_diff / pooled_std
        bootstrapped_effect_sizes.append(boot_effect_size)
    conf_int = np.percentile(bootstrapped_effect_sizes, [(100-ci)/2, 100-(100-ci)/2])
    return conf_int

conf_int = bootstrap_ci_effect_size(log_large_df_lambda, log_results_df_lambda)
print(f"95% CI for effect size: {conf_int}")

U_statistic, p_value = mannwhitneyu(log_large_df_lambda, log_results_df_lambda, alternative='two-sided')
print(f"Mann-Whitney U Statistic: {U_statistic}, p-value: {p_value}")

def cohens_d(x, y):
    diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    effect_size = diff / pooled_std
    return effect_size

d = cohens_d(log_large_df_lambda, log_results_df_lambda)
print(f"Cohen's d: {d}")
# %%
def simulate_wmw_power(n1, n2, delta, tdf=5, alpha=0.05, nsim=10000):
    """
    Simulate Wilcoxon-Mann-Whitney test to estimate power.
    
    Parameters:
    - n1, n2: Sample sizes for the two groups.
    - delta: Location difference between the two groups.
    - tdf: Degrees of freedom for the t-distribution.
    - alpha: Significance level.
    - nsim: Number of simulations.
    
    Returns:
    - Estimated power of the test.
    """
    reject_count = 0
    for _ in range(nsim):
        y1 = np.random.standard_t(tdf, size=n1)
        y2 = np.random.standard_t(tdf, size=n2) + delta
        p_value = stats.mannwhitneyu(y1, y2, alternative='two-sided').pvalue
        if p_value <= alpha:
            reject_count += 1
    return reject_count / nsim

# Example usage
n1 = len(log_large_df_lambda)
n2 = len(log_results_df_lambda)
delta = np.mean(log_large_df_lambda) - np.mean(log_results_df_lambda)  # Location difference
estimated_power = simulate_wmw_power(n1, n2, delta)
print(f"Estimated power: {estimated_power}")

# %%
