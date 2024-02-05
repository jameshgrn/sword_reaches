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

def plot_lambda(data_dict, max_gamma=1000, max_superelevation=30, frac=.2, ci=90):
    # Determine the number of rows needed based on the number of names and a max of 4 columns
    num_plots = len(data_dict)
    num_columns = min(num_plots, 4)
    num_rows = (num_plots + num_columns - 1) // num_columns  # Ceiling division to get number of rows needed

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(3*num_columns, 3*num_rows), squeeze=False)
    for name, details in data_dict.items():
        ax = axs.flatten()[list(data_dict.keys()).index(name)]
        print(f'Processing {name}...')
        df = process_data(f'data/{name}_output.csv', max_gamma=max_gamma, max_superelevation=max_superelevation)
        if name == 'V7':
            df = df[df['dist_out'] > 2080699]
        df = df[df['lambda'] > .1]
        df = df[df['lambda'] < 300]
        # Convert 'dist_out' from meters to kilometers
        df['dist_out'] = df['dist_out'] / 1000

        # Compute the LOWESS smoothed curve for lambda
        smoothed_lambda = lowess(df['lambda'], df['dist_out'], frac=frac)
        #ax.hlines(y=2, xmin=df['dist_out'].min(), xmax=df['dist_out'].max(), color='black', linestyle='--', lw=1.5, zorder=1)

        ax.set_xlabel('Distance along reach (km)')
        ax.set_ylabel('Lambda')
        ax.set_yscale('log')
        ax.invert_xaxis()  # Reverse the x-axis
        ax.set_title(name)  # Set the title of the plot to the name

        # Fill the area between the start and end of the avulsion belt across the entire y-axis
        ax.fill_betweenx(y=[0, 1], x1=details['avulsion_belt'][0], x2=details['avulsion_belt'][1], color='gray', alpha=0.3, transform=ax.get_xaxis_transform())

        # Plot the avulsion_dist_out as vertical black dashed lines behind the data
        for dist_out in details.get('avulsion_lines', []):
            ax.axvline(x=dist_out, color='k', linestyle='--', zorder=1)

        # Plot the crevasse_splay_dist_out as vertical dark blue dotted lines behind the data
        for dist_out in details.get('crevasse_splay_lines', []):
            ax.axvline(x=dist_out, color='blue', linestyle=':', zorder=1)

        # Ensure scatter plot and LOWESS curve are plotted above the vertical lines
        sns.scatterplot(data=df, x='dist_out', y='lambda', color='#26C6DA', marker='^', edgecolor='k', s=65, ax=ax, zorder=2)
        ax.plot(smoothed_lambda[:, 0], smoothed_lambda[:, 1], 'r-', zorder=2)
        

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# Example usage with a dictionary pre-filled with your data:
data_dict = {
    "B14": {
        "avulsion_lines": [3644.767, 3640.772, 3621.538, 3630.944, 3596.765, 3582.564, 3607.758],  # Converted from avulsion_distances
        "crevasse_splay_lines": [3647.977, 3639.155],  # Converted from crevasse_splay_distances
        "avulsion_belt": (3499.215, 3641.982)  # Converted from the range provided
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
    "MAHAJAMBA": {
        "avulsion_lines": [109.403],  # Converted from avulsion_distances
        "crevasse_splay_lines": [],  # No crevasse splay distances for MAHAJAMBA
        "avulsion_belt": (136.591, 105.403)  # Converted from the range provided
    },
    "ARG_LAKE": {
        "avulsion_lines": [235.852, 204.908, 190.422, 170.924],  # Converted from avulsion_distances
        "crevasse_splay_lines": [],  # Converted from crevasse_splay_distances
        "avulsion_belt": (255.346, 89.952)  # Example avulsion belt range
    },
    "V7": {
        "avulsion_lines": [2101.933, 2106.922],  # Converted from avulsion_distances
        "crevasse_splay_lines": [],  # Converted from crevasse_splay_distances
        "avulsion_belt": (2127.714, 2083.699)  # Example avulsion belt range
    },
    "V11": {
        "avulsion_lines": [1869.058, 1865.705],  # Converted from avulsion_distances
        "crevasse_splay_lines": [1888.197],  # Converted from crevasse_splay_distances (SMALL SPLAY)
        "avulsion_belt": (1872.683, 1860.060)  # Example avulsion belt range
    },
}

plot_lambda(data_dict)

# extra
# "SAMBAO" 48558

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

def plot_binscatter(data_dict, max_gamma=1000, max_superelevation=30):
    num_plots = len(data_dict)
    num_columns = min(num_plots, 4)
    num_rows = (num_plots + num_columns - 1) // num_columns
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(3*num_columns, 3*num_rows), squeeze=False)
    
    for idx, (name, details) in enumerate(data_dict.items()):
        ax = axs.flatten()[idx]
        print(f'Processing {name}...')
        df = process_data(f'data/{name}_output.csv', max_gamma=max_gamma, max_superelevation=max_superelevation)
        df = df[df['lambda'] > 0]
        df = df[df['lambda'] < 1000]
        # Convert 'dist_out' from meters to kilometers
        df['dist_out'] = df['dist_out'] / 1000
        print(name, len(df))
        df_est = binscatter(x='dist_out', y='lambda', data=df, ci=(4,4), noplot=True)
        
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
        ax.set_ylim(.01, 400)  # Set limits in log scale
        ax.set_yscale('log')
        ax.invert_xaxis()  # Reverse the x-axis
        ax.set_title(name)  # Set the title of the plot to the name

        # Add horizontal grid lines, light and dashed
        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
        ax.yaxis.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')

        # Set y tick labels to non-scientific notation
        ax.set_yticks([0.01, 0.1, 1, 10, 100], minor=False)
        ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])

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

def extract_and_analyze_lambda_df(data_dict, max_gamma=1000, max_superelevation=30):
    all_lambda_values = []  # To store lambda values across all names for overall mean calculation
    results_list = []  # To store intermediate results for DataFrame conversion

    for name, details in data_dict.items():
        print(f'Processing {name}...')
        # Process data and perform bin scatter analysis
        df = process_data(f'data/{name}_output.csv', max_gamma=max_gamma, max_superelevation=max_superelevation)
        df = df[df['lambda'] > 0]
        df = df[df['lambda'] < 1000]
        df['dist_out'] = df['dist_out'] / 1000  # Convert 'dist_out' from meters to kilometers

        name_lambda_values = []  # To store lambda values for the current name

        # Process avulsion lines
        for line in details.get('avulsion_lines', []):
            closest_dists = df.iloc[(df['dist_out'] - line).abs().argsort()[:10]]
            lambda_values = closest_dists['lambda'].values
            for lambda_value in lambda_values:
                name_lambda_values.append(lambda_value)
                all_lambda_values.append(lambda_value)
                results_list.append({"Name": name, "Line Type": "Avulsion", "Distance": line, "Lambda Value": lambda_value})

        # Process crevasse splay lines
        for line in details.get('crevasse_splay_lines', []):
            closest_dists = df.iloc[(df['dist_out'] - line).abs().argsort()[:10]]
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
    df = process_data(f'data/{name}_output.csv', max_gamma=500, max_superelevation=100)
    df = df[(df['lambda'] > 0) & (df['lambda'] < 1000)]
    df['dist_out'] = df['dist_out'] / 1000  # Convert 'dist_out' from meters to kilometers
    df_est = binscatter(x='dist_out', y='lambda', data=df, ci=(3,3), noplot=True)
    
    # Concatenate the current df and df_est to the large DataFrames
    large_df = pd.concat([large_df, df], ignore_index=True)
    large_df_est = pd.concat([large_df_est, df_est], ignore_index=True)


# %%
import pandas as pd
import numpy as np

def bootstrap_distribution(source_series, target_length, n_iterations=1000):
    """
    Generate a bootstrapped distribution of a pandas Series to a specified length.
    
    Parameters:
    - source_series: pandas Series from which to bootstrap.
    - target_length: The desired length of the bootstrapped sample.
    - n_iterations: Number of bootstrap iterations to perform.
    
    Returns:
    - bootstrapped_distribution: Array of bootstrapped sample means.
    """
    bootstrapped_means = []
    for _ in range(n_iterations):
        # Sample with replacement to the length of the target DataFrame
        sample = source_series.sample(n=target_length, replace=True)
        # Calculate the mean of the sample and append to the list
        bootstrapped_means.append(sample.mean())
    
    return np.array(bootstrapped_means)

# Assuming results_df and large_df are already defined
source_series = results_df['Mean Lambda per Name'].apply(np.log)
target_length = len(results_df['Mean Lambda per Name'])

# Perform bootstrapping
bootstrapped_distribution = bootstrap_distribution(source_series, target_length)

# Example usage: Display the mean and standard deviation of the bootstrapped distribution
print(f"Bootstrapped Distribution Mean: {bootstrapped_distribution.mean()}")
print(f"Bootstrapped Distribution Std Dev: {bootstrapped_distribution.std()}")

# If you want to visualize the distribution

# %%
#%%
sns.histplot(data=np.log(large_df['lambda']), kde=True)
sns.histplot(data=(bootstrapped_distribution), kde=True)
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Assuming large_df['lambda'] is your original large dataset
# And bootstrapped_distribution is the array containing your bootstrap results

plt.figure(figsize=(8, 5))

y1 = results_df['Mean Lambda per Name'].apply(np.log)
y = bootstrapped_distribution

# Plot the original large data histogram using seaborn
main_plot = sns.histplot(np.log(large_df['lambda']), kde=True, color="#26C6DA", label="Original")
sns.histplot(y, kde=False, color="#880E4F", alpha=1, label="Bootstrap")
sns.histplot(y1, kde=False, color="#F48FB1", alpha=1, label="Collocated")

# Add horizontal grid lines to the main plot for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Create an inset with a zoomed-in view of the bootstrap distribution
# The arguments are [left, bottom, width, height] in figure coordinates (0 to 1)
# Making the inset slightly bigger as requested
# Adjust the position of the inset graph
ax_inset = inset_axes(plt.gca(), width="35%", height="35%", loc='upper left', 
                      bbox_to_anchor=(0.1, 0.05, .8, .9), bbox_transform=plt.gca().transAxes)
sns.histplot(y, kde=False, color="#880E4F", alpha=1, ax=ax_inset)

# Add horizontal grid lines to the inset for better readability
ax_inset.grid(axis='y', linestyle='--', alpha=0.7)

# Set the limits for the inset x-axis to zoom in on the bootstrap distribution
ax_inset.set_xlim([y.min(), y.max()])

# Add labels and title for the main plot
# Adjust label placements to avoid overrunning text
ax_inset.set_xlabel(r'Log($\Lambda$)', labelpad=5)
ax_inset.set_ylabel('Count', labelpad=5)
main_plot.axvline(x=np.log(2), color='black', linestyle='--', lw=2)
main_plot.text(np.log(2)-.55, plt.gca().get_ylim()[1]/2, r'$\Lambda$ = 2', rotation=90, verticalalignment='center')
main_plot.set_xlabel(r'Log($\Lambda$)', labelpad=5)
main_plot.set_ylabel('Count', labelpad=5)
main_plot.legend(loc='upper right')
plt.savefig('/Users/jakegearon/CursorProjects/RORA_followup/lambda_bootstrap.png', dpi=300)
plt.show()


# %%
from scipy import stats

log_original = np.log(large_df['lambda'])
# Assuming log_original and log_bootstrapped are your log-transformed datasets
U_statistic, p_value = stats.mannwhitneyu(log_original, y, alternative='two-sided')

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
ks_statistic, p_value = stats.ks_2samp(log_original, y1)

print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("The two distributions are significantly different.")
else:
    print("The two distributions are not significantly different.")


#%%
import numpy as np
from scipy.stats import mannwhitneyu, norm

# Assuming log_original and log_bootstrapped are numpy arrays
combined = np.concatenate([log_original, y])

# Permutation test
num_permutations = 10000
perm_diffs = []
for _ in range(num_permutations):
    np.random.shuffle(combined)
    perm_log_original = combined[:len(log_original)]
    perm_log_bootstrapped = combined[len(log_original):]
    perm_diff = np.mean(perm_log_original) - np.mean(perm_log_bootstrapped)
    perm_diffs.append(perm_diff)

# Calculate p-value
obs_diff = np.mean(log_original) - np.mean(log_bootstrapped)
p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
print(f"P-value from permutation test: {p_value}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(log_original) - 1) * np.var(log_original) + (len(log_bootstrapped) - 1) * np.var(log_bootstrapped)) / (len(log_original) + len(log_bootstrapped) - 2))
effect_size = obs_diff / pooled_std
print(f"Effect size (Cohen's d): {effect_size}")

# Bootstrap confidence interval for effect size
bootstrapped_effect_sizes = []
for _ in range(10000):
    boot_log_original = np.random.choice(log_original, size=len(log_original), replace=True)
    boot_log_bootstrapped = np.random.choice(log_bootstrapped, size=len(log_bootstrapped), replace=True)
    boot_diff = np.mean(boot_log_original) - np.mean(boot_log_bootstrapped)
    boot_effect_size = boot_diff / pooled_std
    bootstrapped_effect_sizes.append(boot_effect_size)

conf_int = np.percentile(bootstrapped_effect_sizes, [2.5, 97.5])
print(f"95% CI for effect size: {conf_int}")

# %%
