#%%
import pandas as pd
import matplotlib.pyplot as plt
from utils import compute_along_track_distance
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib
matplotlib.use('Qt5Agg')
from sqlalchemy import create_engine
import pandas as pd
from utils import compute_along_track_distance
import warnings
warnings.filterwarnings('ignore')
import os
from shapely import wkb
import numpy as np
name = "A004"
df = pd.read_csv(f'data/{name}_output.csv', header=0, sep=',')

#df = pd.read_csv("data/B9_outputFULL2.csv", header=0, sep=',')

# Reflecting values when only ridge1 exists and ridge2 does not
mask_only_ridge1 = (~df['ridge1_elevation'].isna()) & (df['ridge2_elevation'].isna())
df.loc[mask_only_ridge1, 'ridge2_elevation'] = df.loc[mask_only_ridge1, 'ridge1_elevation']
df.loc[mask_only_ridge1, 'floodplain2_elevation'] = df.loc[mask_only_ridge1, 'floodplain1_elevation']
df.loc[mask_only_ridge1, 'ridge2_dist_along'] = -df.loc[mask_only_ridge1, 'ridge1_dist_along']
df.loc[mask_only_ridge1, 'floodplain2_dist_to_river_center'] = -df.loc[mask_only_ridge1, 'floodplain1_dist_to_river_center']

#### SUPERELEVATION ####

# Calculate superelevation for ridge1 and ridge2
df['superelevation1'] = (df['ridge1_elevation'] - df['floodplain1_elevation']) / (df['ridge1_elevation'] - df['channel_elevation'])
df['superelevation2'] = (df['ridge2_elevation'] - df['floodplain2_elevation']) / (df['ridge2_elevation'] - df['channel_elevation'])

# Average the superelevation values
df['superelevation_mean'] = (df['superelevation1'] + df['superelevation2']) / 2

# Remove rows with negative superelevation values
df = df[(df['superelevation1'] >= 0) & (df['superelevation2'] >= 0)]

#### RIDGE WIDTH ####
# Computing ridge width
ridge_width = df['floodplain2_dist_to_river_center'] + df['floodplain1_dist_to_river_center']

#### GAMMA ####
# Convert parent_channel_slope from m/km to m/m
df['channel_slope'] = df['channel_slope'] / 1000

# Calculate slope for ridge1
ridge1_slope = df.apply(lambda row: (row['ridge1_elevation'] - row['floodplain1_elevation']) / abs(row['ridge1_dist_along']), axis=1)
df['ridge1_slope'] = ridge1_slope

# Since ridge2 data is mirrored from ridge1 when missing, we use the same calculation for ridge2_slope
df['ridge2_slope'] = df.apply(lambda row: (row['ridge2_elevation'] - row['floodplain2_elevation']) / abs(row['ridge2_dist_along']), axis=1)

# Calculate gamma values
df['gamma1'] = np.abs(df['ridge1_slope']) / df['channel_slope']
df['gamma2'] = np.abs(df['ridge2_slope']) / df['channel_slope']

# Calculate mean gamma
df['gamma_mean'] = df[['gamma1', 'gamma2']].mean(axis=1, skipna=True)


# df = df[df['gamma_mean'] < 3000]
# df = df[df['gamma_mean'] > .1]

# Computing theta
df['theta'] = df['gamma_mean'] * df['superelevation_mean']
# Calculate ridge height for ridge1 and ridge2
df['ridge1_height'] = df['ridge1_elevation'] - df['floodplain1_elevation']
df['ridge2_height'] = df['ridge2_elevation'] - df['floodplain2_elevation']

# Assuming ridge_width is the total width between floodplain1 and floodplain2,
# if you need individual widths, you would need to define how to calculate them.
# For example, if you have the distance from each ridge to the river center, you could use:
df['ridge1_width'] = df['floodplain1_dist_to_river_center'] * 2  # Assuming symmetry around the river center
df['ridge2_width'] = df['floodplain2_dist_to_river_center'] * 2  # Assuming symmetry around the river center

# If the above assumption is not correct, please provide the correct method or data to calculate individual ridge widths.
#drop inf or nan theta values
df = df.replace([np.inf, -np.inf], np.nan)
#df = df.dropna(subset=['theta'])

# Drop rows with NaN values (if desired at this stage)
### three subplots

 #%%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
sns.set_context('paper', font_scale = 1.8)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 13))

# Plot superelevation_mean on the first plot
ax1.scatter(df['dist_out'], df['superelevation_mean'], edgecolor='k', s=80)
ax1.set_xlabel('Distance from outlet (m)')
ax1.set_ylabel('Superelevation Mean')
ax1.invert_xaxis()  # Reverse the x-axis

# Plot gamma_mean on the second plot
ax2.scatter(df['dist_out'], df['gamma_mean'], edgecolor='k', s=80)
ax2.set_xlabel('Distance from outlet (m)')
ax2.set_ylabel('Gamma Mean')
ax2.invert_xaxis()  # Reverse the x-axis

# Plot theta on the third plot
ax3.scatter(df['dist_out'], df['theta'], color='w', marker='^', edgecolor='k', s=100)
ax3.set_xlabel('Distance from outlet (m)')
ax3.set_ylabel('Theta')
ax3.invert_xaxis()  # Reverse the x-axis

# Add vertical lines and set yscale to log for all plots
for ax in [ax1, ax2, ax3]:
    ax.set_yscale('log')

ax1.legend(ncol=1, fontsize=10, loc='upper left')
ax2.legend(ncol=1, fontsize=10, loc='upper left')
ax3.legend(ncol=1, fontsize=10, loc='upper left')

# Compute the LOWESS smoothed curve for each plot
frac = .25

# For superelevation_mean
smoothed1 = lowess(df['superelevation_mean'], df['dist_out'], frac=frac)
ax1.plot(smoothed1[:, 0], smoothed1[:, 1], 'b-')

# For gamma_mean
smoothed2 = lowess(df['gamma_mean'], df['dist_out'], frac=frac)
ax2.plot(smoothed2[:, 0], smoothed2[:, 1], color='orange')

# For theta
smoothed3 = lowess(df['theta'], df['dist_out'], frac=frac)
ax3.plot(smoothed3[:, 0], smoothed3[:, 1], 'r-')

plt.tight_layout()
plt.show()

#%%
import seaborn as sns
sns.set_context('paper', font_scale = 1.0)
import statsmodels.api as sm

# Calculate the mean of ridge heights and widths
df['ridge_height_mean'] = df[['ridge1_height', 'ridge2_height']].mean(axis=1)
df['ridge_width_mean'] = df[['ridge1_width', 'ridge2_width']].mean(axis=1)

# Filter out non-positive values before plotting
df_plot = df[(df['ridge_height_mean'] > 0) & (df['ridge_width_mean'] > 0)].copy()

# Fit OLS to the data
X = sm.add_constant(df_plot['ridge_width_mean'])  # adding a constant
model = sm.OLS(df_plot['ridge_height_mean'], X)
results = model.fit()

# Scatter plot for mean ridge height vs mean ridge width using seaborn
plt.figure(figsize=(6, 3))
sns.scatterplot(x='ridge_width_mean', y='ridge_height_mean', hue='dist_out', data=df_plot)
plt.xlabel('Mean Ridge Width (m)')
plt.ylabel('Mean Ridge Height (m)')
plt.title('Mean Ridge Width vs Mean Ridge Height')

# Plot the OLS fit
plt.plot(df_plot['ridge_width_mean'], results.fittedvalues, color='r', lw=2, ls='--', label='OLS Fit')

plt.legend()
plt.tight_layout()
plt.show()

#%%
cross_section_stats = pd.read_parquet(f'cross_section_stats_{name}.parquet')

# Define the number of rows and columns for the subplot
n_rows = 4
n_cols = 5

# Create a figure and axes for the subplot
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 16))

# List of variables to plot
variables = ['elevation_mean', 'elevation_var', 'elevation_skew', 'elevation_kurtosis', 'elevation_median', 'elevation_std', 'slope_mean', 'slope_std', 'slope_skew', 'slope_kurtosis', 'relief', 'azimuth_range']

# Iterate over each subplot and create a scatter plot with lowess fit
for i, ax in enumerate(axs.flatten()):
    if i < len(variables):
        sns.lineplot(x='dist_out', y=variables[i], data=cross_section_stats, ax=ax)
        lowess_results = lowess(cross_section_stats[variables[i]], cross_section_stats['dist_out'], frac=0.20)
        ax.plot(lowess_results[:, 0], lowess_results[:, 1], color='red')
        ax.set_title(variables[i])
        ax.invert_xaxis()

# Remove empty subplots
for i in range(len(variables), n_rows*n_cols):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.show()

# %%
