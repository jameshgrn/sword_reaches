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

df 

# Superelevation calculations for ridges 1 and 2 and their mean.
df['superelevation1'] = (df['ridge1_elevation'] - df['floodplain1_elevation']) / (df['ridge1_elevation'] - df['channel_elevation'])
df['superelevation2'] = (df['ridge2_elevation'] - df['floodplain2_elevation']) / (df['ridge2_elevation'] - df['channel_elevation'])
df['superelevation_mean'] = (df['superelevation1'] + df['superelevation2']) / 2

# Reflecting values when only ridge1 or ridge2 exists
mask_only_ridge1 = (~df['ridge1_elevation'].isna()) & (df['ridge2_elevation'].isna())
df.loc[mask_only_ridge1, 'ridge2_elevation'] = df.loc[mask_only_ridge1, 'ridge1_elevation']
df.loc[mask_only_ridge1, 'floodplain2_elevation'] = df.loc[mask_only_ridge1, 'floodplain1_elevation']
df.loc[mask_only_ridge1, 'ridge2_along_track_distance'] = -df.loc[mask_only_ridge1, 'ridge1_along_track_distance']
df.loc[mask_only_ridge1, 'floodplain2_dist_to_river_center'] = -df.loc[mask_only_ridge1, 'floodplain1_dist_to_river_center']

# Computing ridge width
ridge_width = df['floodplain2_dist_to_river_center'] + df['floodplain1_dist_to_river_center']

# Computing lambda
# df['lambda'] = ridge_width / df['width']

# Convert parent_channel_slope from m/km to m/m
#parent_channel_slope = 0.00732885151463427 / 1000
parent_channel_slope = df['slope'] / 1000

condition_both = (~df['ridge1_elevation'].isna()) & (~df['ridge2_elevation'].isna())
condition_ridge1 = mask_only_ridge1

# Slope and gamma calculations
selected_rows1 = df[condition_ridge1 | condition_both].dropna(subset=['ridge1_along_track_distance'])
ridge1_slope = selected_rows1.apply(lambda row: (row['ridge1_elevation'] - row['floodplain1_elevation']) / abs(row['ridge1_along_track_distance'] - row['floodplain1_along_track_distance']), axis=1)

selected_rows2 = df[condition_both].dropna(subset=['ridge2_along_track_distance'])  # Not including condition_ridge2 since we've mirrored the data
ridge2_slope = selected_rows2.apply(lambda row: (row['ridge2_elevation'] - row['floodplain2_elevation']) / abs(row['ridge2_along_track_distance'] - row['floodplain2_along_track_distance']), axis=1)

df.loc[selected_rows1.index, 'ridge1_slope'] = ridge1_slope
df.loc[selected_rows2.index, 'ridge2_slope'] = ridge2_slope

df['gamma1'] = ridge1_slope / parent_channel_slope
df['gamma2'] = ridge2_slope / parent_channel_slope
df['gamma_mean'] = df[['gamma1', 'gamma2']].mean(axis=1, skipna=True)

df = df[df['gamma_mean'] < 3000]
df = df[df['gamma_mean'] > .1]

# Computing theta
df['theta'] = df['gamma_mean'] * df['superelevation_mean']

#drop inf or nan theta values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['theta'])

# Drop rows with NaN values (if desired at this stage)
### three subplots

 #%%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
sns.set_context('paper', font_scale = 1.8)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9))

# Plot superelevation_mean, gamma_mean, and lambda on the same plot
ax1.scatter(df['dist_out'], df['superelevation_mean'], edgecolor='k', s=80)
ax1.scatter(df['dist_out'], df['gamma_mean'], edgecolor='k', s=80)
#ax1.scatter(df['dist_out'], df['lambda'], edgecolor='k', s=120)
ax1.set_xlabel('Distance from outlet (m)')
ax1.set_ylabel('Value')
ax1.invert_xaxis()  # Reverse the x-axis

# Plot theta on a separate plot
ax2.scatter(df['dist_out'], df['theta'], color='w', marker='^', edgecolor='k', s=100)
ax2.set_xlabel('Distance from outlet (m)')
ax2.set_ylabel('Theta')

ax2.invert_xaxis()  # Reverse the x-axis

# Add vertical lines and set yscale to log for both plots
for ax in [ax1, ax2]:
    ax.set_yscale('log')
    ax.axvline(x = 177686.469351933, color = 'k', linestyle = '--', lw=2, label='A 2023')
    ax.axvline(x = 105199.109490599, color = 'g', linestyle = '--', lw=2, label='CS 2016')
    ax.axvline(x = 82587.345773731, color = 'b', linestyle = '--', lw=2, label='A / CS 2017')
    ax.axvline(x = 6435.7054400418, color = 'r', linestyle = '--', lw=2, label='A 1982')
    # ax.axvline(x = 44858.8270155966, color = 'k', linestyle = '--', lw=2, label='RA')
    # ax.axvline(x = 36223.4518396408, color = 'g', linestyle = '--', lw=2, label='A 2001')
    # ax.axvline(x = 28245.2441120963, color = 'b', linestyle = '--', lw=2, label='NC 2018')
ax1.legend(ncol=1, fontsize=10, loc='upper left')
ax2.legend(ncol=1, fontsize=12)


# Compute the LOWESS smoothed curve for each plot
frac = .25

# For superelevation_mean, gamma_mean, and lambda
smoothed1 = lowess(df['superelevation_mean'], df['dist_out'], frac=frac)
ax1.plot(smoothed1[:, 0], smoothed1[:, 1], 'b-')

smoothed2 = lowess(df['gamma_mean'], df['dist_out'], frac=frac)
ax1.plot(smoothed2[:, 0], smoothed2[:, 1], color='orange')

# For theta
smoothed4 = lowess(df['theta'], df['dist_out'], frac=frac)
ax2.plot(smoothed4[:, 0], smoothed4[:, 1], 'r-')

plt.tight_layout()

plt.savefig('VENEZ_figure.png', dpi=300)
plt.show()

