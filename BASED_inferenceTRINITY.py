#%%
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multi_plotter import binscatter
import binsreg

trinity = pd.read_csv('data/TRINITY_output.csv')
dicharge_rid = {75140100351: 393.472, 75140100341: 403.076, 75140100331:404.937, 75140100321:412.443,
                75140100311: 417.56, 75140100301: 418.495, 75140100291: 418.932, 75140100281: 421.378,
                75140100271: 422.977, 75140100261: 423.044, 75140100251: 424.339, 75140100241: 425.094,
                75140100061: 426.197, 75140100051: 425.743, 75140100041: 422.382, 75140100033: 422.382,
                }

discharge_series = pd.DataFrame(dicharge_rid, index=[0]).T
discharge_series.reset_index(inplace=True)
discharge_series.columns = ['reach_id', 'discharge_uncorrected']
discharge_series['reach_id'] = discharge_series['reach_id'].astype(int)
trinity = trinity.merge(discharge_series, on='reach_id', how='left').dropna()

with open('data/inverted_discharge_params.pickle', 'rb') as f:
    params = pickle.load(f)
def inverse_power_law(y, a, b):
    return (y / a) ** (1 / b)
trinity['corrected_discharge'] = inverse_power_law(trinity['discharge_uncorrected'], *params)

guesswork = trinity[['width', 'slope', 'corrected_discharge']].astype(float)
guesswork.columns = ['width', 'slope', 'discharge']
xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("data/based_us_sans_trampush_early_stopping_combat_overfitting.ubj")
trinity['XGB_depth'] = xgb_reg.predict(guesswork)
print(trinity.XGB_depth.describe())
df = trinity

# Reflecting values when only ridge1 exists and ridge2 does not
mask_only_ridge1 = (~df['ridge1_elevation'].isna()) & (df['ridge2_elevation'].isna())
df.loc[mask_only_ridge1, 'ridge2_elevation'] = df.loc[mask_only_ridge1, 'ridge1_elevation']
df.loc[mask_only_ridge1, 'floodplain2_elevation'] = df.loc[mask_only_ridge1, 'floodplain1_elevation']
df.loc[mask_only_ridge1, 'ridge2_dist_along'] = -df.loc[mask_only_ridge1, 'ridge1_dist_along']
df.loc[mask_only_ridge1, 'floodplain2_dist_to_river_center'] = -df.loc[mask_only_ridge1, 'floodplain1_dist_to_river_center']

df['slope'] = df['slope'] / 1000

#### SUPERELEVATION ####

# Calculate superelevation for ridge1 and ridge2
df['superelevation1'] = (df['ridge1_elevation'] - df['floodplain1_elevation']) / (df['ridge1_elevation'] - (df['ridge1_elevation'] - df['XGB_depth']))
df['superelevation2'] = (df['ridge2_elevation'] - df['floodplain2_elevation']) / (df['ridge2_elevation'] - (df['ridge2_elevation'] - df['XGB_depth']))

# Average the superelevation values
df['superelevation_mean'] = (df['superelevation1'] + df['superelevation2']) / 2

# Calculate slope for ridge1
ridge1_slope = df.apply(lambda row: (row['ridge1_elevation'] - row['floodplain1_elevation']) / abs(row['ridge1_dist_along']), axis=1)
df['ridge1_slope'] = ridge1_slope

# Since ridge2 data is mirrored from ridge1 when missing, we use the same calculation for ridge2_slope
df['ridge2_slope'] = df.apply(lambda row: (row['ridge2_elevation'] - row['floodplain2_elevation']) / abs(row['ridge2_dist_along']), axis=1)

    # Calculate ridge1_height and ridge2_height
df['ridge1_height'] = df['ridge1_elevation'] - df['floodplain1_elevation']
df['ridge2_height'] = df['ridge2_elevation'] - df['floodplain2_elevation']

# Calculate ridge_height_mean
df['ridge_height_mean'] = (df['ridge1_height'] + df['ridge2_height']) / 2

# Calculate ridge_slope_mean
df['ridge_slope_mean'] = (df['ridge1_slope'] + df['ridge2_slope']) / 2

df['ridge_width'] = df['floodplain1_dist_to_river_center'] + df['floodplain2_dist_to_river_center']


# Calculate gamma values
df['gamma1'] = np.abs(df['ridge1_slope']) / df['slope']
df['gamma2'] = np.abs(df['ridge2_slope']) / df['slope']

# Calculate mean gamma
df['gamma_mean'] = df[['gamma1', 'gamma2']].mean(axis=1, skipna=True)

# Computing theta
df['lambda'] = df['gamma_mean'] * df['superelevation_mean']

df = df[df['lambda'] > 0.0001]
df = df[df['lambda'] < 1000]

df.to_csv('data/TRINITY_output_corrected.csv', index=False)


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
sns.scatterplot(x='dist_out', y='lambda', data=df_est, s=180, color='#26C6DA', alpha=0.7, edgecolor='black', marker='D', zorder=0)

# Use plt.errorbar to plot the error bars, passing the errors array directly
plt.errorbar(df_est['dist_out'], df_est['lambda'], yerr=errors, fmt='none', ecolor='k', elinewidth=1, capsize=3, alpha=0.5)

plt.xlabel('Distance along reach (km)')
plt.ylabel(r'$\Lambda$', rotation=0, labelpad=5)
plt.ylim(.01, 400)  # Set limits in log scale
plt.yscale('log')
plt.gca().invert_xaxis()  # Reverse the x-axis

# Add horizontal grid lines, light and dashed
plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey', axis='y')

# Set y tick labels to non-scientific notation
plt.yticks([0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])

# Add minor tick marks
plt.minorticks_on()
plt.show()
# sns.scatterplot(x='dist_out', y='lambda', data=df, s=180, color='#26C6DA', alpha=0.7, edgecolor='black', marker='D', zorder=0)
# plt.yscale('log')
# plt.show()

# %%
