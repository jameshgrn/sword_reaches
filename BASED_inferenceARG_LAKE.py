#%%
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multi_plotter import binscatter
import binsreg

ARG_LAKE = pd.read_csv('data/ARG_LAKE_output.csv')
dicharge_rid = {64401700031: 356.79, 64401700021: 356.902, 64401700011: 356.912,
                64401700381: 307.915, 64401700371: 307.915, 64401700361: 306.053,
                64401700351: 287.324, 64401700341: 286.084, 64401700331: 278.659,
                64401700321: 278.761, 64401700311: 279.066, 64401700301: 279.869,
                64401700291: 280.126, 64401700281: 225.854, 64401700271: 214.09,
                64401700261: 190.353, 64401700251: 177.73, 64401700241: 164.483,
                64401500231: 162.024, 64401500221: 155.101, 64401500211: 218.953,
                64401500201: 222.599, 64401500191: 217.146, 64401500181: 208.431,
                64401500171: 200.566, 64401500161: 197.624, 64401500151: 195.972,
                64401500141: 195.972, 64401500121: 199.102, 64401500111: 209.107,
                64401500101: 208.545, 64401500091: 205.174, 64401500081: 201.995,
                64401500071: 198.728, 64401500431: 194.082, 64401500421: 191.084,
                64401400401: 189.456, 64401500391: 187.702, 64401500481: 188.639}

discharge_series = pd.DataFrame(dicharge_rid, index=[0]).T
discharge_series.reset_index(inplace=True)
discharge_series.columns = ['reach_id', 'discharge_uncorrected']
discharge_series['reach_id'] = discharge_series['reach_id'].astype(int)
ARG_LAKE = ARG_LAKE.merge(discharge_series, on='reach_id', how='left').dropna()

with open('data/inverted_discharge_params.pickle', 'rb') as f:
    params = pickle.load(f)
def inverse_power_law(y, a, b):
    return (y / a) ** (1 / b)
ARG_LAKE['corrected_discharge'] = inverse_power_law(ARG_LAKE['discharge_uncorrected'], *params)
ARG_LAKE['slope'] = ARG_LAKE['slope'] / 1000

guesswork = ARG_LAKE[['width', 'slope', 'corrected_discharge']].astype(float)
guesswork.columns = ['width', 'slope', 'discharge']
xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("data/based_us_sans_trampush_early_stopping_combat_overfitting.ubj")
ARG_LAKE['XGB_depth'] = xgb_reg.predict(guesswork)
ARG_LAKE['XGB_depth'] = ARG_LAKE['XGB_depth'].clip(lower=0)
df = ARG_LAKE

# Reflecting values when only ridge1 exists and ridge2 does not
mask_only_ridge1 = (~df['ridge1_elevation'].isna()) & (df['ridge2_elevation'].isna())
df.loc[mask_only_ridge1, 'ridge2_elevation'] = df.loc[mask_only_ridge1, 'ridge1_elevation']
df.loc[mask_only_ridge1, 'floodplain2_elevation'] = df.loc[mask_only_ridge1, 'floodplain1_elevation']
df.loc[mask_only_ridge1, 'ridge2_dist_along'] = -df.loc[mask_only_ridge1, 'ridge1_dist_along']
df.loc[mask_only_ridge1, 'floodplain2_dist_to_river_center'] = -df.loc[mask_only_ridge1, 'floodplain1_dist_to_river_center']


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
df['a_b_1'] = (df['ridge1_elevation'] - df['channel_elevation']) / (df['XGB_depth'])
df['a_b_2'] = (df['ridge2_elevation'] - df['channel_elevation']) / (df['XGB_depth'])

df['a_b'] = (df['a_b_1'] + df['a_b_2']) / 2

# Define conditions
conditions = [
    df['a_b'] <= 5,
    # (df['a_b'] > 2) & (df['a_b'] <= 2.5),
    df['a_b'] > 5
]

# Define choices based on conditions
choices = [
    df['XGB_depth'],  # If a_b is < 1, then depth is equal to XGB_depth
    # (df['ridge1_elevation'] - (df['channel_elevation'] - df['XGB_depth'])) + (df['ridge2_elevation'] - (df['channel_elevation'] - df['XGB_depth'])) / 2,  # If 1 <= a_b < 1.5
    (df['ridge1_elevation'] - (df['channel_elevation'])) + (df['ridge2_elevation'] - (df['channel_elevation'])) / 2  # If a_b >= 1.5
]

# Apply conditions and choices to calculate corrected_depth
df['corrected_denominator'] = np.select(conditions, choices)

df['superelevation1'] = (df['ridge1_elevation'] - df['floodplain1_elevation']) / (df['corrected_denominator'])
df['superelevation2'] = (df['ridge2_elevation'] - df['floodplain2_elevation']) / (df['corrected_denominator'])
df['superelevation_mean'] = (df['superelevation1'] + df['superelevation2']) / 2

df['lambda'] = df['gamma_mean'] * df['superelevation_mean']

df = df[df['lambda'] > 0]
print(df['lambda'].describe())

df.to_csv('data/ARG_LAKE_output_corrected.csv', index=False)

df_est = binscatter(x='dist_out', y='lambda', data=df, ci=(3,3), noplot=True)

min_threshold = 0.0001

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
