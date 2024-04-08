#%%
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np


# Load river discharge data
discharge_data = pd.read_csv('river_discharge_data.csv')
#'TAMBOPATA', 
names = ['ARG_LAKE', 'B1', 'B14', 'BERMARIVO', 'MALAWI_2014', 'RIOPIRAI', 'RUVU', 'SULENGGUOLE', 'V7', 'V11', 'VENEZ_2023', 'VENEZ_2022_N', 'VENEZ_2023_W']

# Load model and parameters
xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("data/based_us_sans_trampush_early_stopping_combat_overfitting.ubj")

with open('data/inverted_discharge_params.pickle', 'rb') as f:
    params = pickle.load(f)

def inverse_power_law(y, a, b):
    return (y / a) ** (1 / b)

all_rivers_processed = []  # List to hold processed data for each river

for name in names:
    # Filter discharge data for the current river
    river_discharge_data = discharge_data[discharge_data['river_name'] == name]
    
    # Load river output data
    river_output = pd.read_csv(f'data/{name}_output.csv')
    river_output['reach_id'] = river_output['reach_id'].astype(int)
    river_discharge_data['reach_id'] = river_discharge_data['reach_id'].astype(int)
    
    # Merge discharge data
    river_output = river_output.merge(river_discharge_data[['reach_id', 'discharge_value']], on='reach_id', how='left')#.dropna()
    river_output.rename(columns={'discharge_value': 'discharge_uncorrected'}, inplace=True)
    
    # Correct discharge
    river_output['corrected_discharge'] = inverse_power_law(river_output['discharge_uncorrected'], *params)
    river_output['slope'] = river_output['slope'] / 1000
    
    # Prepare data for XGB prediction
    guesswork = river_output[['width', 'slope', 'corrected_discharge']].astype(float)
    guesswork.columns = ['width', 'slope', 'discharge']
    
    # Predict depth
    river_output['XGB_depth'] = xgb_reg.predict(guesswork)
    river_output['XGB_depth'] = river_output['XGB_depth'].clip(lower=0)
    
    # Add river name to the DataFrame
    river_output['river_name'] = name
    
    # Reflecting values when only ridge1 exists and ridge2 does not
    mask_only_ridge1 = (~river_output['ridge1_elevation'].isna()) & (river_output['ridge2_elevation'].isna())
    river_output.loc[mask_only_ridge1, 'ridge2_elevation'] = river_output.loc[mask_only_ridge1, 'ridge1_elevation']
    river_output.loc[mask_only_ridge1, 'floodplain2_elevation'] = river_output.loc[mask_only_ridge1, 'floodplain1_elevation']
    river_output.loc[mask_only_ridge1, 'ridge2_dist_along'] = -river_output.loc[mask_only_ridge1, 'ridge1_dist_along']
    river_output.loc[mask_only_ridge1, 'floodplain2_dist_to_river_center'] = -river_output.loc[mask_only_ridge1, 'floodplain1_dist_to_river_center']

    # Calculate slope for ridge1
    ridge1_slope = river_output.apply(lambda row: (row['ridge1_elevation'] - row['floodplain1_elevation']) / abs(row['ridge1_dist_along']), axis=1)
    river_output['ridge1_slope'] = ridge1_slope

    # Since ridge2 data is mirrored from ridge1 when missing, we use the same calculation for ridge2_slope
    river_output['ridge2_slope'] = river_output.apply(lambda row: (row['ridge2_elevation'] - row['floodplain2_elevation']) / abs(row['ridge2_dist_along']), axis=1)

        # Calculate ridge1_height and ridge2_height
    river_output['ridge1_height'] = river_output['ridge1_elevation'] - river_output['floodplain1_elevation']
    river_output['ridge2_height'] = river_output['ridge2_elevation'] - river_output['floodplain2_elevation']

    # Calculate ridge_height_mean
    river_output['ridge_height_mean'] = (river_output['ridge1_height'] + river_output['ridge2_height']) / 2

    # Calculate ridge_slope_mean
    river_output['ridge_slope_mean'] = (river_output['ridge1_slope'] + river_output['ridge2_slope']) / 2

    river_output['ridge_width'] = river_output['floodplain1_dist_to_river_center'] + river_output['floodplain2_dist_to_river_center']

    # Calculate gamma values
    river_output['gamma1'] = np.abs(river_output['ridge1_slope']) / river_output['slope']
    river_output['gamma2'] = np.abs(river_output['ridge2_slope']) / river_output['slope']

    # Calculate mean gamma
    river_output['gamma_mean'] = river_output[['gamma1', 'gamma2']].mean(axis=1, skipna=True)

    river_output['a_b_1'] = (river_output['ridge1_elevation'] - river_output['channel_elevation']) / (river_output['XGB_depth'])
    river_output['a_b_2'] = (river_output['ridge2_elevation'] - river_output['channel_elevation']) / (river_output['XGB_depth'])

    river_output['a_b'] = (river_output['a_b_1'] + river_output['a_b_2']) / 2

    # Define conditions
    conditions = [
        river_output['a_b'] <= 4,
        # (river_output['a_b'] > 2) & (river_output['a_b'] <= 2.5),
        river_output['a_b'] > 4
    ]

    # Define choices based on conditions
    choices = [
        river_output['XGB_depth'],  # If a_b is < 1, then depth is equal to XGB_depth
        # (river_output['ridge1_elevation'] - (river_output['channel_elevation'] - river_output['XGB_depth'])) + (river_output['ridge2_elevation'] - (river_output['channel_elevation'] - river_output['XGB_depth'])) / 2,  # If 1 <= a_b < 1.5
        (river_output['ridge1_elevation'] - (river_output['channel_elevation'])) + (river_output['ridge2_elevation'] - (river_output['channel_elevation'])) / 2  # If a_b >= 1.5
    ]

    # Apply conditions and choices to calculate corrected_depth
    river_output['corrected_denominator'] = river_output['XGB_depth']#np.select(conditions, choices)

    river_output['superelevation1'] = (river_output['ridge1_elevation'] - river_output['floodplain1_elevation']) / (river_output['corrected_denominator'])
    river_output['superelevation2'] = (river_output['ridge2_elevation'] - river_output['floodplain2_elevation']) / (river_output['corrected_denominator'])
    river_output['superelevation_mean'] = (river_output['superelevation1'] + river_output['superelevation2']) / 2

    river_output['lambda'] = river_output['gamma_mean'] * river_output['superelevation_mean']
    print(river_output['lambda'].describe())
    river_output = river_output[river_output['lambda'] > 0]

    # Append the processed DataFrame to the list
    all_rivers_processed.append(river_output)

# Concatenate all DataFrames in the list into one
all_rivers_river_output = pd.concat(all_rivers_processed, ignore_index=True)

# Save the concatenated DataFrame to a Parquet file
all_rivers_river_output.to_parquet('data/all_rivers_processed.parquet', index=False)

#%%
all_rivers_river_output.columns

#%%

