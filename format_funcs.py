import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def process_data(csv_filepath, max_gamma=None, max_superelevation=None):
    df = pd.read_csv(csv_filepath, header=0, sep=',')

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

    # Filter by max_superelevation if provided
    if max_superelevation is not None:
        df = df[(df['superelevation1'] <= max_superelevation) & (df['superelevation2'] <= max_superelevation)]

    #### GAMMA ####
    # Convert parent_channel_slope from m/km to m/m
    df['slope'] = df['slope'] / 1000

    # Calculate slope for ridge1
    ridge1_slope = df.apply(lambda row: (row['ridge1_elevation'] - row['floodplain1_elevation']) / abs(row['ridge1_dist_along']), axis=1)
    df['ridge1_slope'] = ridge1_slope

    # Since ridge2 data is mirrored from ridge1 when missing, we use the same calculation for ridge2_slope
    df['ridge2_slope'] = df.apply(lambda row: (row['ridge2_elevation'] - row['floodplain2_elevation']) / abs(row['ridge2_dist_along']), axis=1)

    # Calculate gamma values
    df['gamma1'] = np.abs(df['ridge1_slope']) / df['slope']
    df['gamma2'] = np.abs(df['ridge2_slope']) / df['slope']

    # Filter by max_gamma if provided
    if max_gamma is not None:
        df = df[(df['gamma1'] <= max_gamma) & (df['gamma2'] <= max_gamma)]

    # Calculate mean gamma
    df['gamma_mean'] = df[['gamma1', 'gamma2']].mean(axis=1, skipna=True)

    # Computing theta
    df['lambda'] = df['gamma_mean'] * df['superelevation_mean']

    # Continue with the rest of the processing...
    # ...

    return df

# Example usage:
# processed_df = process_data('data/B14_output.csv')