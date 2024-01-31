#%%
import pandas as pd
import matplotlib.pyplot as plt
from utils import compute_along_track_distance
import geopandas as gpd
import matplotlib
matplotlib.use('Qt5Agg')
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

def plot_lambda(data_dict, max_gamma=1000, max_superelevation=30, frac=.3):
    # Determine the number of rows needed based on the number of names and a max of 4 columns
    num_plots = len(data_dict)
    num_columns = min(num_plots, 4)
    num_rows = (num_plots + num_columns - 1) // num_columns  # Ceiling division to get number of rows needed

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(3*num_columns, 3*num_rows), squeeze=False)
    for name, details in data_dict.items():
        ax = axs.flatten()[list(data_dict.keys()).index(name)]
        print(f'Processing {name}...')
        df = process_data(f'data/{name}_output.csv', max_gamma=max_gamma, max_superelevation=max_superelevation)

        # Convert 'dist_out' from meters to kilometers
        df['dist_out'] = df['dist_out'] / 1000

        # Compute the LOWESS smoothed curve for lambda
        smoothed_lambda = lowess(df['lambda'], df['dist_out'], frac=frac)

        # Plotting the scatterplot and the LOWESS smoothed curve
        sns.scatterplot(data=df, x='dist_out', y='lambda', color='#26C6DA', marker='^', edgecolor='k', s=100, ax=ax)
        ax.plot(smoothed_lambda[:, 0], smoothed_lambda[:, 1], 'r-')
        ax.set_xlabel('Distance along reach (km)')
        ax.set_ylabel('Lambda')
        ax.set_yscale('log')
        ax.invert_xaxis()  # Reverse the x-axis
        ax.set_title(name)  # Set the title of the plot to the name

        # Plot the avulsion_dist_out as vertical black dashed lines
        for dist_out in details['vertical_lines']:
            ax.axvline(x=dist_out, color='k', linestyle='--')

        # Fill the area between the start and end of the avulsion belt
        ax.fill_betweenx(ax.get_ylim(), details['avulsion_belt'][0], details['avulsion_belt'][1], color='gray', alpha=0.5)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# Example usage with a dictionary pre-filled with your data:
data_dict = {
    "B14": {
        "vertical_lines": [3621.538],  # Converted from avulsion_distances
        "avulsion_belt": (3499.215, 3641.982)  # Converted from the range provided
    },
    "B1": {
        "vertical_lines": [4403.519],  # Converted from avulsion_distances
        "avulsion_belt": (4434.684, 4395.922)  # Converted from the range provided
    },
    "VENEZ_2023": {
        "vertical_lines": [177.700],  # Converted from avulsion_distances
        "avulsion_belt": (178.085, 147.912)  # Example avulsion belt range
    },
    "VENEZ_2023_W": {
        "vertical_lines": [444.137],  # Converted from avulsion_distances
        "avulsion_belt": (400, 450)  # Example avulsion belt range
    },
    "MAHAJAMBA": {
        "vertical_lines": [109.403],  # Converted from avulsion_distances
        "avulsion_belt": (136.591, 105.403)  # Converted from the range provided
    },
    # "V5": {
    #     "vertical_lines": [2490.295],  # Converted from avulsion_distances
    #     "avulsion_belt": (2497.331, 2485.512)  # Example avulsion belt range
    # },
    # "V7": {
    #     "vertical_lines": [2085.706],  # Converted from avulsion_distances
    #     "avulsion_belt": (2127.714, 2042.699)  # Example avulsion belt range
    # },
    "V11": {
        "vertical_lines": [1869.058, 1865.705],  # Converted from avulsion_distances
        "avulsion_belt": (1872.683, 1852.060)  # Example avulsion belt range
    },
    "RioPauto": {
        "vertical_lines": [2224.554],  # Converted from avulsion_distances
        "avulsion_belt": (2250, 2150)  # Example avulsion belt range
    }
}
    

plot_lambda(data_dict)
    # },
    # "BLACK": {
    #     "vertical_lines": [1636.147],  # Converted from avulsion_distances
    #     "avulsion_belt": (1682.694, 1580.487)  # Converted from the range provided
    # }



# extra
# "SAMBAO" 48558
# B14 range 3641982 to 3499215
# MAHAJAMBA range 136591 to 109403

# # Example usage with zipped names and avulsion distances:
# names = ["B14", "B1", "VENEZ_2023", "VENEZ_2023_W", "V5", "V7", "V11", "RioPauto", "MAHAJAMBA"]
# avulsion_distances = [3621538, 4403519, 177700, 444137, 2490295, 2085706, 1869058, 2224554, 109403]



# %%
