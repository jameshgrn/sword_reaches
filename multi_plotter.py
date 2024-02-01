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
        if name == 'V7':
            df = df[df['dist_out'] > 2080699]
        df = df[df['lambda'] > .1]
        # Convert 'dist_out' from meters to kilometers
        df['dist_out'] = df['dist_out'] / 1000
        df['lambda_error'] = df['lambda'] * 0.2

        # Compute the LOWESS smoothed curve for lambda
        smoothed_lambda = lowess(df['lambda'], df['dist_out'], frac=frac)

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
        "avulsion_lines": [3621.538, 3630.944, 3596.765, 3582.564],  # Converted from avulsion_distances
        "crevasse_splay_lines": [],  # Converted from crevasse_splay_distances
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

# %%
