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

def plot_lambda(names, avulsion_dist_out, max_gamma=1000, max_superelevation=30, frac=.3):
    # Determine the number of rows needed based on the number of names and a max of 4 columns
    num_plots = len(names)
    num_columns = min(num_plots, 4)
    num_rows = (num_plots + num_columns - 1) // num_columns  # Ceiling division to get number of rows needed

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(3*num_columns, 3*num_rows), squeeze=False)
    for (name, dist_out), ax in zip(zip(names, avulsion_dist_out), axs.flatten()):
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

        # Plot the avulsion_dist_out as a vertical black dashed line
        ax.axvline(x=dist_out, color='k', linestyle='--')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# Example usage with zipped names and avulsion distances:
names = ["B14", "B1", "VENEZ_2023", "VENEZ_2023_W", "V5", "V7", "V11", "SAMBAO"]
avulsion_distances = [3640772, 4403519, 178286, 444137, 2490295, 2085706, 1869058, 48558]
avulsion_dist_out = [x / 1000 for x in avulsion_distances]  # Convert to kilometers
plot_lambda(names, avulsion_dist_out)

# %%
