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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_parquet('data/all_elevations_gdf_B1.parquet')
plt.figure(figsize=(8, 6))
sns.set_context('paper', font_scale = 1.5)
plotdf = df.reset_index().query('cross_id == 500')
sns.lineplot(data=plotdf, x='dist_along', y='elevation', color='black', lw=1.5)
sns.scatterplot(data=plotdf, x='dist_along', y='elevation', marker='+', color='black', s=80)
#plt.ylim(181.5, 188)
plt.grid(axis='y', linestyle='--')
plt.grid(axis='x', visible=False)
plt.xlabel('Distance along cross section (m)')
plt.ylabel('Elevation (m)')
plt.savefig('/Users/jakegearon/CursorProjects/RORA_followup/fig3B1_cross_section.png', dpi=300)

# %%
name = "V11"
df = process_data(f'data/{name}_output.csv', max_gamma=max_gamma, max_superelevation=max_superelevation)

sns.set_context('paper', font_scale = 1.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
df['dist_out'] = df['dist_out'] / 1000
df = df[df['gamma_mean'] > 0.1]
df = df[df['gamma_mean'] < 300]

df = df[df['superelevation_mean'] > 0.01]
df = df[df['superelevation_mean'] < 10]

# Plot superelevation_mean on the first plot
ax1.scatter(df['dist_out'], df['superelevation_mean'], edgecolor='k', color='gray', s=80, alpha=0.7, marker='o')
ax1.set_xlabel('Distance from outlet (m)')
ax1.set_ylabel(r'$\beta$', rotation=0, labelpad=15)
ax1.invert_xaxis()  # Reverse the x-axis

# Plot gamma_mean on the second plot
ax2.scatter(df['dist_out'], df['gamma_mean'], color = 'pink', edgecolor='k', s=80, alpha=0.7, marker='o')
ax2.set_xlabel('Distance from outlet (km)')
ax2.set_ylabel(r'$\gamma$', rotation=0, labelpad=15)
ax2.invert_xaxis()  # Reverse the x-axis
plt.tight_layout()
plt.show()

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from format_funcs import process_data

def plot_with_regplot_and_bins_enhanced(data_dict, max_gamma=1000, max_superelevation=30):
    # Determine the number of rows and columns for the subplots based on the number of datasets
    num_plots = len(data_dict)
    num_columns = min(num_plots, 4)
    num_rows = (num_plots + num_columns - 1) // num_columns

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(4*num_columns, 4*num_rows), squeeze=False)
    axs = axs.flatten()  # Flatten the array to easily iterate over it

    for index, (name, details) in enumerate(data_dict.items()):
        print(f'Processing {name}...')
        df = process_data(f'data/{name}_output.csv', max_gamma=max_gamma, max_superelevation=max_superelevation)
        if name == 'V7':
            df = df[df['dist_out'] > 2080699]
        df['dist_out'] = df['dist_out'] / 1000  # Convert 'dist_out' from meters to kilometers
        
        df = df[(df['gamma_mean'] > 0.1) & (df['gamma_mean'] < 500)]
        df = df[(df['superelevation_mean'] > 0.01) & (df['superelevation_mean'] < 40)]
        smoothed_lambda = lowess(df['lambda'], df['dist_out'], frac=frac)

        ax = axs[index]
        sns.regplot(data=df, x='dist_out', y='lambda', scatter_kws={'s': 100, 'color': '#283593', 'edgecolor': 'black'}, marker='D', fit_reg=False, x_bins=len(df)//10, ax=ax, line_kws={'capsize': 6, 'lw': 1, 'zorder': 0}, x_estimator=np.median,)
        ax.invert_xaxis()  # Reverse the x-axis
        ax.set_title(name)
        ax.set_xlabel('Distance along reach (km)')
        ax.set_yscale('log')
        
        ax.set_ylabel('Lambda')

        # Plot the avulsion and crevasse splay lines
        for dist_out in details.get('avulsion_lines', []):
            ax.axvline(x=dist_out, color='k', linestyle='--', zorder=1, lw=2.5)
        for dist_out in details.get('crevasse_splay_lines', []):
            ax.axvline(x=dist_out, color='blue', linestyle=':', zorder=1, lw=2.5)
            
        ax.plot(smoothed_lambda[:, 0], smoothed_lambda[:, 1], 'r-', zorder=2, lw=3.5)


    # Adjust layout to prevent overlap and hide unused subplots
    for ax in axs[num_plots:]:
        ax.set_visible(False)  # Hide unused subplots
    plt.tight_layout()
    plt.show()

# Example usage
plot_with_regplot_and_bins_enhanced(data_dict)
# %%
