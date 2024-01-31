#%%
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
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# VENEZ_2023_W last processed index: 386

name = "V5"
n = 1  # adjust this value to skip cross sections


# Load the data
sword_data_gdf = gpd.read_parquet(f'data/all_elevations_gdf_{name}.parquet')
sword_data_gdf.crs = 'EPSG:4326'

#%%
# Organize data
def organize_data_by_reach_and_node(data_gdf):
    grouped = data_gdf.groupby(['reach_id', 'node_id'])
    # Order from upstream to downstream by using negative dist_out
    cross_sections = sorted([group[1] for group in grouped], key = lambda x: x['dist_out'].mean(),
                            reverse = True)
    return cross_sections


cross_sections = organize_data_by_reach_and_node(sword_data_gdf)


def compute_attributes(df, labeled_points):
    df.set_index('dist_along', inplace=True, drop=False)

    # Slope_r calculation
    df['slope_r'] = np.abs(df['elevation'].diff()) / df.index.to_series().diff()
    df['slope_r'].iloc[0] = np.abs(df['elevation'].iloc[1] - df['elevation'].iloc[0])

    # Curvature (second derivative)
    df['curvature'] = df['slope_r'].diff() / df.index.to_series().diff()
    df['curvature'].iloc[0] = df['curvature'].iloc[1]
    if len(df) > 2:  # Just to ensure we have more than 2 points
        df['curvature'].iloc[1] = (df['curvature'].iloc[0] + df['curvature'].iloc[2]) / 2

    # Extract attributes
    position_dependent_attrs = ['elevation', 'slope_r', 'curvature', 'dist_along', 'dist_to_river_center']
    global_attributes = ['width', 'width_var', 'sinuosity', 'max_width', 'dist_out',
                         'n_chan_mod', 'n_chan_max', 'facc', 'meand_len', 'type', 'reach_id', 'node_id', 'slope']

    attr_values = {}

    # Extract position-dependent attributes for labeled points
    for label, positions in labeled_points.items():
        for idx, (x, _) in enumerate(positions):
            closest_position = pd.Series(df.index - x).abs().values.argmin()
            for attr in position_dependent_attrs:
                key_name = f"{label}_{idx + 1}_{attr}" if len(positions) > 1 else f"{label}_{attr}"
                attr_values[key_name] = df.iloc[closest_position][attr]

    # Extract global attributes including 'slope'
    first_row = df.iloc[0]
    for attr in global_attributes:
        attr_values[attr] = first_row[attr]
    print(attr_values)

    return attr_values


def get_empty_dataframe_with_columns():
    position_dependent_attrs = ['elevation', 'slope_r', 'curvature', 'dist_along', 'dist_to_river_center']
    global_attributes = ['width', 'width_var', 'sinuosity', 'max_width', 'dist_out', 'n_chan_mod', 'n_chan_max', 'facc',
                         'meand_len', 'type', 'reach_id', 'node_id', 'slope']
    labels = ['channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2']

    columns = []
    for label in labels:
        for attr in position_dependent_attrs:
            columns.append(f"{label}_{attr}")
    columns.extend(global_attributes)
    return pd.DataFrame(columns = columns)


def update_and_save_to_csv(df, labeled_points, filename=f"{name}_output.csv"):
    # Start with an empty DataFrame with all the required columns
    attr_df = get_empty_dataframe_with_columns()
    
    if any(v for v in labeled_points.values()):  # Check if labeled_points is not empty
        # Compute the attributes as normal if labeled_points are provided
        attributes = compute_attributes(df, labeled_points)
        
        # Update the values in attr_df for the columns corresponding to the selected labels
        for key in attr_df.columns:  # Ensure consistent column order
            attr_df.at[0, key] = attributes.get(key, np.nan)  # Use np.nan for missing data
    
    # Check if the file exists, if not, create one with a header
    if not os.path.exists(filename):
        attr_df.to_csv(filename, index=False)
    else:
        # Append to the existing CSV
        with open(filename, 'a', newline='') as f:  # Open the file in append mode
            attr_df.to_csv(f, header=False, index=False)


def compute_distance_to_river_center(df, labeled_points):
    if 'channel' in labeled_points and labeled_points['channel']:
        channel_point_x = labeled_points['channel'][0][0]  # Assuming one channel point per cross-section
        df['dist_to_river_center'] = (df['dist_along'] - channel_point_x).abs()
    else:
        df['dist_to_river_center'] = np.nan  # If no channel point is labeled, set distance to NaN
    return df

def get_last_processed_index(filename="last_processed.txt"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return int(f.read().strip())
    return 0



def save_last_processed_index(idx, filename="last_processed.txt"):
    with open(filename, "w") as f:
        f.write(str(idx + 1))


def label_cross_section(df):
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 8))
    
    # Adjust subplot parameters to give the map more space
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, projection=ccrs.PlateCarree())

    # Plot the elevation profile on the left subplot
    ax1.plot(df['dist_along'], df['elevation'], '-o', markersize=2, color='blue')
    ax1.set_xlabel('Along Track Distance')
    ax1.set_ylabel('Elevation')

    # Determine the viewing direction based on the first and last point in the cross-section
    start_point = df.iloc[0]
    end_point = df.iloc[-1]
    viewing_direction = "West to East" if start_point.geometry.x < end_point.geometry.x else "East to West"

    # Add an arrow to indicate the start of the cross-section
    ax1.annotate('', xy=(min(df['dist_along']), start_point['elevation']), xytext=(-50, 0),
                 textcoords='offset points', arrowprops=dict(arrowstyle="->", color='red'))

    # Add the satellite background image on the right subplot with a higher zoom level
    osm_background = cimgt.GoogleTiles(style='satellite', cache=True)
    ax2.add_image(osm_background, 15)  # Adjust zoom level as needed

    # Ensure cross_section_points is in the correct CRS
    plot_gdf = df.to_crs('EPSG:4326')

    # Check the bounds
    bounds = plot_gdf.total_bounds

    # Plot the elevation points on the map, colored by elevation
    scatter = ax2.scatter(
        plot_gdf.geometry.x,
        plot_gdf.geometry.y,
        c=plot_gdf['elevation'],
        cmap='terrain',
        marker='o',
        edgecolor='k',
        linewidth=0.5,
        s=10,
        transform=ccrs.PlateCarree()
    )

    start_point_map = plot_gdf.iloc[2].geometry
    # Assuming the second point is close enough to create a small tail for the arrow
    tail_point_map = plot_gdf.iloc[0].geometry
    ax2.annotate('', xy=(tail_point_map.x, tail_point_map.y), xytext=(start_point_map.x, start_point_map.y),
                 arrowprops=dict(arrowstyle="<-",
                                 color='red',
                                 lw=1),
                 transform=ccrs.PlateCarree())

    # Add a colorbar for the elevation
    plt.colorbar(scatter, ax=ax2, orientation='vertical', label='Elevation')

    # Set the extent to the bounds of the cross section points, with a small buffer
    buffer = .001  # Add a small buffer to ensure all points are within the view
    ax2.set_extent([bounds[0] - buffer, bounds[2] + buffer, bounds[1] - buffer, bounds[3] + buffer], crs=ccrs.PlateCarree())

    # Rest of the labeling logic remains the same...
    points = {'channel': [], 'ridge1': [], 'floodplain1': [], 'ridge2': [], 'floodplain2': []}
    labels = list(points.keys())
    current_label_idx = 0

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    ax1.set_title(f"Pick {labels[current_label_idx]}")

    plotted_points = []

    def onclick(event):
        nonlocal current_label_idx
        ix, iy = event.xdata, event.ydata

        points[labels[current_label_idx]].append((ix, iy))
        print(f"Labelled {labels[current_label_idx]} at x = {ix}, y = {iy}")

        colors = {'channel': 'b', 'ridge1': 'g', 'ridge2': 'g',
                  'floodplain1': 'r', 'floodplain2': 'r'}

        plotted_point, = ax1.plot(ix, iy, 'X', markersize=10, color=colors[labels[current_label_idx]])
        plotted_points.append(plotted_point)

        current_label_idx += 1
        if current_label_idx < len(labels):
            ax1.set_title(f"Pick {labels[current_label_idx]}")
        else:
            ax1.set_title("All points labeled. Press 'd' to save and continue.")
        fig.canvas.draw()

    def onkey(event):
        nonlocal current_label_idx
        if event.key == 'u':
            if plotted_points:
                current_label_idx -= 1
                removed_point = points[labels[current_label_idx]].pop()
                plotted_point_to_remove = plotted_points.pop()
                plotted_point_to_remove.remove()
                ax1.set_title(f"Pick {labels[current_label_idx]}")
                fig.canvas.draw()
        elif event.key == 'd':
            plt.close()

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show()

    return df, points


# Define n
labeled_data = []

total_cross_sections_to_process = len(cross_sections[::n])

# Get the last processed index
start_idx = get_last_processed_index()

# Process each cross-section
for idx in range(start_idx, len(cross_sections), n):
    df = cross_sections[idx]
    # Create a GeoDataFrame for the current cross section points
    
    remaining = total_cross_sections_to_process - (idx // n) - 1
    print(f"Processing cross-section {idx + 1} of {len(cross_sections)} ({remaining} remaining)")

    df, labeled_points = label_cross_section(df)

    # If no points are labeled and it's the first cross-section, save the empty dataframe
    if idx == 0 and all(not v for v in labeled_points.values()):
        empty_df = get_empty_dataframe_with_columns()
        empty_df.to_csv(f"data/{name}_output.csv", index = False)
    else:
        df = compute_distance_to_river_center(df, labeled_points)
        labeled_data.append((df, labeled_points))
        # Update and save to CSV after every cross section
        update_and_save_to_csv(df, labeled_points, filename = f"data/{name}_output.csv")
        save_last_processed_index(idx)

# %%

