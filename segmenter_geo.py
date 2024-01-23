#%%
from geometric_operations import perform_geometric_operations
from gee_sampler import perform_cross_section_sampling
import uuid
import ee
import geopandas as gpd
from sqlalchemy import create_engine
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from shapely.geometry import shape
import json
# Define the PostgreSQL connection parameters
db_params = {
    'dbname': 'jakegearon',
    'user': 'jakegearon',
    'password': 'Derwood15',
    'host': 'localhost',
    'port': 5432
}

# Create a connection to the PostgreSQL database
engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

# Get the entire dataframe
sql = f"""
    SELECT * FROM sword_reachesv16;
    """
df = gpd.read_postgis(sql, engine, geom_col='geom', crs='EPSG:4326')

random_reach_id = 61569000491
name = "A002_03_4"

# Create a GeoDataFrame for the result with the correct geometry column and CRS
result = gpd.GeoDataFrame(columns=df.columns, geometry='geom', crs='EPSG:4326')
result.set_crs(epsg=4326, inplace=True)

# Keep track of visited nodes
visited = set()

def process_neighbours(current_reach_id, direction, tributaries=True):
    # Check if the current node has already been visited to prevent loops
    if current_reach_id in visited:
        return

    # Mark the current node as visited
    visited.add(current_reach_id)

    # Get the row corresponding to the current reach
    row = df[df['reach_id'] == current_reach_id]

    if not row.empty:
        row = row.iloc[0]
        # Add the row to the result DataFrame
        result.loc[current_reach_id, df.columns.difference(['geom'])] = row[df.columns.difference(['geom'])]
        result.loc[current_reach_id, 'geom'] = row['geom']

        # Identify if the current reach is a tributary
        upstream_neighbours = str(row['rch_id_up']).split()
        downstream_neighbours = str(row['rch_id_dn']).split()
        if len(upstream_neighbours) > 1 and len(downstream_neighbours) == 1:
            if not tributaries:  # If we're not processing tributaries, stop here
                return

        # Process neighbours
        neighbours = row['rch_id_up'] if direction == 'up' else row['rch_id_dn']
        if pd.notnull(neighbours) and neighbours != '':
            neighbours = [int(neighbour) for neighbour in str(neighbours).split()]
            for neighbour in neighbours:
                # Recursive call for each neighbour's neighbours
                process_neighbours(neighbour, direction, tributaries)

# Start the traversal from the random reach id
# Start the traversal from the random reach id for upstream
process_neighbours(random_reach_id, 'up', tributaries=False)

# Reset the visited set
visited = set()

# Start the traversal from the random reach id for downstream
process_neighbours(random_reach_id, 'down', tributaries=False)

# Convert the reach_ids in result to a list
reach_ids = set(result.index.tolist())
# Convert the list to a string format suitable for SQL IN clause
reach_ids_str = ','.join(map(str, reach_ids))

# Get the matching rows from sword_nodesv16
sql_nodes = f"""
    SELECT * FROM sword_nodesv16
    WHERE reach_id IN ({reach_ids_str});
    """
node_gdf = gpd.read_postgis(sql_nodes, engine, geom_col='geom', crs='EPSG:4326')
node_gdf = node_gdf.join(result[['slope']], on='reach_id')
node_gdf.rename(columns={'geom': 'original_geom'}, inplace=True)
node_gdf.set_geometry('original_geom', inplace=True)
node_gdf.to_crs('EPSG:3857', inplace=True)
#%%
# Plot the nodes
osm_background = cimgt.GoogleTiles(style='satellite')
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.add_image(osm_background, 10)
plot_gdf = node_gdf.copy().to_crs('EPSG:4326')
plot_gdf.plot(ax=ax, column='width', cmap='jet', legend=True, markersize=1)
bounds= plot_gdf.total_bounds
ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

cross_section_points = perform_geometric_operations(node_gdf)
cross_section_points = cross_section_points.to_crs('EPSG:4326')
osm_background = cimgt.GoogleTiles(style='satellite')
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.add_image(osm_background, 10)
plot_gdf = cross_section_points.copy().to_crs('EPSG:4326')
plot_gdf.plot(ax=ax, column='width', cmap='jet', legend=True, markersize=1)
bounds= plot_gdf.total_bounds
ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

# Generate a random UUID
unique_id = uuid.uuid4()
unique_id_str = str(unique_id)
all_elevations_gdf = perform_cross_section_sampling(cross_section_points, unique_id_str)
all_elevations_gdf.to_parquet(f'all_elevations_gdf_{name}.parquet')
#%%
from scipy.stats import kurtosis

def calculate_relief(group):
    center_third = group[group['dist_along'].between(group['dist_along'].quantile(0.33), group['dist_along'].quantile(0.66))]
    outer_two_thirds = group[~group.index.isin(center_third.index)]
    
    relief = center_third['elevation'].quantile(0.95) - outer_two_thirds['elevation'].quantile(0.15)
    return relief

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd

# Check if 'cross_id' is in the columns and set it as the index if it is
if 'cross_id' in all_elevations_gdf.columns:
    all_elevations_gdf.set_index('cross_id', inplace=True)
else:
    # If 'cross_id' is not in the columns, it might already be the index
    # If it's not the index either, raise an error
    if 'cross_id' != all_elevations_gdf.index.name:
        raise KeyError("'cross_id' is neither in the columns nor the index.")

# Perform the aggregation
cross_section_stats = all_elevations_gdf.groupby('cross_id').agg({
    'elevation': ['mean', 'var', 'skew', lambda x: kurtosis(x), 'median', 'std'],
    'slope': ['mean', 'std', 'skew', lambda x: kurtosis(x)]
}).ffill()

# Flatten the MultiIndex columns
cross_section_stats.columns = ['_'.join(col).strip() for col in cross_section_stats.columns.values]
cross_section_stats.rename(columns={'elevation_<lambda_0>': 'elevation_kurtosis', 'slope_<lambda_0>': 'slope_kurtosis'}, inplace=True)

# Calculate additional statistics
cross_section_stats['relief'] = all_elevations_gdf.groupby('cross_id').apply(calculate_relief)
cross_section_stats['azimuth_range'] = all_elevations_gdf.groupby('cross_id')['azimuth'].apply(lambda x: x.max() - x.min())
cross_section_stats['mean_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].mean()
cross_section_stats['std_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].std()
cross_section_stats['skew_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].skew()
cross_section_stats['kurtosis_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].apply(kurtosis)

# Join the aggregated stats back to the original DataFrame to include all information
# Make sure 'cross_id' is a column before joining
if 'cross_id' != all_elevations_gdf.index.name:
    all_elevations_gdf.reset_index(inplace=True)

cross_section_stats = all_elevations_gdf.merge(cross_section_stats, on='cross_id', how='left')

# Assuming 'geometry' is a column in all_elevations_gdf containing shapely geometry objects
# If 'geometry' is not a column, you should adjust the code accordingly.
cross_section_stats = gpd.GeoDataFrame(cross_section_stats, geometry='geometry')

# Save the DataFrame to a Parquet file
cross_section_stats.to_parquet(f'cross_section_stats_{name}.parquet')
#%%
# # Plotting
# plt.figure(figsize=(10, 10))
# plt.scatter(cross_section_stats['mean_slope'], cross_section_stats['relief'])
# plt.xlabel('Mean Slope')
# plt.ylabel('Relief')
# plt.title('Slope vs Relief')
# plt.show()


# %%
