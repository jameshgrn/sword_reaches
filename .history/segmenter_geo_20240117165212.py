#%%
from geometric_operations import perform_geometric_operations, select_nodes_based_on_meander_length_sinuosity_and_azimuth
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

# # Select a random reach_id
# sql_rand = f"""
#     SELECT reach_id FROM sword_reachesv16
#     ORDER BY RANDOM()
#     LIMIT 1;
#     """
# rand = pd.read_sql(sql_rand, engine)
# random_reach_id = rand.iloc[0]['reach_id']
random_reach_id = 11493100371

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
all_elevations_gdf.rename(columns={'b1': 'elevation'}, inplace=True)
all_elevations_gdf.to_parquet('all_elevations_gdf_AFR.parquet')
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

cross_section_stats = all_elevations_gdf.groupby('cross_id').agg({
    'elevation': ['mean', 'var', 'skew', lambda x: kurtosis(x), 'median', 'std'],
    'slope': ['mean', 'std', 'skew', lambda x: kurtosis(x)]
}).ffill()

cross_section_stats.columns = ['_'.join(col).strip() for col in cross_section_stats.columns.values]
cross_section_stats.rename(columns={'elevation_<lambda_0>': 'elevation_kurtosis', 'slope_<lambda_0>': 'slope_kurtosis'}, inplace=True)

cross_section_stats['relief'] = all_elevations_gdf.groupby('cross_id').apply(calculate_relief)

# Calculate the azimuth range
cross_section_stats['azimuth_range'] = all_elevations_gdf.groupby('cross_id')['azimuth'].apply(lambda x: x.max() - x.min())

# Calculate the mean slope
cross_section_stats['mean_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].mean()

# Calculate the standard deviation of the slope
cross_section_stats['std_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].std()

# Calculate the skewness of the slope
cross_section_stats['skew_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].skew()

# Calculate the kurtosis of the slope
cross_section_stats['kurtosis_slope'] = all_elevations_gdf.groupby('cross_id')['slope'].apply(kurtosis)

# Merge the node_gdf data with cross_section_stats
cross_section_stats_df = cross_section_stats.merge(all_elevations_gdf, left_index=True, right_on='cross_id', how='left')

# Convert the DataFrame back to a GeoDataFrame before saving to parquet
cross_section_stats = gpd.GeoDataFrame(cross_section_stats_df, geometry='.geo')

cross_section_stats.to_parquet('cross_section_stats_AFR.parquet')
#%%
# Plotting
plt.figure(figsize=(10, 10))
plt.scatter(cross_section_stats['mean_slope'], cross_section_stats['relief'])
plt.xlabel('Mean Slope')
plt.ylabel('Relief')
plt.title('Slope vs Relief')
plt.show()

# Define the number of rows and columns for the subplot
n_rows = 4
n_cols = 5

# Create a figure and axes for the subplot
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 16))

# List of variables to plot
variables = ['elevation_mean', 'elevation_var', 'elevation_skew', 'elevation_kurtosis', 'elevation_median', 'elevation_std', 'slope_mean', 'slope_std', 'slope_skew', 'slope_kurtosis', 'relief', 'azimuth_range']

# Iterate over each subplot and create a scatter plot with lowess fit
for i, ax in enumerate(axs.flatten()):
    if i < len(variables):
        sns.lineplot(x='dist_out', y=variables[i], data=cross_section_stats, ax=ax)
        lowess_results = lowess(cross_section_stats[variables[i]], cross_section_stats['dist_out'], frac=0.20)
        ax.plot(lowess_results[:, 0], lowess_results[:, 1], color='red')
        ax.set_title(variables[i])
        ax.invert_xaxis()

# Remove empty subplots
for i in range(len(variables), n_rows*n_cols):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.show()
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Assuming 'df' is your DataFrame containing all the data
X = df.drop(['channel_dist_along', 'ridge1_dist_along', 'ridge2_dist_along', 'floodplain1_dist_along', 'floodplain2_dist_along', 'reach_id', 'node_id'], axis=1)  # Drop non-feature columns
y = df[['channel_dist_along', 'ridge1_dist_along', 'ridge2_dist_along', 'floodplain1_dist_along', 'floodplain2_dist_along']]  # Target variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the base regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the MultiOutputRegressor
multi_output_rf = MultiOutputRegressor(rf)

# Train the model
multi_output_rf.fit(X_train, y_train)

# Make predictions
y_pred = multi_output_rf.predict(X_test)

# Evaluate the model for each target
for i, target in enumerate(y.columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    print(f"Mean Squared Error for {target}: {mse}")

