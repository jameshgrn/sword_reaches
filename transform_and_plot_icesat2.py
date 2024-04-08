Build ATL06 Request


def calculate_azimuth(linestring):
    """
    Calculate the azimuth of a linestring geometry.
    """
    start_point = linestring.coords[0]
    end_point = linestring.coords[-1]
    azimuth = np.arctan2(end_point[0] - start_point[0], end_point[1] - start_point[1])
    azimuth_degrees = np.degrees(azimuth) % 360
    return azimuth_degrees

def calculate_bearing_for_df(df):
    """
    Calculate the bearing between consecutive points in a dataframe.
    """
    bearings = []
    for i in range(len(df) - 1):
        lat1, lon1 = df.iloc[i]['transformed_lat'], df.iloc[i]['transformed_lon']
        lat2, lon2 = df.iloc[i + 1]['transformed_lat'], df.iloc[i + 1]['transformed_lon']
        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        bearings.append(bearing)
    # Append NaN for the last point which has no subsequent point to form a pair
    bearings.append(np.nan)
    return bearings

def adjust_slope_per_segment(slope, azimuth_difference):
    """
    Adjust the slope based on the azimuth difference using trigonometric functions.
    This function assumes the azimuth difference is in degrees.
    """
    # Convert azimuth difference from degrees to radians for trigonometric functions
    azimuth_difference_rad = np.radians(azimuth_difference)
    
    # Calculate the correction factor based on the azimuth difference.
    # This example uses the cosine of the azimuth difference, assuming that the slope
    # should be fully applied when the segment is aligned (difference = 0) and
    # no slope should be applied when perpendicular (difference = 90 or -90 degrees).
    correction_factor = np.cos(azimuth_difference_rad)
    
    # Adjust the slope by the correction factor
    adjusted_slope = slope * correction_factor
    
    return adjusted_slope

parms = {
    "poly": region['poly'],
    "srt": icesat2.SRT_INLAND_WATER,
    "len": 60.0,
    "res": 30.0,
    "maxi": 5,
    "cnf": 4,
    
}
rsps = sliderule.icesat2.atl06p(parms)
df_sr = gpd.GeoDataFrame(rsps)

if len(df_sr) == 0:
    raise RuntimeError("No results returned, try a larger area!")
# print length of final dataframe
print("Length of returned GeoDataFrame: ", len(df_sr))
df_sr['UID'] = df_sr.groupby(['rgt', 'cycle', 'spot']).ngroup().add(1)


df_sr['lat'] = df_sr.geometry.y  # lat col
df_sr['lon'] = df_sr.geometry.x  # lon col

# Calculate along track distance
df_sr['along_track'] = (((df_sr["x_atc"])) - min_d) - (((df_sr["x_atc"])) - min_d).iloc[0]

grouped = df_sr.groupby(['rgt', 'cycle', 'spot'])

# Calculate the minimum 'x_atc' for each group
df_sr['min_x_atc'] = grouped['x_atc'].transform('min')
ginterpolator = GeoidKarney("/Users/jakegearon/Downloads/geoids/egm2008-1.pgm")
# Calculate 'along_track' for each point relative to the minimum 'x_atc' within its group
df_sr['along_track'] = df_sr['x_atc'] - df_sr['min_x_atc']              
# Initialize the transformer
transformer = pyproj.Transformer.from_crs("EPSG:7912", "EPSG:9518", always_xy=True)
# Convert latitude and longitude columns to a NumPy array
lat_lon_array = np.column_stack((df_sr['lat'].values, df_sr['lon'].values, df_sr['h_mean'].values))
# Use Numpy2LatLon to handle the array as LatLon points
lat_lon_points = Numpy2LatLon(lat_lon_array, ilat=0, ilon=1)
df_sr['transformed_lon'], df_sr['transformed_lat'], df_sr['transformed_z'] = transformer.transform(lat_lon_array[:, 1], lat_lon_array[:, 0], lat_lon_array[:, 2])
lat_lon_array = np.column_stack((df_sr['transformed_lon'].values, df_sr['transformed_lat'].values, df_sr['transformed_z'].values))
# Use Numpy2LatLon to handle the array as LatLon points
lat_lon_points = Numpy2LatLon(lat_lon_array, ilat=0, ilon=1)
geoid_height = ginterpolator(lat_lon_points)
orthometric_height = df_sr['transformed_z'] - geoid_height  # Subtract geoid height from ellipsoidal height
df_sr['transformed_z'] = orthometric_height
icesat_gdf = df_sr.reset_index()

# Apply the function to calculate bearings for the entire dataframe
icesat_gdf['bearing'] = calculate_bearing_for_df(icesat_gdf)

# Now, you can calculate the azimuth difference as before
# Note: You might want to handle the NaN value for the last point in 'bearing'
icesat_gdf['azimuth_difference'] = icesat_gdf['bearing'] - linestring_azimuth

# Apply the adjusted slope for each segment
# Assuming 'icesat_gdf' has a 'azimuth_difference' column calculated as before
icesat_gdf['adjusted_slope'] = icesat_gdf['azimuth_difference'].apply(lambda x: adjust_slope_per_segment(slope, x))

# Now, apply the adjusted slope for detrending
# Assuming 'along_track' is the distance along the track and 'transformed_z' is the elevation
icesat_gdf['detrended_z'] = icesat_gdf.apply(lambda row: row['transformed_z'] - (row['along_track'] * row['adjusted_slope']), axis=1)

# Ensure 'time' column is of datetime type
icesat_gdf['time'] = pd.to_datetime(icesat_gdf['time'])

# Normalize 'time' to the start of each day for grouping
icesat_gdf['day'] = icesat_gdf['time'].dt.normalize()

# Extract year for grouping
icesat_gdf['year'] = icesat_gdf['time'].dt.year

# Access the 'slope' value for the specific reach_id from the 'df' DataFrame
slope = df.loc[df['reach_id'] == rid, 'slope'].values[0] / 1000

# Detrend the water surface elevations
icesat_gdf['detrended_z'] = icesat_gdf['transformed_z'] - (icesat_gdf['along_track'] * slope)

# Set up the figure with a GridSpec
fig = plt.figure(figsize=(12, 8))
# Adjust GridSpec layout to 2 rows and 2 columns
gs = gridspec.GridSpec(2, 2)

# Plot 1: Daily Water Surface Elevations (Detrended) by Day
ax0 = plt.subplot(gs[0, 0])  # First row, first column
sns.boxplot(x='day', y='detrended_z', data=icesat_gdf, ax=ax0, color='lightgrey')
sns.stripplot(x='day', y='detrended_z', data=icesat_gdf, ax=ax0, color='blue', size=2)
ax0.set_title('Daily Water Surface Elevations (Detrended) by Day')
ax0.set_xlabel('Day')
ax0.set_ylabel('Detrended Water Surface Elevation')
ax0.tick_params(axis='x', rotation=45)

# Plot 2: Daily Water Surface Elevations (Detrended) by Year
ax1 = plt.subplot(gs[1, 0])  # Second row, first column
sns.boxplot(x='year', y='detrended_z', data=icesat_gdf, ax=ax1, color='lightgrey')
sns.stripplot(x='year', y='detrended_z', data=icesat_gdf, ax=ax1, color='blue', size=2)
ax1.set_title('Daily Water Surface Elevations (Detrended) by Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('Detrended Water Surface Elevation')

# Plot 3: Daily Water Surface Elevations (Detrended) for All Data
ax2 = plt.subplot(gs[0, 1])  # First row, second column
sns.boxplot(y='detrended_z', data=icesat_gdf, ax=ax2, color='lightblue')
sns.stripplot(y='detrended_z', data=icesat_gdf, ax=ax2, color='blue', size=2)
ax2.set_title('Daily Water Surface Elevations (Detrended) for All Data')
ax2.set_xlabel('')
ax2.set_ylabel('Detrended Water Surface Elevation')

# Plot 4: Histogram of Detrended Water Surface Elevations
ax3 = plt.subplot(gs[1, 1])  # Second row, second column
sns.histplot(icesat_gdf['detrended_z'], ax=ax3, color='skyblue', kde=True)
ax3.set_title('Histogram of Detrended Water Surface Elevations')
ax3.set_xlabel('Detrended Water Surface Elevation')
ax3.set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('water_surface_elevations_plots.png', dpi=300)

# Show the plots
plt.show()

# %%
import plotly.io as pio
import plotly.express as px

pio.renderers.default = "vscode"
icesat_gdf['lat'] = icesat_gdf.geometry.y  # lat col
icesat_gdf['lon'] = icesat_gdf.geometry.x  # lon col
subsample = icesat_gdf.to_crs(epsg=3857)

year = 2021
month = '07'

fig = px.scatter_mapbox(subsample,
                        lat="transformed_lat",
                        lon="transformed_lon",
                        hover_name="detrended_z",
                        color="detrended_z",
                        hover_data=["detrended_z", "UID", "time", "along_track"], #"lat", "lon", "along_track",
                        height=500,
                        )
fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Planet",
                "source": [
                    "https://tiles0.planet.com/basemaps/v1/planet-tiles/global_monthly_%s_%s_mosaic/gmap/{z}/{x}/{y}.png?api_key=PLAKe691b336e29e445ca4ecc9490148e47d" % (year, month)
                    #"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                ]
            }])

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
# %%

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
UID = 3

temp_df_pre = icesat_gdf.query(f"UID == {UID}")
year = temp_df_pre.time.dt.year.unique()[0]
month = temp_df_pre.time.dt.month.unique()[0]
temp_df_pre["along_track"] -= temp_df_pre["along_track"].min()
#crop based on along_track
#temp_df_pre = temp_df_pre.query("along_track >= 5752 and along_track <= 8200")

fig = make_subplots(rows=2, cols=1, subplot_titles=(f"UID {UID}", ""), specs=[[{"type": "xy"}], [{"type": "mapbox"}]], vertical_spacing=0.1)

scatter_graph = px.scatter(temp_df_pre, x="along_track", y="detrended_z").update_traces(marker={"size":3})
line_graph = px.line(temp_df_pre, x="along_track", y="detrended_z").update_traces(line=dict(color="Black", width=0.5))
scatter_graph.add_trace(line_graph.data[0])

for trace in scatter_graph.data:
    fig.add_trace(trace, row=1, col=1)

mapbox_graph = px.scatter_mapbox(temp_df_pre, lat="transformed_lat", lon="transformed_lon", color="detrended_z",
                                 color_continuous_scale=px.colors.sequential.Viridis, size_max=5, zoom=10, hover_data=["transformed_z", "UID", "time", "along_track"])

for trace in mapbox_graph.data:
    fig.add_trace(trace, row=2, col=1)

# Convert the month to a string with a leading zero if necessary
formatted_month = f"{month:02d}"

fig.update_layout(
    height=750, 
    showlegend=False,
    mapbox_style="white-bg",
    mapbox_layers=[{
        "below": "traces",
        "sourcetype": "raster",
        "sourceattribution": "Planet",
        #"source": ["https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"]
        "source": [f"https://tiles0.planet.com/basemaps/v1/planet-tiles/global_monthly_{year}_{formatted_month}_mosaic/gmap/{{z}}/{{x}}/{{y}}.png?api_key=PLAKe691b336e29e445ca4ecc9490148e47d"]
    }],
    mapbox=dict(center=dict(lat=temp_df_pre['transformed_lat'].mean(), lon=temp_df_pre['transformed_lon'].mean()), zoom=10),
    template="plotly_white"  # Add this line

)

fig.show()
# %%
