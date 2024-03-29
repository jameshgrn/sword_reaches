import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

def calculate_azimuth(node_gdf):
    node_gdf = node_gdf.sort_values(['reach_id', 'dist_out'])
    node_gdf['prev_geom'] = node_gdf.groupby('reach_id')['original_geom'].shift(1)
    node_gdf['next_geom'] = node_gdf.groupby('reach_id')['original_geom'].shift(-1)
    node_gdf.dropna(subset=['prev_geom', 'next_geom'], inplace=True)
    node_gdf['dx'] = (node_gdf['next_geom'].x - node_gdf['original_geom'].x) + (node_gdf['original_geom'].x - node_gdf['prev_geom'].x)
    node_gdf['dy'] = (node_gdf['next_geom'].y - node_gdf['original_geom'].y) + (node_gdf['original_geom'].y - node_gdf['prev_geom'].y)
    node_gdf['azimuth'] = np.arctan2(node_gdf['dy'], node_gdf['dx'])
    return node_gdf

def make_cross_section(row):
    start = (
        row['original_geom'].x + 15 * row['width'] * np.cos(row['azimuth'] + np.pi / 2),
        row['original_geom'].y + 10 * row['width'] * np.sin(row['azimuth'] + np.pi / 2)
    )
    end = (
        row['original_geom'].x + 10 * row['width'] * np.cos(row['azimuth'] - np.pi / 2),
        row['original_geom'].y + 10 * row['width'] * np.sin(row['azimuth'] - np.pi / 2)
    )
    return LineString([start, end]) 

def create_cross_sections(node_gdf):
    sword_cross_sections = node_gdf.copy()
    sword_cross_sections['perp_geometry'] = sword_cross_sections.apply(make_cross_section, axis=1)
    sword_cross_sections = sword_cross_sections.set_geometry('perp_geometry')

    # Drop unwanted geometry columns
    sword_cross_sections.drop(['original_geom', 'prev_geom', 'next_geom', 'dx', 'dy'], axis=1, inplace=True)

    return sword_cross_sections

def create_points(row):
    length = row['perp_geometry'].length
    num_points = int(length / 30)
    distances = [i*30 for i in range(num_points+1)]
    points = [row['perp_geometry'].interpolate(dist) for dist in distances]
    return points

def create_cross_section_points(sword_cross_sections):
    sword_cross_sections['points'] = sword_cross_sections.apply(create_points, axis=1)
    cross_section_points = sword_cross_sections.explode('points').reset_index(drop=True)
    cross_section_points.rename(columns={'points': 'geometry'}, inplace=True)
    cross_section_points.set_geometry('geometry', inplace=True, crs='EPSG:3857')
    cross_section_points = cross_section_points.drop(columns=[col for col in cross_section_points.columns if isinstance(cross_section_points[col].dtype, gpd.array.GeometryDtype) and col != 'geometry'])
    return cross_section_points

def calculate_distance_along_cross_section(gdf):
    """
    Calculates the cumulative distance along each cross-section for each point in the GeoDataFrame.
    
    :param gdf: GeoDataFrame with Point geometries representing points along cross-sections.
    :return: GeoDataFrame with an additional column 'dist_along' representing the distance along the cross-section.
    """
    # Iterate over each unique cross_id
    for cross_id in gdf['cross_id'].unique():
        # Select all points belonging to the current cross-section
        cross_section = gdf[gdf['cross_id'] == cross_id].sort_values(by='y')
        distances = [0]
        
        # Calculate the cumulative distance for each point in the cross-section
        for i in range(1, len(cross_section)):
            # Calculate the distance between this point and the previous point
            dist = cross_section.iloc[i].geometry.distance(cross_section.iloc[i-1].geometry)
            distances.append(distances[-1] + dist)
        
        # Assign the calculated distances to the 'dist_along' column for the current cross-section
        gdf.loc[gdf['cross_id'] == cross_id, 'dist_along'] = distances

    return gdf

def select_nodes_based_on_meander_length_sinuosity_and_azimuth(node_gdf, meander_fraction, sinuosity_threshold, min_distance, azimuth_range):
    # Select nodes where the cumulative distance is a multiple of the meander fraction
    # and the sinuosity is below the threshold
    potential_nodes = node_gdf[(node_gdf['dist_out'] % (node_gdf['meand_len'] * meander_fraction) < 10) & 
                               (node_gdf['sinuosity'] <= sinuosity_threshold)]

    # Convert azimuth range to radians
    azimuth_min, azimuth_max = np.radians(azimuth_range)

    # Initialize an empty GeoDataFrame to store the selected nodes
    selected_nodes = gpd.GeoDataFrame(columns=potential_nodes.columns)

    # Iterate over the potential nodes
    for idx, node in potential_nodes.iterrows():
        # Calculate the azimuth of the node and ensure it's within the range
        node_azimuth = node['azimuth']
        if azimuth_min <= node_azimuth <= azimuth_max:
            # If there are no selected nodes yet, select the first node
            if selected_nodes.empty:
                selected_nodes = selected_nodes.append(node)
            else:
                # Check the distance to the last selected node
                last_node = selected_nodes.iloc[-1]
                distance = node['original_geom'].distance(last_node['original_geom'])
                # If the distance is greater than the minimum distance, select the node
                if distance >= min_distance:
                    selected_nodes = selected_nodes.append(node)

    return selected_nodes

def perform_geometric_operations(node_gdf):
    # Select nodes based on meander length and sinuosity
    node_gdf = calculate_azimuth(node_gdf)
    # selected_nodes = select_nodes_based_on_meander_length_sinuosity_and_azimuth(node_gdf, meander_fraction, sinuosity_threshold, min_distance=100, azimuth_range=(0, 2*np.pi))    
    sword_cross_sections = create_cross_sections(node_gdf)
    cross_section_points = create_cross_section_points(sword_cross_sections)
    cross_section_points['cross_id'] = cross_section_points.groupby(['node_id', 'reach_id']).ngroup()
    cross_section_points = calculate_distance_along_cross_section(cross_section_points)
    return cross_section_points

