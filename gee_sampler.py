
import os
import ee
import geemap
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import gcsfs
from geopandas import GeoDataFrame
from shapely import wkt
import time
from gcsfs.core import GCSFileSystem
import google.auth
import json
from shapely.geometry import shape

def initialize_gee():
    try:
        service_account = 'levee-cloud-storage@leveefinders.iam.gserviceaccount.com'
        credentials_path = 'leveefinders-a9d0bde21676.json'
        
        # Check if the credentials file exists
        if not os.path.isfile(credentials_path):
            print(f"Credentials file does not exist at: {credentials_path}")
            return

        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)
        print("Google Earth Engine has been initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Google Earth Engine: {e}")
    
def get_fabdem():
    fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    return fabdem.mosaic().setDefaultProjection('EPSG:4326',None,30)

def sample_point_elevation(feature, fabdem):
    elevation = fabdem.sample(region=feature.geometry(), scale=30).first().get('b1')
    return feature.set('elevation', elevation).copyProperties(feature)

def process_all_points(fabdem, cross_section_points, unique_id):
    # Convert the GeoDataFrame to a FeatureCollection
    fc_points = geemap.gdf_to_ee(cross_section_points)

    # Split the points into chunks
    chunk_size = 5000  # Adjust this value based on your data
    chunks = [cross_section_points.iloc[i:i+chunk_size] for i in range(0, len(cross_section_points), chunk_size)]

    tasks = []  # List to hold all tasks

    for i, chunk in enumerate(chunks):
        # Convert the chunk to a FeatureCollection
        fc_chunk = geemap.gdf_to_ee(chunk)
        # print(chunk.columns)
        # print(fc_chunk.getInfo())

        # Sample the elevation at all points in the current chunk
        profiles = fabdem.sampleRegions(collection=fc_chunk, scale=30, geometries=True)  # Add geometries=True to retain geometry

        # Export the results to Cloud Storage
        task = ee.batch.Export.table.toCloudStorage(
            collection=profiles,
            description=f'{unique_id}_Cross_Section_Sampling_{i}',
            bucket='leveefinders-test',
            fileNamePrefix=f'{unique_id}_Cross_Section_Sampling_{i}',
            fileFormat='CSV'
        )
        task.start()
        tasks.append(task)  # Add the task to the list

    # Now wait for all tasks to complete
    check_tasks_status(tasks)

def check_tasks_status(tasks):
    for task in tasks:
        while task.active():
            print(f'Task {task.id} is still running')
            time.sleep(10)
        print(f'Task {task.id} is completed')

def read_data_from_gcs(bucket_name, file_prefix):
    service_account = 'levee-cloud-storage@leveefinders.iam.gserviceaccount.com'
    credentials_path = 'leveefinders-a9d0bde21676.json'
    credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
    fs = GCSFileSystem('LeveeFinders', token=credentials)
    files = fs.ls(bucket_name)
    dfs = []
    for file in files:
        if file_prefix in file:
            with fs.open(file) as f:
                df = pd.read_csv(f)
                dfs.append(df)
    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)
    try:
        # If the geometry column is present, use it
        if 'geometry' in final_df.columns:
            final_df['geometry'] = final_df['geometry'].apply(wkt.loads)
            gdf = GeoDataFrame(final_df, geometry='geometry')
        else:
            # If the geometry is in the '.geo' column, parse it from JSON
            final_df['.geo'] = final_df['.geo'].apply(lambda x: shape(json.loads(x)))
            gdf = GeoDataFrame(final_df, geometry='.geo')
    except KeyError as e:
        print(f"Error processing geometry: {e}")
        gdf = GeoDataFrame(final_df)
        
    return gdf

def perform_cross_section_sampling(cross_section_points, unique_id):
    initialize_gee()
    fabdem = get_fabdem()
    process_all_points(fabdem, cross_section_points, unique_id)
    # Read the data from GCS
    cross_section_points_elevations = read_data_from_gcs('leveefinders-test', f'{unique_id}_Cross_Section_Sampling_')
    return cross_section_points_elevations