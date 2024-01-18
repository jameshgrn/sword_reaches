
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

    # Sample the elevation at all points
    profiles = fabdem.sampleRegions(collection=fc_points, scale=30)
    
    task = ee.batch.Export.table.toBigQuery(
      collection=myFeatureCollection,
      table=f'leveefinders.swords_and_crosses.{unique_id}',
      description=f'placed {unique_id} which corresponds to in bigquery',
      append=False)
    task.start()

    
    task.start()

    # Now wait for the task to complete
    while task.active():
        print(f'Task {task.id} is still running')
        time.sleep(10)
    print(f'Task {task.id} is completed')

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
        final_df['geometry'] = final_df['geometry'].apply(wkt.loads)
        gdf = GeoDataFrame(final_df, geometry='geometry')
    except KeyError:
        final_df['.geo'] = final_df['.geo'].apply(lambda x: shape(json.loads(x)))
        gdf = GeoDataFrame(final_df, geometry='.geo')
        
    return gdf

from google.oauth2 import service_account
from pandas import read_gbq

def read_data_from_bigquery(project_id, dataset_id, table_id):
    credentials_path = 'leveefinders-a9d0bde21676.json'
    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    # Construct the fully-qualified table name
    table_name = f"{project_id}.{dataset_id}.{table_id}"

    # Read the data from BigQuery
    df = read_gbq(table_name, credentials=credentials, project_id=project_id)

    try:
        df['geometry'] = df['geometry'].apply(wkt.loads)
        gdf = GeoDataFrame(df, geometry='geometry')
    except KeyError:
        df['.geo'] = df['.geo'].apply(lambda x: shape(json.loads(x)))
        gdf = GeoDataFrame(df, geometry='.geo')

    return gdf

def perform_cross_section_sampling(cross_section_points, unique_id):
    initialize_gee()
    fabdem = get_fabdem()
    process_all_points(fabdem, cross_section_points, unique_id)
    # Read the data from GCS
    cross_section_points_elevations = read_data_from_bigquery('leveefinders', 'crosses_and_swords', f'{unique_id}')
    return cross_section_points_elevations