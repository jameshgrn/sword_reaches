#%%
# Standard library imports
import configparser
import requests

# Third-party library imports for data handling
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
import fsspec
import aiohttp


# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Geospatial and geodesy libraries
from pygeodesy.points import Numpy2LatLon
from pygeodesy.ellipsoidalKarney import LatLon
from pygeodesy.geoids import GeoidKarney
from shapely.geometry import LineString, Polygon
from shapely.geometry.geo import mapping

# Database connection library
from sqlalchemy import create_engine

# ICESat-2 data processing libraries
import sliderule
from sliderule import earthdata, icesat2

# STAC client for searching and accessing geospatial assets
from pystac_client import Client


rid = pd.read_csv('/Users/jakegearon/CursorProjects/sword_reaches/river_discharge_data.csv').reach_id.iloc[35]
rid_df = pd.read_csv('/Users/jakegearon/CursorProjects/sword_reaches/river_discharge_data.csv')
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
    SELECT * FROM sword_reachesv16 WHERE reach_id = {rid};
    """
df = gpd.read_postgis(sql, engine, geom_col='geom', crs='EPSG:4326')
df_proj = df.to_crs(epsg=3857)
#icesat2.init("slideruleearth.io")

polygon = df_proj.buffer(df_proj['width']*3).to_crs(epsg=4326).unary_union
region = sliderule.toregion(polygon)


# Convert region to a shapely Polygon
region_polygon = Polygon([(point['lon'], point['lat']) for point in region['poly']])

# Simplify the polygon (adjust the tolerance as needed)
simplified_polygon = region_polygon.simplify(tolerance=0.01, preserve_topology=True)

# Convert the simplified polygon back to GeoJSON
region_geojson = mapping(simplified_polygon)

# Now use the simplified GeoJSON with the STAC API
catalog = Client.open("https://cmr.earthdata.nasa.gov/stac/NSIDC_ECS",)
query = catalog.search(
    collections=["ATL13"], limit=10, intersects=region_geojson
)

items = list(query.items())
print(f"Found: {len(items):d} datasets")
url = items[0].assets['data'].href


# Custom session class to handle redirects with authentication
class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']

# Load credentials from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
username = config.get('earthdata', 'username')
password = config.get('earthdata', 'password')

# Ensure that credentials are provided
assert username and password, "You must provide your Earthdata Login credentials"

# Assuming 'url' is the URL to the HDF5 file
with fsspec.open(url, mode='rb', client_kwargs={'auth': aiohttp.BasicAuth(username, password)}) as f:
    with h5py.File(f, 'r') as hdf:
        # Display the root keys in the HDF5 file
        print("Keys in the HDF5 file:", list(hdf.keys()))
        
        # Example: Access a dataset within the HDF5 file
        # This assumes there's a dataset named 'data' at the root of the HDF5 file
        # Adjust the string 'data' to match the actual dataset you're interested in
        if 'data' in hdf:
            data = hdf['data'][:]
            print("Data from the 'data' dataset:", data)
        else:
            print("Dataset 'data' not found in the HDF5 file.")
#%%
#