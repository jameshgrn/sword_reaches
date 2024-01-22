# tests/test_geometric_operations.py
import unittest
from geometric_operations import calculate_azimuth, make_cross_section, create_cross_sections, create_points, create_cross_section_points, calculate_distance_along_cross_section
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np

class TestGeometricOperations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the GeoDataFrame from the parquet file once for all tests
        cls.node_gdf = gpd.read_parquet("tests/data/for_testing.parquet")

    def test_calculate_azimuth(self):
        # Use the loaded GeoDataFrame for testing
        self.node_gdf = calculate_azimuth(self.node_gdf)
        calculated_azimuth = self.node_gdf['azimuth'].iloc[0]
        # Assert that the calculated azimuth matches the expected value
        expected_azimuth = -0.9317067209092407  # The expected value you confirmed
        self.assertAlmostEqual(calculated_azimuth, expected_azimuth, places=5)
    
    def test_make_cross_section(self):
        # Use the loaded GeoDataFrame for testing
        row = self.node_gdf.iloc[0]
        # Expected result is a LineString perpendicular to the azimuth
        result_line = make_cross_section(row)
        print("result_line:", result_line)
        self.assertTrue(expected_line.equals(result_line), "The cross section line is not as expected.")
    
    def test_create_cross_sections(self):
        # Use the loaded GeoDataFrame for testing
        result_gdf = create_cross_sections(self.node_gdf)
        self.assertIsInstance(result_gdf, gpd.GeoDataFrame, "The result should be a GeoDataFrame.")
        self.assertTrue('perp_geometry' in result_gdf, "The result GeoDataFrame should have a 'perp_geometry' column.")
        self.assertTrue(isinstance(result_gdf.iloc[0]['perp_geometry'], LineString), "The 'perp_geometry' should contain LineString objects.")
        
    def test_create_points(self):
        # Use the loaded GeoDataFrame for testing
        row = {
            'perp_geometry': self.node_gdf.iloc[0]['perp_geometry']
        }
        # Expected result is a list of Points along the LineString at intervals of 30
        expected_points = [Point(0, 0), Point(30, 0)]
        result_points = create_points(row)
        self.assertEqual(len(result_points), len(expected_points), "The number of points created is not as expected.")
        for result_point, expected_point in zip(result_points, expected_points):
            self.assertTrue(result_point.equals(expected_point), "The points created are not as expected.")
            
    def test_create_cross_section_points(self):
        # Use the loaded GeoDataFrame for testing
        sword_cross_sections = self.node_gdf
        # Expected result is a GeoDataFrame with points along the cross-sections
        result_gdf = create_cross_section_points(sword_cross_sections)
        self.assertIsInstance(result_gdf, gpd.GeoDataFrame, "The result should be a GeoDataFrame.")
        self.assertTrue('geometry' in result_gdf, "The result GeoDataFrame should have a 'geometry' column.")
        self.assertTrue(isinstance(result_gdf.iloc[0]['geometry'], Point), "The 'geometry' should contain Point objects.")
    
    def test_calculate_distance_along_cross_section(self):
        # Use the loaded GeoDataFrame for testing
        gdf = self.node_gdf
        # Expected result is a GeoDataFrame with a new 'dist_along' column containing cumulative distances
        result_gdf = calculate_distance_along_cross_section(gdf)
        self.assertIsInstance(result_gdf, gpd.GeoDataFrame, "The result should be a GeoDataFrame.")
        self.assertTrue('dist_along' in result_gdf, "The result GeoDataFrame should have a 'dist_along' column.")
        self.assertEqual(result_gdf['dist_along'].iloc[0], 0, "The distance along the cross-section should start at 0.")
        self.assertEqual(result_gdf['dist_along'].iloc[1], 30, "The distance along the cross-section should be 30 for the second point.")

if __name__ == '__main__':
    unittest.main()
