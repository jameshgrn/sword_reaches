import numpy as np
import pandas as pd

def compute_along_track_distance(cross_section):
    """
    Compute the along-track distance for each point in a cross section.

    Parameters:
    - cross_section: A GeoDataFrame representing a cross section of points.

    Returns:
    - A pandas Series with the cumulative distance for each point in the cross section.
    """
    # Extract coordinates for each point in the cross section
    coords = cross_section.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

    # Compute the cumulative distance for each point
    distances = [0]  # Start with a distance of 0 for the first point
    for i in range(1, len(coords)):
        prev_coord, curr_coord = coords[i - 1], coords[i]
        distance = np.sqrt((curr_coord[0] - prev_coord[0]) ** 2 + (curr_coord[1] - prev_coord[1]) ** 2)
        distances.append(distances[-1] + distance)

    return pd.Series(distances, index = cross_section.index)