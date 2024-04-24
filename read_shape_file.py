import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd

# open and read file
shapefile_path = "shapefiles/99bfd9e7-bb42-4728-87b5-07f8c8ac631c2020328-1-1vef4ev.lu5nk.shp"
gdf = gpd.read_file(shapefile_path)

# extract mainsland polygon of norway
# print(gdf)
# mainland_polygon = gdf["geometry"][4].geoms[1]
mainland_polygon = gdf["geometry"][166].geoms[1]
print(list(mainland_polygon.centroid.xy))
# define the bounding box outside norway
buffer = 2  # num latitude and longitude lines extra on all sides of grid 
bbox = mainland_polygon.bounds # bounding box
lat_min, lon_min, lat_max, lon_max = bbox # limits of mainland norway 
num_intervals = 100  # number of intervals
latitudes = np.linspace(lat_min - buffer, lat_max + buffer, num_intervals) # evenly spaced latitudes
longitudes = np.linspace(lon_min - buffer, lon_max + buffer, num_intervals) # evenly spaces longitudes
grid = np.array(np.meshgrid(latitudes, longitudes)) # meshgrid of latitudes and longitudes of shape (2, 30, 30)

# generate grid points within mainland
grid_points = np.vstack((grid[0].ravel(), grid[1].ravel())).T # reshapes to (900, 2) 
within_mainland_mask = np.array(
    [Point(x, y).within(mainland_polygon) for x, y in grid_points]) # gets indices which are within the mainland polygon (mask)
mainland_grid_points = grid_points[within_mainland_mask] # applies the mask to grid_points
variables = np.array([num_intervals, buffer]) # stores variables needed for later


# plots borders of norway on top of grid withing borders of norway
country_border = mainland_polygon.exterior.xy
x, y = country_border
np.savez("saved_arrays/norway_grid_files.npz",grid_points=grid_points,mainland_grid_points=mainland_grid_points,
         country_border=country_border,variables=variables)
plt.scatter(x, y, s=3, label="Mainland Boundary")
plt.scatter(mainland_grid_points[:, 0], mainland_grid_points[:,1], s=3, label="Grid Points Within Mainland")
plt.legend()
plt.show()
