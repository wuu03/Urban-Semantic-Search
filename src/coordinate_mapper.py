import pandas as pd
import numpy as np
import math
from scipy.interpolate import Rbf

class VeniceCoordinateMapper:
    """
    A utility class to handle the coordinate transformation from the historical map's 
    pixel coordinates to the modern geographical coordinate system (EPSG:3857).
    
    It uses Thin Plate Spline (TPS) interpolation based on Ground Control Points (GCPs)
    exported from QGIS.
    """

    def __init__(self, gcp_file_path):
        """
        Initialize the mapper by loading GCPs and training the TPS model.
        
        Args:
            gcp_file_path (str): Path to the .points file exported from QGIS.
        """
        print(f"[Mapper] Loading GCPs from {gcp_file_path}...")
        self.gcp_df = self._load_gcps(gcp_file_path)
        
        # Check for inverted Y-axis (common in image coordinate systems vs. geo systems)
        # If all pixel_y values are negative, we assume QGIS exported them with an inverted axis.
        self.invert_y = (self.gcp_df['pixel_y'] < 0).all()
        if self.invert_y:
            print("[Mapper] Detected inverted Y-axis in GCPs. Input pixel Y will be automatically flipped.")
            
        print("[Mapper] Training TPS model...")
        self.rbf_x, self.rbf_y = self._train_tps_model()
        print("[Mapper] Initialization complete.")

    def _load_gcps(self, filename):
        """
        Internal method to parse the QGIS .points file format.
        """
        gcps = []
        try:
            with open(filename, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    # Skip metadata, comments, and headers
                    if not line or line.startswith('#') or line.startswith('CRS') or 'mapX' in line:
                        continue
                    try:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            # QGIS format: mapX, mapY, pixelX, pixelY
                            gcps.append({
                                'map_x': float(parts[0]),
                                'map_y': float(parts[1]),
                                'pixel_x': float(parts[2]),
                                'pixel_y': float(parts[3])
                            })
                    except ValueError:
                        continue
        except Exception as e:
            raise IOError(f"Failed to read GCP file: {e}")
            
        if not gcps:
            raise ValueError("No valid GCP points found in the file.")
            
        return pd.DataFrame(gcps)

    def _train_tps_model(self):
        """
        Trains the Radial Basis Function (Rbf) models for X and Y interpolation.
        'thin_plate' function is used to simulate the physical TPS deformation.
        """
        # Train model for Map X
        rbf_x = Rbf(self.gcp_df['pixel_x'], self.gcp_df['pixel_y'], self.gcp_df['map_x'], function='thin_plate', smooth=0)
        # Train model for Map Y
        rbf_y = Rbf(self.gcp_df['pixel_x'], self.gcp_df['pixel_y'], self.gcp_df['map_y'], function='thin_plate', smooth=0)
        return rbf_x, rbf_y

    def transform_point(self, pixel_x, pixel_y):
        """
        Transforms a single (x, y) pixel coordinate to EPSG:3857 (Web Mercator).
        """
        # Handle Y-axis inversion if necessary
        if self.invert_y and pixel_y > 0:
            pixel_y = -pixel_y
            
        geo_x = self.rbf_x(pixel_x, pixel_y)
        geo_y = self.rbf_y(pixel_x, pixel_y)
        return float(geo_x), float(geo_y)

    def to_wgs84(self, x, y):
        """
        Converts EPSG:3857 (Meters) to WGS84 (Latitude, Longitude).
        Useful for frontend mapping tools like Leaflet or Mapbox.
        """
        lon = (x / 20037508.34) * 180
        lat = (y / 20037508.34) * 180
        lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
        return lat, lon

    def process_patch_coordinates(self, patch_x, patch_y, patch_size):
        """
        Calculates the geographical geometry for a specific image patch.
        It computes the coordinates for the Center and the 4 Corners.
        
        Args:
            patch_x (int): Top-left X pixel coordinate of the patch.
            patch_y (int): Top-left Y pixel coordinate of the patch.
            patch_size (int): The width/height of the patch.
            
        Returns:
            dict: A dictionary containing 'epsg3857' and 'wgs84' coordinates for 
                  'center', 'top_left', 'top_right', 'bottom_right', 'bottom_left'.
        """
        if isinstance(patch_size, (tuple, list)):
            # If it's a tuple like (256, 256), take width and height
            # Note: Assuming (width, height) or (height, width). 
            # If square, it doesn't matter.
            w = patch_size[0]
            h = patch_size[1] if len(patch_size) > 1 else patch_size[0]
        else:
            # If it's a single integer like 256
            w = h = patch_size

        # Define key points in pixel space (relative to top-left origin)
        points_pixel = {
            'center':       (patch_x + w / 2, patch_y + h / 2),
            'top_left':     (patch_x, patch_y),
            'top_right':    (patch_x + w, patch_y),
            'bottom_right': (patch_x + w, patch_y + h),
            'bottom_left':  (patch_x, patch_y + h)
        }
        
        result = {'epsg3857': {}, 'wgs84': {}}
        
        for key, (px, py) in points_pixel.items():
            # 1. Transform to Projected Metric Coordinates (EPSG:3857)
            gx, gy = self.transform_point(px, py)
            result['epsg3857'][key] = [gx, gy]
            
            # 2. Convert to Geographic Coordinates (Lat, Lon)
            lat, lon = self.to_wgs84(gx, gy)
            result['wgs84'][key] = [lat, lon]
            
        return result