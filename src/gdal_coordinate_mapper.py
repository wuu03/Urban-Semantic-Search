from osgeo import gdal, osr
import math

class VeniceGDALMapper:
    """
    Coordinate mapper using GDAL's internal TPS (Thin Plate Spline) engine.
    
    Purpose:
    To strictly replicate the georeferencing logic used in QGIS. Using GDAL directly 
    avoids the slight mathematical discrepancies found in scipy's Rbf implementation, 
    especially regarding regularization and matrix solving strategies.
    """

    def __init__(self, gcp_file_path):
        """
        Initialize the mapper by loading Ground Control Points (GCPs) and 
        building the TPS transformer.
        
        Args:
            gcp_file_path (str): Path to the .points file exported from QGIS.
        """
        print(f"[GDAL Mapper] Loading GCPs from {gcp_file_path}...")
        self.gcps = self._load_gcps_gdal(gcp_file_path)
        
        # Initialize the Transformer
        # We create a temporary in-memory dataset to hold the GCPs for GDAL to process.
        self.transformer = self._create_transformer()
        print("[GDAL Mapper] TPS Transformer initialized successfully.")

    def _load_gcps_gdal(self, filename):
        """
        Parses the QGIS .points file and converts them into GDAL GCP objects.
        """
        gdal_gcps = []
        try:
            # QGIS .points files often use latin-1 encoding for special characters (like degree symbols)
            with open(filename, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments, CRS definitions, and header lines
                    if not line or line.startswith('#') or line.startswith('CRS') or 'mapX' in line:
                        continue
                    try:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            # Parse columns: mapX, mapY, pixelX, pixelY
                            map_x = float(parts[0])
                            map_y = float(parts[1])
                            pixel_x = float(parts[2])
                            pixel_y = float(parts[3])
                            
                            # Note: QGIS exports often have negative pixel_y values.
                            # We pass these raw values to GDAL; we will handle the sign logic
                            # during the transformation step if necessary.
                            # GDAL GCP args: (x, y, z, pixel, line)
                            gcp = gdal.GCP(map_x, map_y, 0, pixel_x, pixel_y)
                            gdal_gcps.append(gcp)
                    except ValueError:
                        continue
        except Exception as e:
            raise IOError(f"Failed to read GCP file: {e}")
            
        if not gdal_gcps:
            raise ValueError("No valid GCP points found in file.")
            
        return gdal_gcps

    def _create_transformer(self):
        """
        Creates the GDAL Transformer object configured for Thin Plate Spline (TPS).
        """
        # 1. Define the target Spatial Reference System (EPSG:3857 - Web Mercator)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        wkt = srs.ExportToWkt()
        
        # 2. Create a dummy in-memory dataset (1x1 pixel) to attach GCPs to
        driver = gdal.GetDriverByName('MEM')
        ds = driver.Create('', 1, 1, 0) 
        ds.SetGCPs(self.gcps, wkt)
        
        # 3. Create the Transformer
        # 'METHOD=GCP_TPS' explicitly selects the Thin Plate Spline algorithm.
        return gdal.Transformer(ds, None, ['METHOD=GCP_TPS'])

    def transform_point(self, pixel_x, pixel_y):
        """
        Transforms a single pixel coordinate (x, y) to Map Coordinates (EPSG:3857).
        
        Args:
            pixel_x (float): The X coordinate in image space.
            pixel_y (float): The Y coordinate in image space (usually positive).
            
        Returns:
            tuple: (map_x, map_y)
        """
        # Auto-detection for Y-axis inversion
        # QGIS GCPs usually store Y as negative values (Cartesian system).
        # Standard image processing uses positive Y (Matrix system).
        # If the first GCP has a negative Y line, but input is positive, we flip the input.
        if self.gcps[0].GCPLine < 0 and pixel_y > 0:
            input_line = -pixel_y
        else:
            input_line = pixel_y
            
        input_pixel = pixel_x

        # Execute transformation
        # TransformPoint arguments: (bDstToSrc, x, y, z)
        # 0 = Forward transform (Pixel -> Geo)
        success, point = self.transformer.TransformPoint(0, input_pixel, input_line)
        
        if not success:
            raise RuntimeError("GDAL Transformation failed for point.")
            
        return point[0], point[1] # Returns MapX, MapY

    def to_wgs84(self, x, y):
        """
        Helper: Converts EPSG:3857 (Meters) to WGS84 (Lat/Lon).
        Useful for frontend mapping libraries (Leaflet/Mapbox).
        """
        lon = (x / 20037508.34) * 180
        lat = (y / 20037508.34) * 180
        lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
        return lat, lon

    def process_patch(self, patch_x, patch_y, patch_size):
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