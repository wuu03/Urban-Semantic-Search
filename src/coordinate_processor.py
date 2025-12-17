# src/data_processor.py
import pandas as pd
from shapely import wkt
from tqdm import tqdm

# Enable tqdm for pandas
tqdm.pandas()


def _parse_and_get_centroid(wkt_str: str):
    """
    Helper function to parse WKT and return the centroid WKT.
    """
    if pd.isna(wkt_str) or wkt_str == "":
        return None

    try:
        # Load geometry from WKT string
        geo_obj = wkt.loads(wkt_str)
        # Calculate centroid
        centroid = geo_obj.centroid
        return centroid.wkt
    except Exception:
        # Return None for invalid geometries to be dropped later
        return None


def convert_geometry_to_centroid(df: pd.DataFrame, geometry_col: str = 'geometry') -> pd.DataFrame:
    """
    Converts a DataFrame column containing WKT geometries (Polygon/MultiPolygon)
    into their centroid Points.

    Args:
        df: Input Pandas DataFrame.
        geometry_col: Name of the column containing WKT strings.

    Returns:
        A new DataFrame with transformed geometry and dropped invalid rows.
    """
    print(f"Processing {len(df)} rows: Converting geometries to centroids...")

    # Create a copy to avoid SettingWithCopyWarning
    processed_df = df.copy()

    # Apply transformation with progress bar
    processed_df[geometry_col] = processed_df[geometry_col].progress_apply(_parse_and_get_centroid)

    # Drop rows where transformation failed
    original_count = len(processed_df)
    processed_df = processed_df.dropna(subset=[geometry_col])
    dropped_count = original_count - len(processed_df)

    if dropped_count > 0:
        print(f"Warning: Dropped {dropped_count} rows due to invalid geometry.")

    print(f"✅ Conversion complete. Remaining rows: {len(processed_df)}")
    return processed_df
