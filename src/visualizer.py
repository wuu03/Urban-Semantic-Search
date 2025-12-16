# src/visualizer.py
import math
import matplotlib.pyplot as plt
from PIL import Image


def visualize_search_results(search_results, original_image: Image.Image, patch_size: tuple, query_text: str):
    """
    Visualizes search results from Qdrant by cropping patches from the original map.

    Args:
        search_results: List of ScoredPoint objects from Qdrant.
        original_image: The original full-resolution PIL Image.
        patch_size: Tuple (width, height) of the patches.
        query_text: The processed_text query used for searching.
    """
    k_neighbors = len(search_results)
    if k_neighbors == 0:
        print("No results found.")
        return

    cols_per_row = 5
    rows = math.ceil(k_neighbors / cols_per_row)

    # Dynamically adjust figure size
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 4 * rows))
    if k_neighbors == 1:
        axes = [axes]  # Handle single result case

    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    fig.suptitle(f"Search Results for: '{query_text}'", fontsize=16)

    for i, hit in enumerate(search_results):
        ax = axes_flat[i]
        payload = hit.payload

        # Retrieve coordinates from payload
        x, y = payload['pixel_coords']
        w, h = patch_size

        # Crop the patch dynamically from the source image
        try:
            result_patch = original_image.crop((x, y, x + w, y + h))
            ax.imshow(result_patch)

            # Display metadata in title
            lat = payload.get('location', {}).get('lat', 0)
            lon = payload.get('location', {}).get('lon', 0)
            ax.set_title(f"Score: {hit.score:.4f}\nLoc: {lat:.4f}, {lon:.4f}", fontsize=10)
        except Exception as e:
            ax.set_title(f"Error loading patch: {e}", fontsize=8)

        ax.axis('off')

    # Hide empty subplots
    for i in range(k_neighbors, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()
