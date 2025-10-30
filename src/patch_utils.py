# src/patch_utils.py
from PIL import Image
from tqdm import tqdm
import os

def extract_patches(image_path, patch_size=(224, 224), stride=(112, 112)):
    """
    Extracts overlapping patches from an image.

    Args:
        image_path (str): The file path to the image.
        patch_size (tuple): (width, height) The size of each patch.
        stride (tuple): (x_stride, y_stride) The overlap between patches.
                        (112, 112) represents a 50% overlap.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              a 'patch' (PIL.Image) and its 'coords' (x, y).
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return []

    try:
        image = Image.open(image_path).convert("RGB")
    except IOError:
        print(f"Error: Cannot open image at {image_path}")
        return []

    img_width, img_height = image.size
    patch_width, patch_height = patch_size
    stride_x, stride_y = stride

    patches_data = []

    # Calculate the total number of patches for the tqdm progress bar
    num_x = (img_width - patch_width) // stride_x + 1
    num_y = (img_height - patch_height) // stride_y + 1
    total_patches = num_x * num_y

    print(f"Extracting patches from {image_path}...")
    with tqdm(total=total_patches, desc="Patching Image") as pbar:
        for y in range(0, img_height - patch_height + 1, stride_y):
            for x in range(0, img_width - patch_width + 1, stride_x):
                box = (x, y, x + patch_width, y + patch_height)
                patch = image.crop(box)
                patches_data.append({
                    "patch": patch,
                    "coords": (x, y)
                })
                pbar.update(1)

    print(f"Successfully extracted {len(patches_data)} patches.")
    return patches_data