# src/visualization.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def create_heatmap(image_shape, patch_coords, scores, patch_size, blur_kernel_size=(31, 31)):
    """
    Creates a 2D heatmap array by accumulating scores at each patch location.
    
    Args:
        image_shape (tuple): (height, width) of the original map.
        patch_coords (list): List of (x, y) coordinates for all patches.
        scores (np.ndarray): A sparse array (default 0) containing the similarity score for each patch.
        patch_size (tuple): (width, height) of the patch.
        blur_kernel_size (tuple): Odd-numbered tuple (width, height) for the
                                  Gaussian blur kernel to smooth the heatmap.
                                  Set to None or (1, 1) to disable.
        
    Returns:
        np.ndarray: A smoothed, unnormalized heatmap array with the same size as the original image.
    """
    heatmap = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
    patch_width, patch_height = patch_size
    
    # Use np.maximum to ensure overlapping regions take the highest score.
    print("Creating heatmap array...")
    for i, (x, y) in enumerate(patch_coords):
        if scores[i] > 0: # Only process patches that have a score
            y_end = min(y + patch_height, image_shape[0])
            x_end = min(x + patch_width, image_shape[1])
            heatmap[y:y_end, x:x_end] = np.maximum(
                heatmap[y:y_end, x:x_end],
                scores[i]
            )
    
    # Apply Gaussian blur to smooth the blocky heatmap into a continuous one
    if blur_kernel_size and blur_kernel_size[0] > 1 and blur_kernel_size[1] > 1:
        print(f"Applying Gaussian blur with kernel {blur_kernel_size}...")
        heatmap = cv2.GaussianBlur(heatmap, blur_kernel_size, 0)
            
    return heatmap

def _overlay_blend(base, blend):
    """
    Numpy-vectorized implementation of the 'Overlay' blend mode.
    Blends the 'blend' layer onto the 'base' layer.
    """
    # Where base < 0.5 (dark), use Multiply: 2 * A * B
    low_part = 2 * base * blend
    
    # Where base >= 0.5 (light), use Screen: 1 - 2 * (1 - A) * (1 - B)
    high_part = 1 - 2 * (1 - base) * (1 - blend)
    
    # Combine based on the base layer's luminance
    return np.where(base < 0.5, low_part, high_part)

def apply_heatmap_overlay(original_image_path,
                          heatmap_array,
                          colormap=cv2.COLORMAP_INFERNO):
    """
    [VERSION 1: OVERLAY BLEND]
    Applies a color heatmap using 'Overlay' blending, which preserves
    the underlying texture, shadows, and highlights of the original image.
    The heatmap score acts as an alpha mask for this blend.
    """
    try:
        image_bgr = cv2.imread(original_image_path)
        if image_bgr is None:
            raise IOError(f"OpenCV could not read image: {original_image_path}")
        
        h, w, _ = image_bgr.shape
        
        # Normalize the heatmap to [0, 1]
        heatmap_min = heatmap_array.min()
        heatmap_max = heatmap_array.max()
        epsilon = 1e-6
        
        if (heatmap_max - heatmap_min) < epsilon:
            mask = np.zeros_like(heatmap_array, dtype=np.float32)
        else:
            mask = (heatmap_array - heatmap_min) / (heatmap_max - heatmap_min + epsilon)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Create the color heatmap
        heatmap_8bit = (mask * 255).astype(np.uint8)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_8bit, colormap)

        # Convert images to float [0, 1] for blending math
        base_float = image_bgr.astype(np.float32) / 255.0
        colormap_float = heatmap_color_bgr.astype(np.float32) / 255.0

        # Calculate the "Overlay" blend result
        overlay_blend_float = _overlay_blend(base_float, colormap_float)
        
        # Expand mask to 3 channels
        mask_expanded = cv2.merge([mask, mask, mask])
        
        # Composite: (original * (1 - mask)) + (overlay * mask)
        final_float = (base_float * (1.0 - mask_expanded)) + (overlay_blend_float * mask_expanded)

        # Convert back to uint8 [0, 255]
        final_bgr = (np.clip(final_float, 0, 1) * 255.0).astype(np.uint8)

        # Convert BGR (OpenCV) to RGB (PIL)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_rgb)

    except IOError as e:
        print(f"Error in apply_heatmap_overlay: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in apply_heatmap_overlay: {e}")
        return None

def apply_heatmap_shadow(original_image_path,
                         heatmap_array,
                         shadow_color=(70, 70, 70),
                         contrast_factor=2.5,
                         max_shadow_opacity=0.85):
    """
    [VERSION 2: SHADOW/SPOTLIGHT EFFECT - UPDATED]
    Applies a shadow layer over the map. The heatmap score determines
    the opacity of the shadow. High scores are clear (0% shadow),
    low scores are dark (max_shadow_opacity).

    Args:
        original_image_path (str): Path to the original map image.
        heatmap_array (np.ndarray): The raw heatmap array from create_heatmap.
        shadow_color (tuple): The BGR color of the shadow. Default is dark gray.
        contrast_factor (float): The exponent applied to the mask to increase
                                 visual contrast. > 1.0 increases contrast.
        max_shadow_opacity (float): The opacity of the shadow at the lowest score
                                    (0.0 to 1.0). e.g., 0.85 = 85% opaque.
    Returns:
        PIL.Image: The final image with the shadow effect, or None on error.
    """
    try:
        # 1. Read the original image (the base layer)
        image_bgr = cv2.imread(original_image_path)
        if image_bgr is None:
            raise IOError(f"OpenCV could not read image: {original_image_path}")
        
        h, w, _ = image_bgr.shape
        
        # 2. Normalize the heatmap to [0, 1]
        heatmap_min = heatmap_array.min()
        heatmap_max = heatmap_array.max()
        epsilon = 1e-6
        
        if (heatmap_max - heatmap_min) < epsilon:
            mask = np.zeros_like(heatmap_array, dtype=np.float32)
        else:
            mask = (heatmap_array - heatmap_min) / (heatmap_max - heatmap_min + epsilon)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3. --- NEW: Apply Contrast (Gamma) ---
        # This is the key to making the gradient visible
        # (mask=0.3) ** 2.5 = 0.049  (gets much darker)
        # (mask=0.8) ** 2.5 = 0.57   (stays bright)
        mask = mask ** contrast_factor

        # 4. --- NEW: Calculate Shadow Alpha ---
        # Invert the mask and apply max opacity
        # shadow_alpha = (1.0 - mask) * max_shadow_opacity
        shadow_alpha = (1.0 - mask) * max_shadow_opacity
        
        # Expand alpha mask from (h, w) to (h, w, 3)
        shadow_alpha_expanded = cv2.merge([shadow_alpha, shadow_alpha, shadow_alpha])

        # 5. Create the shadow layer
        image_float = image_bgr.astype(np.float32) / 255.0
        shadow_color_float = (shadow_color[0]/255.0, 
                              shadow_color[1]/255.0, 
                              shadow_color[2]/255.0)
        shadow_layer_float = np.full_like(image_float, shadow_color_float)

        # 6. Blend the shadow layer onto the image
        # final = (shadow_layer * shadow_alpha) + (image * (1 - shadow_alpha))
        final_float = cv2.add(
            cv2.multiply(shadow_layer_float, shadow_alpha_expanded),
            cv2.multiply(image_float, 1.0 - shadow_alpha_expanded)
        )

        # 7. Convert back to uint8 [0, 255]
        final_bgr = (np.clip(final_float, 0, 1) * 255.0).astype(np.uint8)

        # 8. Convert BGR (OpenCV) to RGB (PIL)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_rgb)

    except IOError as e:
        print(f"Error in apply_heatmap_shadow: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in apply_heatmap_shadow: {e}")
        return None
    

def apply_heatmap_colorize(original_image_path,
                           heatmap_array,
                           colormap=cv2.COLORMAP_INFERNO,
                           contrast_factor=2.5):
    """
    [VERSION 3: GRAYSCALE COLORIZE]
    Blends a color heatmap onto a grayscale version of the original map.
    The heatmap score acts as the alpha, creating a very clear gradient
    from grayscale (low score) to full color (high score).

    Args:
        original_image_path (str): Path to the original map image.
        heatmap_array (np.ndarray): The raw heatmap array from create_heatmap.
        colormap (int): The OpenCV colormap to use (e.g., cv2.COLORMAP_INFERNO).
        contrast_factor (float): Exponent applied to the mask to increase
                                 visual contrast. > 1.0 increases contrast.
    Returns:
        PIL.Image: The final image with the colorize effect, or None on error.
    """
    try:
        # 1. Read the original image
        image_bgr = cv2.imread(original_image_path)
        if image_bgr is None:
            raise IOError(f"OpenCV could not read image: {original_image_path}")
        
        h, w, _ = image_bgr.shape

        # 2. --- NEW: Create Grayscale Base Layer ---
        # Convert to grayscale, then convert back to 3-channel BGR
        # so we can blend it with the color heatmap
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        image_gray_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        # 3. Normalize the heatmap to [0, 1]
        heatmap_min = heatmap_array.min()
        heatmap_max = heatmap_array.max()
        epsilon = 1e-6
        
        if (heatmap_max - heatmap_min) < epsilon:
            mask = np.zeros_like(heatmap_array, dtype=np.float32)
        else:
            mask = (heatmap_array - heatmap_min) / (heatmap_max - heatmap_min + epsilon)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. Create the Color Layer
        # We use the *original* mask for the colormap to get a smooth color transition
        heatmap_8bit = (mask * 255).astype(np.uint8)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_8bit, colormap)

        # 5. Create the Blending Mask (with contrast)
        # We use the *contrasted* mask for blending
        blending_mask = mask ** contrast_factor
        blending_mask_expanded = cv2.merge([blending_mask, blending_mask, blending_mask])

        # 6. Convert to float for blending math
        base_float = image_gray_bgr.astype(np.float32) / 255.0
        color_float = heatmap_color_bgr.astype(np.float32) / 255.0
        
        # 7. Blend
        # final = (grayscale_base * (1.0 - blending_mask)) + (color_heatmap * blending_mask)
        final_float = cv2.add(
            cv2.multiply(base_float, 1.0 - blending_mask_expanded),
            cv2.multiply(color_float, blending_mask_expanded)
        )

        # 8. Convert back to uint8 [0, 255]
        final_bgr = (np.clip(final_float, 0, 1) * 255.0).astype(np.uint8)

        # 9. Convert BGR (OpenCV) to RGB (PIL)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_rgb)

    except IOError as e:
        print(f"Error in apply_heatmap_colorize: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in apply_heatmap_colorize: {e}")
        return None
    
def apply_heatmap_hsv_colorize(original_image_path,
                               heatmap_array,
                               colormap=cv2.COLORMAP_INFERNO,
                               contrast_factor=2.0):
    """
    [VERSION 4: HSV COLORIZE (Detail Preserving)]
    Colorizes a grayscale version of the map using HSV logic.
    - Hue (Color) comes from the colormap.
    - Saturation (Vibrancy) comes from the heatmap score.
    - Value (Brightness/Lines) comes from the original grayscale image.
    This preserves lines/details even at 100% score.

    Args:
        original_image_path (str): Path to the original map image.
        heatmap_array (np.ndarray): The raw heatmap array from create_heatmap.
        colormap (int): The OpenCV colormap to use (e.g., cv2.COLORMAP_INFERNO).
        contrast_factor (float): Exponent applied to the saturation mask.
    Returns:
        PIL.Image: The final image with the HSV colorize effect, or None on error.
    """
    try:
        # 1. Read the original image
        image_bgr = cv2.imread(original_image_path)
        if image_bgr is None:
            raise IOError(f"OpenCV could not read image: {original_image_path}")
        
        h, w, _ = image_bgr.shape

        # 2. --- Create the V (Value) Channel ---
        # This is the grayscale map, which preserves the lines and texture.
        v_channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Normalize the heatmap to [0, 1]
        heatmap_min = heatmap_array.min()
        heatmap_max = heatmap_array.max()
        epsilon = 1e-6
        
        if (heatmap_max - heatmap_min) < epsilon:
            mask = np.zeros_like(heatmap_array, dtype=np.float32)
        else:
            mask = (heatmap_array - heatmap_min) / (heatmap_max - heatmap_min + epsilon)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. --- Create the H (Hue) Channel ---
        # We get this from the colormap
        heatmap_8bit_hue = (mask * 255).astype(np.uint8)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_8bit_hue, colormap)
        heatmap_color_hsv = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2HSV)
        h_channel, _, _ = cv2.split(heatmap_color_hsv)

        # 5. --- Create the S (Saturation) Channel ---
        # We get this from the contrasted mask
        s_mask = mask ** contrast_factor
        s_channel = (s_mask * 255).astype(np.uint8)

        # 6. Merge the H, S, V channels
        final_hsv = cv2.merge([h_channel, s_channel, v_channel])

        # 7. Convert back to BGR, then to RGB
        final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(final_rgb)

    except IOError as e:
        print(f"Error in apply_heatmap_hsv_colorize: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in apply_heatmap_hsv_colorize: {e}")
        return None