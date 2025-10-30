# src/feature_extractor.py
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
import numpy as np

class FeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initializes the feature extractor with a specified CLIP model.

        Args:
            model_name (str): The model name from the Hugging Face Hub.
            device (str): "cuda" or "cpu". Auto-detects if None.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"FeatureExtractor using device: {self.device}")

        # Load the model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode

    def _normalize_features(self, features):
        """L2 normalize feature vectors"""
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / norm

    @torch.no_grad()
    def extract_image_features(self, patch_images, batch_size=64):
        """
        Extracts features for a batch of image patches.

        Args:
            patch_images (list): A list of PIL.Image objects.
            batch_size (int): The batch size for processing.

        Returns:
            np.ndarray: A numpy array of normalized feature vectors.
        """
        all_features = []
        print("Extracting image features using CLIP...")

        for i in tqdm(range(0, len(patch_images), batch_size), desc="Extracting Features"):
            batch = patch_images[i:i + batch_size]

            # Preprocess images
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

            # Get features
            image_features = self.model.get_image_features(**inputs)

            # Move to CPU, convert to numpy
            all_features.append(image_features.cpu().numpy())

        features_array = np.vstack(all_features)

        # Normalize features, crucial for IndexFlatIP (cosine similarity)
        return self._normalize_features(features_array)

    @torch.no_grad()
    def extract_text_features(self, text_query):
        """
        Extracts features for a single text query.

        Args:
            text_query (str): The text query string.

        Returns:
            np.ndarray: A normalized 1D feature vector of shape (1, feature_dim).
        """
        print(f"Extracting text features for: '{text_query}'")
        inputs = self.processor(text=[text_query], return_tensors="pt").to(self.device)

        text_features = self.model.get_text_features(**inputs)

        features_array = text_features.cpu().numpy()

        # Normalize features
        return self._normalize_features(features_array)

