# src/feature_extractor.py
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
import numpy as np


try:
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as pe_transforms
except ImportError:
    print("=" * 50)
    print("ERROR: 'perception_models' or 'open_clip' is not installed.")
    print("Please activate your environment and run the following commands:")
    print("pip install open_clip_torch")
    print("pip install git+https://github.com/facebookresearch/perception_models.git")
    print("=" * 50)
    raise


class PEFeatureExtractor:
    def __init__(self, model_name="PE-Core-B16-224", device=None):
        """
        Initialize the Perception Encoder (PE) core CLIP model.

        Args:
            model_name (str): The full Hugging Face Hub ID of the PE core model.
            device (str): "cuda" or "cpu".
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"FeatureExtractor (Perception Encoder) using device: {self.device}")

        self.model_name = model_name

        # 1. Load the PE core CLIP model
        try:
            self.model = pe.CLIP.from_config(model_name, pretrained=True)
        except OSError:
            print(f"'{model_name}' failed with from_config... trying 'from_pretrained' instead...")
            try:
                hf_hub_id = f"hf-hub:{model_name}"
                print(f"Loading from Hugging Face Hub, ID: {hf_hub_id}")
                self.model = pe.CLIP.from_pretrained(hf_hub_id)

            except Exception as e:
                print(f"ERROR: Failed to load model {model_name}.")
                print(f"Detailed error: {e}")
                raise

        self.model = self.model.to(self.device)
        self.model.eval()

        # 2. Get the image and text preprocessing utilities
        self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)
        print(f"Model {model_name} loaded successfully. Image size: {self.model.image_size}px")

    def _normalize_features(self, features):
        """L2-normalize feature vectors."""
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / norm

    @torch.no_grad()
    @torch.autocast("cuda")  # Use mixed precision for faster inference
    def extract_image_features(self, patch_images, batch_size=64):
        """
        Extract PE core features for batches of image patches.

        Args:
            patch_images (list): A list of PIL.Image objects.
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: L2-normalized feature vectors.
        """
        all_features = []
        print(f"Extracting image features using {self.model_name}...")

        for i in tqdm(range(0, len(patch_images), batch_size), desc="Extracting PE Features"):
            batch = patch_images[i:i + batch_size]

            # 1. Preprocess: convert a list of PIL images to a tensor batch
            image_tensors = [self.preprocess(img) for img in batch]
            image_batch = torch.stack(image_tensors).to(self.device)

            # 2. Encode images
            image_features = self.model.encode_image(image_batch)

            all_features.append(image_features.cpu().numpy())

        features_array = np.vstack(all_features)

        # Normalize features
        return self._normalize_features(features_array)

    @torch.no_grad()
    @torch.autocast("cuda")
    def extract_text_features(self, text_query):
        """
        Extract PE core features for a single text query.

        Args:
            text_query (str): The text query string.

        Returns:
            np.ndarray: A normalized (1, feature_dim) feature vector.
        """
        print(f"Extracting text features for: '{text_query}'")

        # 1. Tokenize: convert text to tensor
        text_tensor = self.tokenizer([text_query]).to(self.device)

        # 2. Encode text
        text_features = self.model.encode_text(text_tensor)

        features_array = text_features.cpu().numpy()

        # Normalize features
        return self._normalize_features(features_array)



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
