# src/feature_extractor.py
import torch
from transformers import AutoProcessor, AutoModel
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

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


class RadioFeatureExtractor:
    def __init__(self, model_version="c-radio_v4-h", device=None, default_clusters=8):
        """
        基于官方 RADIO v4 示例实现的特征提取器。

        Args:
            model_version: 默认使用 "c-radio_v4-h"
            device: 指定运行设备
            default_clusters: 默认聚类中心数量
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_clusters = default_clusters
        self.vector_dim = 1280  # C-RADIO-v4-H 的固定维度

        print(f"🚀 加载 RADIO 模型: {model_version}...")
        # 官方 Example 5: 加载模型并指定 siglip2-g 适配器进行文本对齐
        self.model = torch.hub.load(
            'NVlabs/RADIO',
            'radio_model',
            version=model_version,
            progress=True,
            skip_validation=True,
            adaptor_names=['siglip2-g'],
            force_reload=False
        ).to(self.device).eval()

        # 获取适配器用于文本处理
        self.sig2_adaptor = self.model.adaptors['siglip2-g']
        print(f"✅ RADIO 初始化成功，显存占用已优化。")

    @torch.no_grad()
    def extract_dense_features(self, pil_image):
        """
        方法 1: 提取原始稠密特征 (Dense, pixelwise features)
        已修复 NVIDIA 官方 NCHW + Adaptor 导致的 'upsample_factor' 报错。
        """
        # 预处理
        x = pil_to_tensor(pil_image).to(dtype=torch.float32, device=self.device)
        x.div_(255.0).unsqueeze_(0)

        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

        # 获取高度和宽度上的 Patch 数量 (RADIO patch_size 通常为 16)
        patch_size = self.model.patch_size
        h_feat = x.shape[-2] // patch_size
        w_feat = x.shape[-1] // patch_size

        # 核心修改点：
        # 1. 不传 feature_fmt='NCHW'，让它默认返回 sequence 格式 (N, L, C)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            vis_output = self.model(x)  # 删除了 feature_fmt 参数

            # backbone 空间特征现在的形状是 [1, h_feat * w_feat, 1280]
            _, spatial_features = vis_output['backbone']

        # 2. 我们手动把它展平，其实 NLC 格式正好已经帮我们展平了空间维度！
        # 直接拿来用即可，不需要 permute 和 reshape 了
        dense_feats = spatial_features.squeeze(0)  # 形状变为 [h_feat * w_feat, 1280]

        return dense_feats.to(torch.float32)

    # @torch.no_grad()
    # def extract_dense_features(self, pil_image):
    #     """
    #     方法 1: 提取原始稠密特征 (Dense, pixelwise features)
    #     符合官方 Example 1 & 2。
    #     """
    #     # 官方预处理流程
    #     x = pil_to_tensor(pil_image).to(dtype=torch.float32, device=self.device)
    #     x.div_(255.0).unsqueeze_(0)
    #
    #     # 官方逻辑：调整到最近的受支持分辨率
    #     nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
    #     x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
    #
    #     # 官方逻辑：使用混合精度 (autocast) 和 NCHW 格式
    #     with torch.autocast(self.device, dtype=torch.bfloat16):
    #         vis_output = self.model(x, feature_fmt='NCHW')
    #         # 提取 backbone 空间特征 [1, 1280, H/16, W/16]
    #         _, spatial_features = vis_output['backbone']
    #
    #     # 展平特征 [1, 1280, H, W] -> [H*W, 1280]
    #     c = spatial_features.shape[1]
    #     dense_feats = spatial_features.squeeze(0).permute(1, 2, 0).reshape(-1, c)
    #
    #     return dense_feats.to(torch.float32)  # 返回 GPU 上的 Float32 Tensor

    def cluster_features(self, dense_features, num_clusters=None):
        """
        方法 2: 将稠密特征进行聚类 (Clusters of features)
        Alex 建议的核心步骤：使用 Spherical K-means 压缩信息。
        """
        k = num_clusters if num_clusters else self.default_clusters

        # 确保数据在 CPU 上供 sklearn 使用
        if torch.is_tensor(dense_features):
            dense_features = dense_features.cpu().numpy()

        # Spherical K-Means 核心逻辑
        # 1. 归一化输入（投影到单位超球面）
        norm_feats = normalize(dense_features, axis=1)

        # 2. K-means 聚类
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(norm_feats)

        # 3. 归一化中心点，确保最终特征向量也是单位向量
        centroids = normalize(kmeans.cluster_centers_, axis=1)
        return centroids  # 返回 (k, 1280) 的 numpy 数组

    @torch.no_grad()
    def extract_text_features(self, text_query):
        """
        方法 3: 提取文本特征
        官方 Example 5。
        """
        text_input = self.sig2_adaptor.tokenizer([text_query]).to(self.device)
        text_tokens = self.sig2_adaptor.encode_text(text_input, normalize=True)
        return text_tokens.cpu().numpy()

    @staticmethod
    def compute_similarity(query_clusters, target_clusters):
        """
        匹配逻辑：比较两组聚类中心。
        """
        # 计算 8x8 或 MxN 的相似度矩阵
        sim_matrix = np.dot(query_clusters, target_clusters.T)
        # 多对多最大值匹配 (Chamfer Similarity)
        return np.mean(np.max(sim_matrix, axis=1))


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

        # 2. Get the image and processed_text preprocessing utilities
        self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)
        print(f"Model {model_name} loaded successfully.")

    def _normalize_features(self, features):
        """L2-normalize feature vectors."""
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / norm

    @torch.no_grad()
    @torch.autocast("cuda")
    def extract_text_features_batch(self, text_queries, batch_size=128):
        all_features = []
        for i in range(0, len(text_queries), batch_size):
            batch = text_queries[i:i + batch_size]
            text_tensor = self.tokenizer(batch).to(self.device)
            text_features = self.model.encode_text(text_tensor)
            all_features.append(text_features.cpu().numpy())
        return np.vstack(all_features)

    @torch.no_grad()
    @torch.autocast("cuda")
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
        Extract PE core features for a single processed_text query.

        Args:
            text_query (str): The processed_text query string.

        Returns:
            np.ndarray: A normalized (1, feature_dim) feature vector.
        """
        # print(f"Extracting processed_text features for: '{text_query}'")

        # 1. Tokenize: convert processed_text to tensor
        text_tensor = self.tokenizer([text_query]).to(self.device)

        # 2. Encode processed_text
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
        Extracts features for a single processed_text query.

        Args:
            text_query (str): The processed_text query string.

        Returns:
            np.ndarray: A normalized 1D feature vector of shape (1, feature_dim).
        """
        print(f"Extracting processed_text features for: '{text_query}'")
        inputs = self.processor(text=[text_query], return_tensors="pt").to(self.device)

        text_features = self.model.get_text_features(**inputs)

        features_array = text_features.cpu().numpy()

        # Normalize features
        return self._normalize_features(features_array)
