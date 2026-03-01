import os
import faiss
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def load_and_align_image(image_path, model, max_res=2048):
    """
    External Utility: Downscales the image if necessary and aligns the original image
    dimensions to be perfectly divisible by the patch size required by the model.
    The resulting image serves as the standardized base for all subsequent indexing and visualization.
    """
    img = Image.open(image_path).convert("RGB")

    # Scale down to prevent Out of Memory (OOM) errors
    img.thumbnail((max_res, max_res), Image.Resampling.LANCZOS)

    # Determine the nearest supported resolution perfectly divisible by the model's patch size
    dummy_tensor = torch.zeros(1, 3, img.height, img.width)
    nearest_res = model.get_nearest_supported_resolution(*dummy_tensor.shape[-2:])

    # Resize the image to the computed dimensions (nearest_res is H, W; PIL requires W, H)
    aligned_img = img.resize((nearest_res[1], nearest_res[0]), Image.Resampling.LANCZOS)

    print(f"Image aligned: Original dimensions {img.size} -> Aligned dimensions {aligned_img.size}")
    return aligned_img


class TestPipeline:
    def __init__(self, model_version="c-radio_v4-h", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dev = "cuda" if "cuda" in self.device else "cpu"

        print(f"Initializing RADIO Model (version={model_version}, device={self.device})...")
        self.model = torch.hub.load(
            'NVlabs/RADIO', 'radio_model',
            version=model_version, progress=True,
            skip_validation=True, adaptor_names=['siglip2-g'],
            force_reload=False
        ).to(self.device).eval()

        self.sig2_adaptor = self.model.adaptors['siglip2-g']
        self.patch_size = self.model.patch_size

        # Dynamically determine the text embedding dimension
        with torch.no_grad():
            dummy = self.sig2_adaptor.tokenizer(["test"]).to(self.device)
            dummy_vec = self.sig2_adaptor.encode_text(dummy, normalize=True)
            self.text_dim = dummy_vec.shape[-1]

        # Initialize the FAISS index
        self.dim = self.text_dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self.image_db = {}

        print(
            f"Model initialization complete. Shared latent space dimension: {self.dim}, Patch size: {self.patch_size}")

    @torch.no_grad()
    def extract_and_index(self, image_id, aligned_img):
        """
        Extract dense spatial features and construct the FAISS index.
        The input image is assumed to be pre-aligned.
        """

        # Convert the aligned image to a tensor
        x = pil_to_tensor(aligned_img).to(dtype=torch.float32, device=self.device).div_(255.0).unsqueeze_(0)

        # Calculate the feature grid dimensions
        h_feat = aligned_img.height // self.patch_size
        w_feat = aligned_img.width // self.patch_size
        print(f"Indexing '{image_id}': Dense grid size {h_feat}x{w_feat} ({h_feat * w_feat} patches)")

        # Execute forward pass
        with torch.autocast(self.autocast_dev, dtype=torch.bfloat16):
            vis_output = self.model(x)
            siglip_out = vis_output['siglip2-g']
            spatial_features = siglip_out.features if hasattr(siglip_out, 'features') else siglip_out[1]

        # Restore the spatial grid layout
        grid_feats = spatial_features.squeeze(0).reshape(h_feat, w_feat, self.dim)
        grid_feats = F.normalize(grid_feats, p=2, dim=-1).to(torch.float32).cpu().numpy()

        # Insert features into FAISS index
        flat_feats = np.ascontiguousarray(grid_feats.reshape(-1, self.dim))
        self.index.add(flat_feats)

        # Construct metadata for coordinate mapping
        for i in range(h_feat):
            for j in range(w_feat):
                self.metadata.append({'image_id': image_id, 'h_idx': i, 'w_idx': j})

        # Cache the original image and grid features for visualization
        self.image_db[image_id] = {
            'original_img': aligned_img,
            'grid_features': grid_feats
        }

        print(f"'{image_id}' indexed successfully. Total FAISS index capacity: {self.index.ntotal} patches.\n")

    @torch.no_grad()
    def visualize_top_k_on_image(self, query_text, top_k=5):
        """
        Retrieves the top K scoring patches globally based on cosine similarity,
        draws bounding boxes directly on the target image, and computes
        summary statistics for the similarity distribution.
        """
        if self.index.ntotal == 0:
            print("Error: FAISS index is empty. Please extract and index an image first.")
            return

        print(f"Retrieving top {top_k} regions for query: '{query_text}'...")
        text_input = self.sig2_adaptor.tokenizer([query_text]).to(self.device)
        text_vec = self.sig2_adaptor.encode_text(text_input, normalize=True).cpu().numpy().astype('float32')

        # Query FAISS index
        D, I = self.index.search(text_vec, k=top_k)

        # Group the retrieved results by image identifier
        hits_by_image = {}
        for rank, (score, flat_idx) in enumerate(zip(D[0], I[0])):
            meta = self.metadata[flat_idx]
            img_id = meta['image_id']
            if img_id not in hits_by_image:
                hits_by_image[img_id] = []
            hits_by_image[img_id].append((rank + 1, score, meta['h_idx'], meta['w_idx']))

        # Process and visualize bounding boxes per image
        for img_id, hits in hits_by_image.items():
            data = self.image_db[img_id]
            img = data['original_img']
            grid = data['grid_features']
            h_feat, w_feat, _ = grid.shape

            # Compute and output full-image similarity statistics
            similarity_map = np.dot(grid, text_vec.T).squeeze(-1)
            sim_max = similarity_map.max()
            sim_min = similarity_map.min()
            sim_mean = similarity_map.mean()

            print(f"   [Statistics for '{img_id}'] Max: {sim_max:.4f} | Min: {sim_min:.4f} | Mean: {sim_mean:.4f}")

            # Compute the absolute pixel dimensions of a single patch
            patch_w_px = img.width / w_feat
            patch_h_px = img.height / h_feat

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(img)
            ax.set_title(
                f"Top {top_k} Matches for Query: '{query_text}'\n(Displaying {len(hits)} matches in '{img_id}')",
                fontsize=14)
            ax.axis('off')

            # Render bounding boxes and rank labels
            for rank, score, h_idx, w_idx in hits:
                rect_x = w_idx * patch_w_px
                rect_y = h_idx * patch_h_px

                rect = mpatches.Rectangle(
                    (rect_x, rect_y), patch_w_px, patch_h_px,
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

                ax.text(rect_x, rect_y - 3, f"#{rank} ({score:.2f})",
                        color='red', fontsize=10, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

            plt.tight_layout()
            plt.show()

    @torch.no_grad()
    def visualize_heatmap(self, query_text):
        """
        Computes the cosine similarity between the query text and all image patches,
        and overlays the raw similarity scores as a heatmap on a black and white original image.
        """
        if not self.image_db:
            print("Error: Image database is empty. Please extract and index an image first.")
            return

        print(f"Generating full-image heatmap for query: '{query_text}'...")
        text_input = self.sig2_adaptor.tokenizer([query_text]).to(self.device)
        text_vec = self.sig2_adaptor.encode_text(text_input, normalize=True).cpu().numpy().astype('float32')

        for img_id, data in self.image_db.items():
            img = data['original_img']
            grid = data['grid_features']

            similarity_map = np.dot(grid, text_vec.T).squeeze(-1)

            sim_tensor = torch.tensor(similarity_map).unsqueeze(0).unsqueeze(0)
            sim_resized = F.interpolate(
                sim_tensor, size=(img.height, img.width),
                mode='nearest'
            ).squeeze().numpy()

            # Convert the original image to black and white
            bw_img = img.convert("L").convert("RGB")

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left plot: Black and white original image
            axes[0].imshow(bw_img)
            axes[0].set_title(f"Original Image: '{img_id}'", fontsize=14)
            axes[0].axis('off')

            # Right plot: Heatmap overlay on the black and white image
            axes[1].imshow(bw_img)
            hm = axes[1].imshow(sim_resized, cmap='YlOrRd', alpha=0.6)
            axes[1].set_title(f"Heatmap\nQuery: '{query_text}'", fontsize=14)
            axes[1].axis('off')

            cbar = plt.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label('Raw Cosine Score', rotation=270, labelpad=15)

            plt.tight_layout()
            plt.show()

    @torch.no_grad()
    def visualize_pca(self, image_id, remove_background=True):
        """
        Applies Principal Component Analysis (PCA) to the high-dimensional patch embeddings,
        reducing them to 3 dimensions and mapping them to RGB channels.
        Optionally removes the background by thresholding the first principal component.
        """
        from sklearn.decomposition import PCA

        if image_id not in self.image_db:
            print(f"Error: '{image_id}' not found in the database. Please index it first.")
            return

        print(f"Generating PCA visualization for '{image_id}'...")
        data = self.image_db[image_id]
        img = data['original_img']
        grid = data['grid_features']
        h_feat, w_feat, dim = grid.shape

        # 1. Flatten the spatial grid to a list of patch vectors
        flat_grid = grid.reshape(-1, dim)

        # 2. Fit PCA and extract the top 3 principal components
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(flat_grid)

        # 3. Normalize the 3 components to [0, 1] to serve as R, G, B channels
        pca_min = pca_features.min(axis=0)
        pca_max = pca_features.max(axis=0)
        pca_normalized = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)

        # Reshape back to the 2D spatial grid
        pca_rgb_grid = pca_normalized.reshape(h_feat, w_feat, 3)

        # 4. Background Removal Logic (Thresholding the 1st PCA component)
        if remove_background:
            # The first principal component (PC1) usually captures the most variance,
            # which perfectly correlates with the foreground/background separation.
            # pc1_grid = pca_normalized[:, :, 0]
            pc1_grid = pca_rgb_grid[:, :, 0]

            # Dynamic Heuristic: We assume the 4 corners of the image belong to the background.
            # We calculate the mean value of the corners to determine the background's polarity in PC1.
            corner_mean = (pc1_grid[0, 0] + pc1_grid[0, -1] + pc1_grid[-1, 0] + pc1_grid[-1, -1]) / 4.0
            global_mean = pc1_grid.mean()

            # Create a boolean mask where True = Foreground, False = Background
            if corner_mean > global_mean:
                mask = pc1_grid < global_mean
            else:
                mask = pc1_grid > global_mean

            # Apply the mask: set background pixels to pure black
            pca_rgb_grid[~mask] = [0, 0, 0]

        # 5. Interpolate the PCA grid to match the original image dimensions
        # 'nearest' interpolation preserves the distinct 16x16 patch resolution structure
        pca_tensor = torch.tensor(pca_rgb_grid).permute(2, 0, 1).unsqueeze(0)
        pca_resized = F.interpolate(
            pca_tensor, size=(img.height, img.width), mode='nearest'
        ).squeeze(0).permute(1, 2, 0).numpy()

        # 6. Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Original image
        axes[0].imshow(img)
        axes[0].set_title(f"Original: '{image_id}'", fontsize=14)
        axes[0].axis('off')

        # Right plot: PCA RGB map
        bg_status = "Removed" if remove_background else "Included"
        axes[1].imshow(pca_resized)
        axes[1].set_title(f"PCA Embeddings \nBackground: {bg_status}", fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # max_resolution = 2048
    # patch_size = 16
    # dimension of embedding = 1536
    pipeline = TestPipeline()

    img_path = "data/raw_maps/kids.jpg"
    img_path = "data/raw_maps/animal.jpg"
    img_path = "data/raw_maps/football.png"

    # Process and align the source image (only for test)
    aligned_img = load_and_align_image(img_path, pipeline.model, max_res=2048)

    # Insert features into the index
    pipeline.extract_and_index("image_01", aligned_img)

    test_queries = [
        "football",
        "a round soccer ball rolling on the grass"
    ]

    for query in test_queries:
        pipeline.visualize_top_k_on_image(query, top_k=15)
        pipeline.visualize_heatmap(query)

    pipeline.visualize_pca("image_01", remove_background=True)
