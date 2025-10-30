# src/search_engine.py
import faiss
import numpy as np

class FaissSearch:
    def __init__(self, feature_dim):
        """
        Initializes the FAISS search engine.

        Args:
            feature_dim (int): The dimensionality of the feature vectors
                               (e.g., 512 for CLIP-base).
        """
        self.feature_dim = feature_dim

        # Use IndexFlatIP (Inner Product) as the index.
        # Because our vectors are L2-normalized, Inner Product is equivalent
        # to Cosine Similarity.
        self.index = faiss.IndexFlatIP(feature_dim)

        # Check if faiss-gpu is installed and available
        if faiss.get_num_gpus() > 0:
            print("FAISS: GPU support detected. Moving index to GPU.")
            # Move the index to all available GPUs
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        else:
            print("FAISS: No GPU support detected. Using CPU.")

    def build_index(self, features):
        """
        Builds (adds) the feature vectors to the index.

        Args:
            features (np.ndarray): A numpy array of features with shape
                                   (num_patches, feature_dim).
                                   Must be float32 and normalized!
        """
        # Note: IndexFlatIP does not require training, so is_trained is always true.
        # We can skip the check 'if self.index.is_trained:'.

        print(f"Building index with {features.shape[0]} vectors...")
        self.index.add(features.astype('float32'))
        print(f"Index built. Total vectors in index: {self.index.ntotal}")

    def search(self, query_vector, k=6):
        """
        Searches the index for the K nearest neighbors.

        Args:
            query_vector (np.ndarray): The query vector of shape (1, feature_dim).
                                       Must be float32 and normalized!
            k (int): The number of neighbors to find.

        Returns:
            tuple: (distances, indices)
                   distances: Similarity scores (between 0 and 1, higher is better)
                   indices: The indices of the results in the original dataset.
        """
        if self.index.ntotal == 0:
            print("Error: Index is empty. Please build the index first.")
            return None, None

        # Ensure the query vector is float32
        query_vector = query_vector.astype('float32')

        # D (distances), I (indices)
        distances, indices = self.index.search(query_vector, k)

        return distances, indices