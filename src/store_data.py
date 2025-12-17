import uuid
import json
import pandas as pd
from tqdm import tqdm
from shapely import wkt
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
)


class GeoVectorDB:
    def __init__(self, host="localhost", port=6333,
                 map_collection_name="venice_historical_map_2",
                 doc_collection_name="venice_historical_text_2"):
        """
        Initialize the Vector Database Client.
        Detects if 'host' is a local path or a remote URL.
        """
        if host.startswith(".") or "/" in host or "\\" in host:
            # Local persistence mode
            self.client = QdrantClient(path=host)
        else:
            # Server client mode
            self.client = QdrantClient(host=host, port=port)

        self.MAP_COLLECTION = map_collection_name
        self.DOC_COLLECTION = doc_collection_name

    def init_map_collection(self, vector_dim=1024, force_recreate=False):
        """
        Initialize the main map image collection.
        Configures quantization, HNSW parameters, and payload indexes.
        """
        # 1. Delete if force recreation is requested
        if force_recreate and self.client.collection_exists(self.MAP_COLLECTION):
            self.client.delete_collection(self.MAP_COLLECTION)
            print(f"Deleted old collection: {self.MAP_COLLECTION}")

        # 2. Create Collection
        if not self.client.collection_exists(self.MAP_COLLECTION):
            self.client.create_collection(
                collection_name=self.MAP_COLLECTION,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE,
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                        )
                    )
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    on_disk=False
                )
            )

            # --- Create Indexes ---

            # 2.1 Geo-spatial Index
            self.client.create_payload_index(
                collection_name=self.MAP_COLLECTION,
                field_name="location",
                field_schema=PayloadSchemaType.GEO
            )

            # 2.2 Year Index (Integer) -> optimization for filtering
            self.client.create_payload_index(
                collection_name=self.MAP_COLLECTION,
                field_name="year",
                field_schema=PayloadSchemaType.INTEGER
            )

            print(f"Collection '{self.MAP_COLLECTION}' created successfully.")
        else:
            print(f"Collection '{self.MAP_COLLECTION}' already exists.")

    def init_doc_collection(self, text_dim=384, pe_dim=1024, force_recreate=False):
        """
        Initialize the historical document collection.

        Vectors:
        1. text_vector: Semantic features (e.g., MiniLM) for Text-to-Text search.
        2. pe_vector: Visual features (optional) for Image-to-Text search.
        """
        if force_recreate and self.client.collection_exists(self.DOC_COLLECTION):
            self.client.delete_collection(self.DOC_COLLECTION)
            print(f"Deleted old document collection: {self.DOC_COLLECTION}")

        if not self.client.collection_exists(self.DOC_COLLECTION):
            self.client.create_collection(
                collection_name=self.DOC_COLLECTION,
                vectors_config={
                    # 1. Core: Text Semantic Vector
                    "text_vector": VectorParams(
                        size=text_dim,
                        distance=Distance.COSINE,
                        quantization_config=ScalarQuantization(
                            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
                        )
                    ),
                    # 2. Aux: PE Vector for Visual Alignment
                    "pe_vector": VectorParams(
                        size=pe_dim,
                        distance=Distance.COSINE,
                        quantization_config=ScalarQuantization(
                            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
                        )
                    )
                }
            )

            # --- Create Indexes ---
            self.client.create_payload_index(self.DOC_COLLECTION, "location", PayloadSchemaType.GEO)
            self.client.create_payload_index(self.DOC_COLLECTION, "year", PayloadSchemaType.INTEGER)
            self.client.create_payload_index(self.DOC_COLLECTION, "media_type", PayloadSchemaType.KEYWORD)

            print(f"Document collection '{self.DOC_COLLECTION}' created (Text Dim: {text_dim}, PE Dim: {pe_dim}).")
        else:
            print(f"Document collection '{self.DOC_COLLECTION}' already exists, skipping creation.")

    def ingest_map_data(self, image_features, registered_data_list, source_image_name, year, batch_size=100):
        """
        Ingest map image segments.
        :param year: The year of the map (int or str).
        """
        points_buffer = []
        total = len(image_features)

        # Ensure year is an integer
        try:
            year_int = int(year)
        except ValueError:
            year_int = 0
            print(f"Warning: Year '{year}' format is invalid, defaulting to 0.")

        print(f"Starting Ingest: {source_image_name} (Year: {year_int}) - {total} slices...")

        for idx, (emb, item) in tqdm(enumerate(zip(image_features, registered_data_list)), total=total):

            center_coords = item['geo_geometry']['wgs84']['center']
            lat, lon = center_coords[0], center_coords[1]

            # Construct Payload
            payload = {
                "source_image": source_image_name,
                "year": year_int,
                "orig_index": item.get('index', idx),
                "pixel_coords": item['pixel_coords'],
                "location": {
                    "lat": lat,
                    "lon": lon
                },
                "geo_detail": item['geo_geometry']
            }

            points_buffer.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload
            ))

            # Upsert Batch
            if len(points_buffer) >= batch_size:
                self.client.upsert(
                    collection_name=self.MAP_COLLECTION,
                    points=points_buffer
                )
                points_buffer = []

        # Upsert remaining
        if points_buffer:
            self.client.upsert(
                collection_name=self.MAP_COLLECTION,
                points=points_buffer
            )

        print(f"Ingestion complete! stored {total} items.")

    def ingest_text_data(self, df, pe_model=None, batch_size=1000, pe_batch_size=64):
        """
        Ingest processed_text documents from a DataFrame.
        Computes PE vectors if a model is provided.
        """
        points_buffer = []
        compute_pe = pe_model is not None

        if compute_pe:
            print("PE Model loaded. Generating visual alignment vectors (for Image-to-Text)...")
        else:
            print("No PE Model provided. Storing only Semantic Vectors (Text-to-Text only).")

        print(f"[Text] Starting Ingest: {len(df)} records...")

        # --- Pre-process and collect valid rows ---
        text_rows = []  # tuple: (idx, text_content, payload, vec_data)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing DataFrame"):
            try:
                if pd.isna(row.get('geometry')):
                    continue

                # Parse Geometry
                geo_obj = wkt.loads(row['geometry'])
                lon, lat = geo_obj.x, geo_obj.y

                # Parse Metadata
                meta = row.get('metadata', {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}

                # Determine Year (Dataset specific logic)
                year_int = 0
                source_ds = str(row.get('source_dataset', ''))
                if '1740' in source_ds:
                    year_int = 1740
                elif '1808' in source_ds:
                    year_int = 1808

                text_content = row.get('text_representation', '')

                payload = {
                    "source_dataset": source_ds,
                    "year": year_int,
                    "orig_index": row.get('original_id'),
                    "chunk_id": row.get('chunk_id'),
                    "content": text_content,
                    "location": {"lat": lat, "lon": lon},
                    "full_metadata": meta
                }

                # Parse Embedding
                vec_data = row.get('embedding')
                if isinstance(vec_data, str):
                    vec_data = json.loads(vec_data)
                elif hasattr(vec_data, 'tolist'):
                    vec_data = vec_data.tolist()

                text_rows.append((idx, text_content, payload, vec_data))

            except Exception as e:
                print(f"Skipped row {idx}: {e}")
                continue

        # --- Batch Generate PE Vectors and Upload ---
        for i in tqdm(range(0, len(text_rows), pe_batch_size), desc="Uploading Batches"):
            batch_rows = text_rows[i:i + pe_batch_size]
            texts = [t[1] for t in batch_rows]

            # Compute PE vectors if model exists
            if compute_pe:
                pe_vectors = pe_model.extract_text_features_batch(texts, batch_size=pe_batch_size)
            else:
                pe_vectors = [None] * len(batch_rows)

            # Construct Points
            for (idx, text_content, payload, vec_data), pe_vec in zip(batch_rows, pe_vectors):
                vectors = {}
                if vec_data:
                    vectors["text_vector"] = vec_data
                if pe_vec is not None:
                    vectors["pe_vector"] = pe_vec.tolist()

                points_buffer.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,
                    payload=payload
                ))

                # Upload to Qdrant
                if len(points_buffer) >= batch_size:
                    self.client.upsert(collection_name=self.DOC_COLLECTION, points=points_buffer)
                    points_buffer = []

        # Upload remaining
        if points_buffer:
            self.client.upsert(collection_name=self.DOC_COLLECTION, points=points_buffer)

        print("Text data ingestion complete!")
