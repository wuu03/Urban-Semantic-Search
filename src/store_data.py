import json
import uuid

from shapely import wkt
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, ScalarQuantization, ScalarQuantizationConfig,
    ScalarType, HnswConfigDiff, PointStruct, PayloadSchemaType
)
import pandas as pd


class GeoVectorDB:
    def __init__(self, host="localhost", port=6333, collection_name="venice_historical_map",
                 doc_collection_name="venice_historical_text_test"):
        if host.startswith(".") or "/" in host or "\\" in host:
            self.client = QdrantClient(path=host)
        else:
            self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.DOC_COLLECTION = doc_collection_name

    def init_collection(self, vector_dim=1024, force_recreate=False):
        # 1. 如果强制重建，先删除
        if force_recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        # 2. 创建集合
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
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

            # --- 🔥 修改点 1: 建立索引 ---

            # 2.1 建立地理位置索引 (Geo)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="location",
                field_schema=PayloadSchemaType.GEO
            )

            # 2.2 建立年份索引 (Integer) -> 让筛选速度飞快
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year",
                field_schema=PayloadSchemaType.INTEGER
            )

            print(f"⚡ 集合 '{self.collection_name}' 创建成功 (已开启 Geo 和 Year 索引)")
        else:
            print(f"👌 集合 '{self.collection_name}' 已存在")

        # ==========================================
        # 2. 初始化文献集合 (存历史文本)
        # ==========================================

    def init_doc_collection(self, text_dim=384, pe_dim=1024, force_recreate=False):
        """
        初始化历史文献集合。
        主要向量: text_vector (语义特征)
        辅助向量: pe_vector (视觉特征，用于实现'图搜文')
        """
        if force_recreate and self.client.collection_exists(self.DOC_COLLECTION):
            self.client.delete_collection(self.DOC_COLLECTION)
            print(f"🗑️ 已删除旧集合: {self.DOC_COLLECTION}")

        if not self.client.collection_exists(self.DOC_COLLECTION):
            self.client.create_collection(
                collection_name=self.DOC_COLLECTION,
                vectors_config={
                    # 1. 核心：MiniLM 向量，用于"文搜文"
                    "text_vector": VectorParams(
                        size=text_dim,
                        distance=Distance.COSINE,
                        quantization_config=ScalarQuantization(
                            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
                        )
                    ),
                    # 2. 辅助：PE 向量，用于"图搜文" (可选，建议加上)
                    "pe_vector": VectorParams(
                        size=pe_dim,
                        distance=Distance.COSINE,
                        quantization_config=ScalarQuantization(
                            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
                        )
                    )
                }
            )

            # --- 建立索引 ---
            self.client.create_payload_index(self.DOC_COLLECTION, "location", PayloadSchemaType.GEO)
            self.client.create_payload_index(self.DOC_COLLECTION, "year", PayloadSchemaType.INTEGER)
            self.client.create_payload_index(self.DOC_COLLECTION, "media_type", PayloadSchemaType.KEYWORD)

            print(f"✅ 文献集合 '{self.DOC_COLLECTION}' 创建成功 (Text Dim: {text_dim}, PE Dim: {pe_dim})")
        else:
            print(f"👌 文献集合 '{self.DOC_COLLECTION}' 已存在，跳过创建。")

    def ingest_data(self, image_features, registered_data_list, source_image_name, year, batch_size=100):
        """
        新增 year 参数
        :param year: 年份，可以是字符串 '1704' 或 整数 1704
        """
        points_buffer = []
        total = len(image_features)

        # --- 🔥 修改点 2: 确保年份是整数 ---
        try:
            year_int = int(year)
        except ValueError:
            year_int = 0  # 或者抛出错误，看你怎么处理脏数据
            print(f"⚠️ 警告: 年份 '{year}' 格式不正确，已默认为 0")

        print(f"🚀 开始入库: {source_image_name} (Year: {year_int}) - {total} 个切片...")

        for idx, (emb, item) in tqdm(enumerate(zip(image_features, registered_data_list)), total=total):

            center_coords = item['geo_geometry']['wgs84']['center']
            lat, lon = center_coords[0], center_coords[1]

            # --- 🔥 修改点 3: 写入 Payload ---
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

            if len(points_buffer) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points_buffer
                )
                points_buffer = []

        if points_buffer:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points_buffer
            )

        print(f"✅ 入库完成！已存储 {total} 条数据。")

    def ingest_text_data(self, df, pe_model=None, batch_size=1000, pe_batch_size=64):
        points_buffer = []
        compute_pe = pe_model is not None

        if compute_pe:
            print("🌊 PE 模型已加载，正在为文本生成视觉对齐向量 (用于图搜文)...")
        else:
            print("⚠️ 未提供 PE 模型，将只存储 MiniLM 语义向量 (仅支持文搜文)。")

        print(f"🚀 [Text] 开始入库: {len(df)} 条记录...")

        # --- 收集批量 PE 文本 ---
        text_rows = []  # [(idx, text_content, payload, vec_data)]
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                if pd.isna(row.get('geometry')):
                    continue
                geo_obj = wkt.loads(row['geometry'])
                lon, lat = geo_obj.x, geo_obj.y

                meta = row.get('metadata', {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except:
                        meta = {}

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

                vec_data = row.get('embedding')
                if isinstance(vec_data, str):
                    vec_data = json.loads(vec_data)
                elif hasattr(vec_data, 'tolist'):
                    vec_data = vec_data.tolist()

                text_rows.append((idx, text_content, payload, vec_data))

            except Exception as e:
                print(f"⚠️ 跳过第 {idx} 行: {e}")
                continue

        # --- 批量生成 PE 向量并上传 ---
        for i in tqdm(range(0, len(text_rows), pe_batch_size)):
            batch_rows = text_rows[i:i + pe_batch_size]
            texts = [t[1] for t in batch_rows]
            if compute_pe:
                pe_vectors = pe_model.extract_text_features_batch(texts, batch_size=pe_batch_size)
            else:
                pe_vectors = [None] * len(batch_rows)

            for (idx, text_content, payload, vec_data), pe_vec in zip(batch_rows, pe_vectors):
                vectors = {}
                if vec_data: vectors["text_vector"] = vec_data
                if pe_vec is not None: vectors["pe_vector"] = pe_vec.tolist()

                points_buffer.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,
                    payload=payload
                ))

                if len(points_buffer) >= batch_size:
                    self.client.upsert(collection_name=self.DOC_COLLECTION, points=points_buffer)
                    points_buffer = []

        # 上传剩余
        if points_buffer:
            self.client.upsert(collection_name=self.DOC_COLLECTION, points=points_buffer)

        print("✅ 文本数据入库完成！")

    # def ingest_text_data(self, df, pe_model=None, batch_size=100):
    #     """
    #     将 Pandas DataFrame 存入文献集合。
    #     :param df: 包含 text_representation, embedding, geometry, metadata 的 DataFrame
    #     :param pe_model: (可选) 传入 PE 模型实例。如果传入，会计算 pe_vector，实现"图搜文"。
    #     :param batch_size: 批处理大小
    #     """
    #     points_buffer = []
    #
    #     # 确保 pe_model 不为空才计算图文对齐向量
    #     compute_pe = pe_model is not None
    #     if compute_pe:
    #         print("🌊 PE 模型已加载，正在为文本生成视觉对齐向量 (用于图搜文)...")
    #     else:
    #         print("⚠️ 未提供 PE 模型，将只存储 MiniLM 语义向量 (仅支持文搜文)。")
    #
    #     print(f"🚀 [Text] 开始入库: {len(df)} 条记录...")
    #
    #     for idx, row in tqdm(df.iterrows(), total=len(df)):
    #         try:
    #             # --- A. 解析坐标 (WKT -> Lat/Lon) ---
    #             # 格式: POINT (12.329... 45.430...)
    #             # 如果 geometry 是空的或无效，跳过或设为默认值
    #             if pd.isna(row.get('geometry')):
    #                 continue
    #
    #             geo_obj = wkt.loads(row['geometry'])
    #             lon, lat = geo_obj.x, geo_obj.y
    #
    #             # --- B. 解析 Metadata ---
    #             # Parquet 读取出来的 metadata 可能是 JSON 字符串，也可能是 dict
    #             meta = row.get('metadata', {})
    #             if isinstance(meta, str):
    #                 try:
    #                     meta = json.loads(meta)
    #                 except:
    #                     meta = {}
    #
    #             # --- C. 准备 Payload ---
    #             # 提取年份 (尝试从 metadata 或 source_dataset 获取)
    #             year_int = 0
    #             source_ds = str(row.get('source_dataset', ''))
    #             if '1740' in source_ds:
    #                 year_int = 1740
    #             elif '1808' in source_ds:
    #                 year_int = 1808
    #
    #             text_content = row.get('text_representation', '')
    #
    #             payload = {
    #                 "source_dataset": source_ds,
    #                 "year": year_int,
    #                 "orig_index": row.get('original_id'),
    #                 "chunk_id": row.get('chunk_id'),
    #                 "content": text_content,  # 核心内容
    #                 "location": {"lat": lat, "lon": lon},
    #                 "full_metadata": meta  # 如果 metadata 太大，可以不存，或者只存关键字段
    #             }
    #
    #             # --- D. 准备向量 (Vectors) ---
    #             vectors = {}
    #
    #             # 1. 文本语义向量 (MiniLM - 必选)
    #             # DataFrame 里的 embedding 可能是 numpy array 或 string，需转 list
    #             vec_data = row.get('embedding')
    #             if isinstance(vec_data, str):
    #                 vec_data = json.loads(vec_data)
    #             elif hasattr(vec_data, 'tolist'):
    #                 vec_data = vec_data.tolist()
    #
    #             if vec_data:
    #                 vectors["text_vector"] = vec_data
    #
    #             # 2. 视觉对齐向量 (PE - 可选，但强烈推荐)
    #             # 这让你的文本能被"图片"搜到
    #             if compute_pe and text_content:
    #                 # 注意：这里需要调用你的 PE 模型对文本进行编码
    #                 # 假设 pe_model 有 extract_text_features 方法
    #                 pe_vec = pe_model.extract_text_features(text_content)
    #                 # 确保转为 list
    #                 if hasattr(pe_vec, 'tolist'):
    #                     pe_vec = pe_vec.tolist()
    #                 # 如果返回的是二维数组 [[...]]，取第一个
    #                 if isinstance(pe_vec, list) and len(pe_vec) == 1 and isinstance(pe_vec[0], list):
    #                     pe_vec = pe_vec[0]
    #
    #                 vectors["pe_vector"] = pe_vec
    #
    #             # --- E. 添加到缓冲区 ---
    #             points_buffer.append(PointStruct(
    #                 id=str(uuid.uuid4()),
    #                 vector=vectors,
    #                 payload=payload
    #             ))
    #
    #             if len(points_buffer) >= batch_size:
    #                 self.client.upsert(
    #                     collection_name=self.DOC_COLLECTION,
    #                     points=points_buffer
    #                 )
    #                 points_buffer = []
    #
    #         except Exception as e:
    #             print(f"⚠️ 跳过第 {idx} 行: {e}")
    #             continue
    #
    #     # 上传剩余数据
    #     if points_buffer:
    #         self.client.upsert(
    #             collection_name=self.DOC_COLLECTION,
    #             points=points_buffer
    #         )
    #
    #     print("✅ 文本数据入库完成！")
