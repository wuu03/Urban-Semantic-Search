import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType
from tqdm.notebook import tqdm
from qdrant_client.models import (
    VectorParams,
    Distance,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    HnswConfigDiff
)


class GeoVectorDB:
    def __init__(self, host="localhost", port=6333, collection_name="venice_historical_map"):
        # 自动判断是本地路径还是服务器
        if host.startswith(".") or "/" in host or "\\" in host:
            self.client = QdrantClient(path=host)
        else:
            self.client = QdrantClient(host=host, port=port)

        self.collection_name = collection_name

    def init_collection(self, vector_dim=1024, force_recreate=False):
        # 如果强制重建，先删除
        if force_recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE,

                    # 🔥 核心优化 1: 开启 Int8 量化 (体积缩小4倍，速度提升5倍)
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True  # 强制缓存在内存中
                        )
                    )
                ),

                # 🔥 核心优化 2: 调整 HNSW 索引参数 (牺牲一点写入速度，换取极致读取速度)
                hnsw_config=HnswConfigDiff(
                    m=16,  # 节点连接数 (默认16，越大搜得越准但越慢，保持16即可)
                    ef_construct=100,  # 构建时的搜索深度
                    on_disk=False  # ❌ 严禁设为 True，必须让索引在内存里
                )
            )

            # 建立地理索引
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="location",
                field_schema=PayloadSchemaType.GEO
            )
            print(f"⚡ 集合 '{self.collection_name}' 创建成功 (已开启极速模式)")
        else:
            print(f"👌 集合 '{self.collection_name}' 已存在")

    def ingest_data(self, image_features, registered_data_list, source_image_name, batch_size=100):
        """
        :param image_features: Embedding 向量列表 (N, dim)
        :param registered_data_list: 你的配准后数据列表 (包含 geo_geometry, pixel_coords 等)
        :param source_image_name: 图片ID (如 'venice_1838.jpg')
        """
        points_buffer = []
        total = len(image_features)

        print(f"🚀 开始入库: {source_image_name} ({total} 个切片)...")

        # 使用 zip 同时遍历 向量 和 配准数据
        for idx, (emb, item) in tqdm(enumerate(zip(image_features, registered_data_list)), total=total):

            # 1. 提取 Qdrant 索引必须的 Lat/Lon
            # 你的数据结构: item['geo_geometry']['wgs84']['center'] = [lat, lon]
            center_coords = item['geo_geometry']['wgs84']['center']
            lat, lon = center_coords[0], center_coords[1]

            # 2. 构建 Payload (元数据)
            payload = {
                "source_image": source_image_name,
                "orig_index": item.get('index', idx),
                "pixel_coords": item['pixel_coords'],  # [x, y]

                # Qdrant 专用地理字段 (用于搜索)
                "location": {
                    "lat": lat,
                    "lon": lon
                },

                # 完整地理信息 (用于前端画框)
                "geo_detail": item['geo_geometry']
            }

            # 3. 构建 Point
            points_buffer.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload
            ))

            # 4. 批量上传
            if len(points_buffer) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points_buffer
                )
                points_buffer = []

        # 处理剩余
        if points_buffer:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points_buffer
            )

        print(f"✅ 入库完成！已存储 {total} 条数据。")
