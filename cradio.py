import os
import faiss
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor
from tqdm.notebook import tqdm


class FaissDensePipeline:
    def __init__(self, model_version="c-radio_v4-h", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dev = "cuda" if "cuda" in self.device else "cpu"

        print(f"[1/3] 初始化 RADIO 模型 (version={model_version}, device={self.device})...")
        self.model = torch.hub.load(
            'NVlabs/RADIO', 'radio_model',
            version=model_version, progress=True,
            skip_validation=True, adaptor_names=['siglip2-g'],
            force_reload=False
        ).to(self.device).eval()

        self.sig2_adaptor = self.model.adaptors['siglip2-g']
        self.patch_size = self.model.patch_size

        # 动态获取文本编码的实际维度 (SigLIP2 映射后的维度)
        with torch.no_grad():
            dummy = self.sig2_adaptor.tokenizer(["test"]).to(self.device)
            dummy_vec = self.sig2_adaptor.encode_text(dummy, normalize=True)
            self.text_dim = dummy_vec.shape[-1]

        # FAISS 初始化
        self.dim = self.text_dim
        self.index = faiss.IndexFlatIP(self.dim)  # IndexFlatIP 配合 L2归一化 = 余弦相似度
        self.metadata = []
        self.image_db = {}

        print(f"✅ 模型初始化完毕！文本与视觉的共享 Latent Space 维度为: {self.dim}D\n")

    @torch.no_grad()
    def extract_and_index(self, image_id, pil_image, max_res=1024):
        """提取 siglip2-g 的稠密像素级特征并极速存入 FAISS"""

        # 1. 安全缩放 (防 OOM)
        img = pil_image.copy()
        img.thumbnail((max_res, max_res), Image.Resampling.LANCZOS)

        # 2. 预处理为模型支持的分辨率
        x = pil_to_tensor(img).to(dtype=torch.float32, device=self.device).div_(255.0).unsqueeze_(0)
        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

        h_feat, w_feat = nearest_res[0] // self.patch_size, nearest_res[1] // self.patch_size
        print(
            f"🔧 正在处理: {image_id} | 图像分辨率: {nearest_res} | 稠密网格: {h_feat}×{w_feat} = {h_feat * w_feat} 个点")

        # # 3. 前向推理 (使用 SigLIP 适配器提取特征)
        # with torch.autocast(self.autocast_dev, dtype=torch.bfloat16):
        #     vis_output = self.model(x)
        #     # 获取 adaptor 输出，官方返回对象包含 features 属性
        #     siglip_out = vis_output['siglip2-g']
        #     spatial_features = siglip_out.features if hasattr(siglip_out, 'features') else siglip_out[1]
        #
        # # 4. 严格校验维度
        # actual_dim = spatial_features.shape[-1]
        # if actual_dim != self.dim:
        #     raise ValueError(f"🚨 严重错误：视觉特征维度({actual_dim})与文本维度({self.dim})不匹配！")
        #
        # # 5. 归一化并恢复网格结构
        # # spatial_features 形状: [1, h_feat * w_feat, dim]
        # grid_feats = spatial_features.squeeze(0).reshape(h_feat, w_feat, self.dim)
        # grid_feats = F.normalize(grid_feats, p=2, dim=-1).to(torch.float32).cpu().numpy()

        # 3. 前向推理 (🌟 去掉 NCHW，绕过 NVIDIA 官方适配器的 Bug)
        with torch.autocast(self.autocast_dev, dtype=torch.bfloat16):
            vis_output = self.model(x)  # 使用默认输出格式
            # 兼容不同版本的输出解包
            siglip_out = vis_output['siglip2-g']
            spatial_features = siglip_out.features if hasattr(siglip_out, 'features') else siglip_out[1]

        # 4. 严格校验维度
        actual_dim = spatial_features.shape[-1]
        if actual_dim != self.dim:
            raise ValueError(f"🚨 严重错误：视觉特征维度({actual_dim})与文本({self.dim})不匹配！")

        # 5. 归一化并恢复网格结构 (ViT Patch 是按行排列的，直接 Reshape 是安全的)
        # spatial_features 形状: [1, h_feat * w_feat, dim]
        grid_feats = spatial_features.squeeze(0).reshape(h_feat, w_feat, self.dim)
        grid_feats = F.normalize(grid_feats, p=2, dim=-1).to(torch.float32).cpu().numpy()

        # 6. 极速写入 FAISS (零循环，直接传入矩阵)
        flat_feats = np.ascontiguousarray(grid_feats.reshape(-1, self.dim))
        self.index.add(flat_feats)

        # 7. 构建 Metadata (记录每个点属于哪张图、第几行、第几列)
        for i in range(h_feat):
            for j in range(w_feat):
                self.metadata.append({'image_id': image_id, 'h_idx': i, 'w_idx': j})

        # 8. 缓存数据供渲染热力图使用
        self.image_db[image_id] = {
            'original_img': img,
            'grid_features': grid_feats
        }

        print(f"✅ {image_id} 成功入库！新增 {flat_feats.shape[0]} 个局部特征，FAISS 总库容: {self.index.ntotal}\n")

    @torch.no_grad()
    def search_and_visualize(self, query_text, top_k_pixels=512, top_percent=10):
        """
        通过 FAISS 检索最相关的像素点，并在黑白底图上绘制“黄橙红”热力图。
        top_percent: 只显示得分排名前百分之几的区域（默认 10%，即最相关的核心区域）
        """
        if self.index.ntotal == 0:
            print("❌ FAISS 索引为空，请先调用 extract_and_index()")
            return

        print(f"🔍 正在编码搜索词: '{query_text}'...")
        text_input = self.sig2_adaptor.tokenizer([query_text]).to(self.device)
        text_vec = self.sig2_adaptor.encode_text(text_input, normalize=True).cpu().numpy().astype('float32')

        k = min(top_k_pixels, self.index.ntotal)
        D, I = self.index.search(text_vec, k=k)

        # 统计命中图片 (去重)
        hit_images = set()
        for idx in I[0]:
            hit_images.add(self.metadata[idx]['image_id'])

        # 对命中的图片逐一渲染热力图
        for img_id in hit_images:
            data = self.image_db[img_id]
            img = data['original_img']
            grid = data['grid_features']

            # 1. 计算原始相似度矩阵
            similarity_map = np.dot(grid, text_vec.T).squeeze(-1)

            # 🌟 --- 新增：打印当前图片的相似度统计作为调参参考 --- 🌟
            sim_max = similarity_map.max()
            sim_min = similarity_map.min()
            sim_mean = similarity_map.mean()
            print(
                f"   ➤ [热力图参考] 图像 '{img_id}' 的原始相似度 | 最高: {sim_max:.4f}, 最低: {sim_min:.4f}, 平均: {sim_mean:.4f}")

            # 2. 插值放大回原图尺寸
            sim_tensor = torch.tensor(similarity_map).unsqueeze(0).unsqueeze(0)
            sim_resized = F.interpolate(
                sim_tensor, size=(img.height, img.width),
                mode='bicubic', align_corners=False
            ).squeeze().numpy()

            # 3. 归一化到 0 ~ 1 范围 (为了让颜色条标准统一)
            s_min, s_max = sim_resized.min(), sim_resized.max()
            norm_sim = (sim_resized - s_min) / (s_max - s_min + 1e-8)

            # 🌟 4. 核心修复：透明遮罩处理 (Mask) 🌟
            # 算出排名前 top_percent 的阈值 (比如 10% 就是 90 分位数)
            threshold = np.percentile(norm_sim, 100 - top_percent)

            # 把低于阈值的部分直接“隐身”，这样就不会被涂成大黄块了！
            masked_sim = np.ma.masked_where(norm_sim < threshold, norm_sim)

            # --- 生成黑白底图 ---
            bw_img = img.convert("L").convert("RGB")

            # --- 绘图 ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            axes[0].imshow(bw_img)
            axes[0].set_title(f"B&W Original: {img_id}", fontsize=14)
            axes[0].axis('off')

            axes[1].imshow(bw_img)

            # 直接传入带有透明遮罩的 masked_sim
            hm = axes[1].imshow(masked_sim, cmap='YlOrRd', alpha=0.7)

            axes[1].set_title(f"Heatmap (Top {top_percent}%)\nQuery: '{query_text}'", fontsize=14)
            axes[1].axis('off')

            plt.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04).set_label('Relative Confidence', rotation=270,
                                                                             labelpad=15)

            plt.tight_layout()

            # 🌟 核心修改：以 300 DPI 印刷级画质保存为 JPG 🌟
            # 自动用搜索词和图片ID命名，防止覆盖
            safe_query = query_text.replace(' ', '_').replace('/', '_')
            save_filename = f"Heatmap_{img_id}_{safe_query}.jpg"

            # bbox_inches='tight' 可以自动裁剪掉多余的白边
            plt.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(f"📸 高清原图已生成并保存至同级目录: {save_filename}")

            # 依然在网页里显示预览

            plt.show()

    def stats(self):
        print(f"\n📊 --- 数据库状态统计 ---")
        print(f"   已索引地图切片 : {len(self.image_db)} 张")
        print(f"   FAISS 局部特征 : {self.index.ntotal} 个点")
        print(f"   共享特征维度   : {self.dim} D\n")

    @torch.no_grad()
    def search_and_crop(self, query_text, top_k=5, crop_size=128):
        """
        极速检索：直接截取命中得分最高的 K 个局部区域并单独展示。
        crop_size: 截取框的像素大小 (建议 128 或 256)
        """
        if self.index is None or self.index.ntotal == 0:
            print("❌ FAISS 索引为空，请先入库！")
            return

        print(f"🔍 正在检索文本: '{query_text}'...")
        text_input = self.sig2_adaptor.tokenizer([query_text]).to(self.device)
        text_vec = self.sig2_adaptor.encode_text(text_input, normalize=True).cpu().numpy().astype('float32')

        # 1. FAISS 直接取全局 Top K 个像素点
        k = min(top_k, self.index.ntotal)
        D, I = self.index.search(text_vec, k=k)

        # 2. 准备画板
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
        if k == 1: axes = [axes]  # 防止单图报错
        fig.suptitle(f"Top {k} Cropped Hits for: '{query_text}'", fontsize=16, y=1.05)

        # 3. 逐个反算坐标并截取
        for rank, (score, flat_idx) in enumerate(zip(D[0], I[0])):
            # 从 Metadata 获取这个点的信息
            meta = self.metadata[flat_idx]
            img_id = meta['image_id']
            h_idx = meta['h_idx']
            w_idx = meta['w_idx']

            # 获取对应的原图和网格信息
            # data = self.image_db[img_id]
            # img = data['original_img']
            # h_feat = data['h_feat']
            # w_feat = data['w_feat']

            # 获取对应的原图和网格信息
            data = self.image_db[img_id]
            img = data['original_img']

            # 直接从已存储的网格特征中读取高和宽
            h_feat, w_feat, _ = data['grid_features'].shape

            # --- 核心数学逻辑：将特征网格坐标反算回原图像素坐标 ---
            # 加 0.5 是为了取这个 patch 的中心点
            center_x = (w_idx + 0.5) / w_feat * img.width
            center_y = (h_idx + 0.5) / h_feat * img.height

            # 计算截取框的上下左右边界 (防止越界)
            half_c = crop_size / 2
            left = max(0, center_x - half_c)
            top = max(0, center_y - half_c)
            right = min(img.width, center_x + half_c)
            bottom = min(img.height, center_y + half_c)

            # 抠图
            crop_img = img.crop((left, top, right, bottom))

            # 展示
            axes[rank].imshow(crop_img)
            axes[rank].set_title(f"Rank {rank + 1}\nScore: {score:.3f}\nFrom: {img_id}", fontsize=11)
            axes[rank].axis('off')

            # 画个红色的准星(可选)，标出真正的中心命中点
            axes[rank].plot(crop_img.width / 2, crop_img.height / 2, 'r+', markersize=15, markeredgewidth=2)

        plt.tight_layout()
        plt.show()


# ==========================================
# 测试运行区
# ==========================================
if __name__ == "__main__":
    # 1. 初始化引擎
    pipeline = FaissDensePipeline()

    # 2. 准备数据 (自动检测本地文件，找不到则生成测试图)
    print("--- 准备地图切片 ---")
    map_path = "data/raw_maps/venice_map_1.jpg"

    if os.path.exists(map_path):
        img1 = Image.open(map_path).convert("RGB")
        print(f"✅ 成功加载本地地图: {map_path}")
    else:
        print("⚠️ 未找到本地地图，生成包含河流和屋顶的模拟地图...")
        img1 = Image.new('RGB', (1200, 800), color=(220, 220, 220))
        d = ImageDraw.Draw(img1)
        d.rectangle([200, 200, 400, 800], fill="blue")  # 模拟河流
        d.rectangle([700, 150, 900, 350], fill="red")  # 模拟红屋顶
        d.rectangle([800, 600, 1000, 700], fill="green")  # 模拟植被

    # 3. 提取特征并入库
    # max_res=1024 可以在 A100/V100 上毫无压力地跑，如果你显存极大，可以调到 2048。
    pipeline.extract_and_index("Venice_Map_Tile_01", img1, max_res=1024)

    # 查看状态
    pipeline.stats()

    # 4. 执行 FAISS 全局密集检索 & 热力图渲染
    # 测试不同的语义词，看看热力图的高亮区域是否精确转移
    test_queries = [
        # "water canal",
        # "red building roofs",
        # "church",
        # "garden",
        "arched bridge over water"
    ]

    for q in test_queries:
        pipeline.search_and_visualize(q, top_k_pixels=500)

    # 截取 256x256 的上下文区域单独展示
    # pipeline.search_and_crop("red tiled roofs of old buildings", top_k=5, crop_size=128)

    # 截取 128x128 的特写区域单独展示
    pipeline.search_and_crop("arched bridge over water", top_k=5, crop_size=128)
