from sentence_transformers import SentenceTransformer

class EmbeddingClient:
    """
    Embedding 客户端
    作用：
        负责把文本转换成向量（embedding）

    支持模型：
        - BAAI/bge-small-zh-v1.5   （中文推荐）
        - sentence-transformers/all-MiniLM-L6-v2 （英文常用）
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"): # 另一个模型："sentence-transformers/all-MiniLM-L6-v2" 
        """
        初始化 Embedding 模型

        参数：
            model_name: 使用的 embedding 模型名称
        """

        print(f"正在加载 Embedding 模型: {model_name}")

        # 加载 sentence-transformers 模型
        # 第一次运行会自动下载模型
        self.model = SentenceTransformer(model_name)
        print("模型加载完成")
    

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文本向量

        参数：
            texts: 文本列表

        返回：
            list[list[float]]
            例如：
            [
                [0.12, -0.33, ...],
                [0.45,  0.91, ...]
            ]
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True, # 标准化向量，让向量长度统一，方便后续做相似度计算
        )

        # numpy 数组 -> Python list
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        """
        生成单条查询文本的向量

        参数：
            query: 用户查询文本

        返回：
            list[float]
            例如：
            [0.11, -0.29, 0.88, ...]
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding.tolist()
    

# ==========================================
# 本地测试入口
# ==========================================
if __name__ == "__main__":

    # 创建 Embedding 客户端
    embedder = EmbeddingClient()

    print("\n" + "=" * 50)
    print("测试 1：单条 query embedding")
    print("=" * 50)

    query = "什么是 FastAPI？"

    query_vector = embedder.embed_query(query)

    print("查询文本:", query)
    print("向量维度:", len(query_vector))
    print("前 10 个值:", query_vector[:10])

    print("\n" + "=" * 50)
    print("测试 2：批量文本 embedding")
    print("=" * 50)

    texts = [
        "FastAPI 是一个高性能 Python Web 框架",
        "RAG 可以基于文档内容回答问题",
        "Chroma 是一个向量数据库"
    ]

    vectors = embedder.embed_texts(texts)

    print("文本数量:", len(vectors))
    print("每个向量维度:", len(vectors[0]))

    for i, text in enumerate(texts):
        print(f"\n文本 {i+1}: {text}")
        print("前 10 个向量值:", vectors[i][:10])

