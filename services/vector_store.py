from pathlib import Path
from typing import Any

import chromadb

# =========================
# Chroma 向量库封装
# =========================
class ChromaVectorStore:
    """
    Chroma 向量库封装

    职责：
        1. 保存 chunk 文本
        2. 保存 embedding 向量
        3. 保存 metadata
        4. 根据 query embedding 检索 top-k chunks
    """

    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "documents"):
        """
        初始化 Chroma 向量数据库客户端

        参数:
            collection_name: 集合名称，默认为 "documents"
            persist_directory: 数据持久化目录，默认为 "chroma_db"
        """

        # 创建持久化客户端 # 数据会保存到本地目录
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 获取集合 # 如果不存在就自动创建
        self.collection = self.client.get_or_create_collection(name=collection_name)

    # =========================
    # 添加 chunks
    # =========================
    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        """
        将 chunk 和对应的 embedding 添加到向量库

        参数:
            chunks: List[Dict] 每个 dict 包含 chunk_id, text, source, metadata 等字段
            embeddings: List[List[float]] 每个元素是对应 chunk 的向量
        """
        # 提取 chunk 唯一 ID
        ids = [chunk["chunk_id"] for chunk in chunks]

        # 提取 chunk 文本
        documents = [chunk["text"] for chunk in chunks]

    

        # 提取 metadata
        metadatas = [
            {
                "chunk_id": chunk["chunk_id"],
                "source": str(chunk["source"]),
                "page": int(chunk.get("page") or 0),    # page 可能为 None，所以用 get 方法
                "start": int(chunk["start"]),
                "end": int(chunk["end"]),
            }
            for chunk in chunks
        ]

        # 写入 Chroma 向量库
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    # =========================
    # 相似检索
    # =========================
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        """
        根据 query embedding 检索最相似的 top-k chunks

        参数:
            query_embedding: List[float] 查询文本的向量表示
            top_k: int 返回最相关的 chunk 数量，默认为 5

        返回:
            List[Dict] 每个 dict 包含 chunk_id, text, source, metadata 等字段
        """

        # 向量检索
        results = self.collection.query(
            query_embeddings=[query_embedding],

            # 返回 top_k 个结果
            n_results=top_k,

            # 显式指定返回内容
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_chunks = []

        # results 数据结构类似：
        #
        # {
        #   "ids": [["chunk_1", "chunk_2"]],
        #   "text": [["文本1", "文本2"]],
        #   "metadatas": [[{...}, {...}]],
        #   "distances": [[0.12, 0.33]]
        # }
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                # chunk ID
                "chunk_id": results["ids"][0][i],
                
                # 文本内容
                "text": results["documents"][0][i],

                # metadata
                "metadata": results["metadatas"][0][i],

                # 相似距离# 越小越相似
                "distance": results["distances"][0][i],
            })

        return retrieved_chunks

# =========================
# 本地测试
# =========================
if __name__ == "__main__":
    import sys
    # from pprint import pprint

    # 将项目根目录加入 sys.path，保证可以导入 services 包
    ROOT_DIR = Path(__file__).resolve().parent.parent

    # 加入 Python 搜索路径
    sys.path.insert(0, str(ROOT_DIR))

    from services.document_loader import load_document
    from services.chunker import split_text
    from services.embedding import EmbeddingClient

    # 测试文档路径
    # doc_path = ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para1.md"


    # # 加载文档
    # doc = load_document(doc_path)

    # # # 切 chunk
    # # chunks = split_text(
    # #     text=doc["text"],
    # #     source=doc["source"],
    # #     chunk_size=500,
    # # )

    # # ==========================
    # # 如果是 PDF：按页切分
    # # ==========================
    # if doc["metadata"]["file_type"] == "pdf":
    #     chunks = []

    #     for page_item in doc["pages"]:
    #         page_chunks = split_text(
    #             text=page_item["text"],
    #             source=doc["source"],
    #             chunk_size=500,
    #             page=page_item["page"]
    #         )
    #         chunks.extend(page_chunks)

    # # ==========================
    # # 如果是 TXT / MD：直接按全文切分
    # # ==========================
    # else:
    #     # print(f"doc['text'] 类型: {type(doc['text'])}")
    #     chunks = split_text(
    #         text=doc["text"],
    #         source=doc["source"],
    #         chunk_size=1000,
    #         page=None
    #     )


    # 初始化 embedding
    embedder = EmbeddingClient(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    # #提取文本
    # texts = [chunk["text"] for chunk in chunks]

    # #文档 embedding
    # embeddings = embedder.embed_texts(texts)

    # 初始化向量库
    # vector_store = ChromaVectorStore(
    #     persist_dir=str(ROOT_DIR / "chroma_db"),
    #     collection_name="documents",
    # )


    vector_store = ChromaVectorStore(
        persist_dir=r"D:\SWL_chroma_db\hotpot_supported_sentences",
        collection_name="hotpot_supported_sentences"
    )

    # # 写入向量库
    # vector_store.add_chunks(chunks, embeddings)

    # 查询文本
    query = "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"

    # query embedding
    query_embedding = embedder.embed_query(query)

    # 检索
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=20,
    )

    # 输出结果
    print("\n查询:", query)
    print("=" * 50)

    for i, result in enumerate(results, start=1):
        print(f"\n结果 {i}")
        print("chunk_id:", result["chunk_id"])
        print("distance:", result["distance"])
        print("source:", result["metadata"]["source"])
        print("内容前 100 字符:", result["text"])