# services/reranker.py

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranker / 重排序器

    作用：
        对 Retriever 返回的 top_k chunks 进行二次排序。

    工作流程：
        1. Retriever 先从 Chroma 中召回 top20
        2. Reranker 对 question + chunk_text 做交叉编码
        3. 输出相关性分数
        4. 按分数重新排序
        5. 取 top5 给 LLM
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"正在加载 Reranker 模型: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Reranker 模型加载完成")

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        对检索结果重新排序。

        参数：
            query:
                用户问题

            chunks:
                vector_store.search() 返回的 chunk 列表

            top_k:
                rerank 后保留多少条

        返回：
            reranked_chunks:
                按 reranker_score 从高到低排序后的 chunks
        """

        if not chunks:
            return []

        pairs = [
            (query, chunk.get("text", ""))
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for chunk, score in zip(chunks, scores):
            new_chunk = chunk.copy()
            new_chunk["reranker_score"] = float(score)
            reranked.append(new_chunk)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return reranked[:top_k]