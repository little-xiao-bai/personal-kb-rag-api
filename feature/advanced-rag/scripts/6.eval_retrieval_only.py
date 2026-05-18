"""
评估 retrieval-only 效果，对比 rerank 前后的检索表现。

流程：
1. 读取测试问题
2. 对 query 做 embedding
3. 在 Chroma 中检索候选 chunks
4. 判断是否命中 gold context
5. 使用 reranker 重新排序
6. 再次判断 rerank 后是否命中
7. 保存结果并输出统计
"""
import json
from pathlib import Path

import pandas as pd


# ==========================================================
# 1. 加入项目根目录
# ==========================================================
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

print("ROOT_DIR:", ROOT_DIR)

from services.embedding import EmbeddingClient
from services.vector_store import ChromaVectorStore
from services.reranker import Reranker


# ==========================================================
# 2. 路径配置
# ==========================================================
FILE_DIR = Path(__file__).resolve().parent.parent  #\feature\advanced-rag
# print(FILE_DIR)

TEST_QUESTIONS_FILE = FILE_DIR / "data/test_questions_supported_200.json"

OUTPUT_CSV_FILE = FILE_DIR / "data_retrieval/ retrieval_only_bge_window_top20.csv"




# ==========================================================
# 3. Retrieval / Rerank 配置
# ==========================================================
RETRIEVE_TOP_K = 25      # Chroma 初筛数量
RERANK_TOP_K = 25      # rerank 后保留数量

embedder = EmbeddingClient(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = ChromaVectorStore(
    persist_dir=r"D:\SWL_chroma_db\hotpot_supported_windows_200_bge",
    collection_name="hotpot_supported_windows_200_bge"
)

reranker = Reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==========================================================
# 4. 工具函数
# ==========================================================
def normalize_text(text: str) -> str:
    if text is None:
        return ""

    return str(text).strip().lower()


def check_retrieval_hit(gold_context: list, retrieved_context: str) -> bool:
    """
    判断 retrieved_context 是否覆盖所有 gold sentence。
    """
    retrieved = normalize_text(retrieved_context)

    if not gold_context or not retrieved:
        return False

    for gold in gold_context:
        gold_text = normalize_text(gold.get("text", ""))

        if not gold_text:
            continue

        if gold_text not in retrieved:
            return False

    return True


def format_gold_context(gold_context: list) -> str:
    parts = []

    for gold in gold_context:
        title = gold.get("title", "")
        sentence_index = gold.get("sentence_index", "")
        text = gold.get("text", "")

        parts.append(f"[{title} #{sentence_index}] {text}")

    return "\n\n".join(parts)


def join_context(results: list[dict]) -> str:
    return "\n\n".join(
        item.get("text", "")
        for item in results
    )


def join_sources(results: list[dict]) -> str:
    return "\n".join(
        item.get("source", "")
        for item in results
    )


def join_chunk_ids(results: list[dict]) -> str:
    return "\n".join(
        item.get("chunk_id", "")
        for item in results
    )


# ==========================================================
# 5. 读取测试问题
# ==========================================================
with open(TEST_QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

print("测试问题数量:", len(questions))
print("RETRIEVE_TOP_K:", RETRIEVE_TOP_K)
print("RERANK_TOP_K:", RERANK_TOP_K)


# ==========================================================
# 6. 逐题检索：同时评估 before rerank / after rerank
# ==========================================================
rows = []

for idx, q in enumerate(questions, start=1):
    query = q["query"]
    expected_answer = q.get("expected_answer", "")
    gold_context = q.get("gold_context", [])

    print("=" * 80)
    print(f"正在检索第 {idx}/{len(questions)} 题")
    print("Q:", query)

    # -----------------------------
    # 6.1 Query embedding
    # -----------------------------
    query_embedding = embedder.embed_query(query)

    # -----------------------------
    # 6.2 Chroma 粗召回
    # -----------------------------
    retrieved_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=RETRIEVE_TOP_K
    )

    retrieved_context_before = join_context(retrieved_results)

    retrieval_hit_before = check_retrieval_hit(
        gold_context=gold_context,
        retrieved_context=retrieved_context_before
    )

    # -----------------------------
    # 6.3 Reranker 精排
    # -----------------------------
    reranked_results = reranker.rerank(
        query=query,
        chunks=retrieved_results,
        top_k=RERANK_TOP_K
    )

    retrieved_context_after = join_context(reranked_results)

    retrieval_hit_after = check_retrieval_hit(
        gold_context=gold_context,
        retrieved_context=retrieved_context_after
    )

    print("Retrieval Hit Before Rerank:", retrieval_hit_before)
    print("Retrieval Hit After Rerank:", retrieval_hit_after)

    rows.append({
        "id": q.get("id", ""),
        "query": query,
        "expected_answer": expected_answer,
        "gold_context": format_gold_context(gold_context),

        "retrieve_top_k": RETRIEVE_TOP_K,
        "rerank_top_k": RERANK_TOP_K,

        "retrieval_hit_before_rerank": retrieval_hit_before,
        "retrieval_hit_after_rerank": retrieval_hit_after,

        "retrieved_context_before_rerank": retrieved_context_before,
        "retrieved_context_after_rerank": retrieved_context_after,

        "retrieved_sources_before_rerank": join_sources(retrieved_results),
        "retrieved_sources_after_rerank": join_sources(reranked_results),

        "retrieved_chunk_ids_before_rerank": join_chunk_ids(retrieved_results),
        "retrieved_chunk_ids_after_rerank": join_chunk_ids(reranked_results),

        "type": q.get("type", ""),
        "level": q.get("level", "")
    })


# ==========================================================
# 7. 保存 CSV
# ==========================================================
df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 8. 输出统计
# ==========================================================
total = len(df)

before_hits = df["retrieval_hit_before_rerank"].sum()
after_hits = df["retrieval_hit_after_rerank"].sum()

print("=" * 80)
print(f"CSV 已保存到: {OUTPUT_CSV_FILE}")
print(f"总问题数: {total}")
print(f"Before Rerank Retrieval Hit: {before_hits}/{total} = {before_hits / total:.2%}")
print(f"After Rerank Retrieval Hit: {after_hits}/{total} = {after_hits / total:.2%}")

print("\n按 type 分组：")
print(
    df.groupby("type")[
        ["retrieval_hit_before_rerank", "retrieval_hit_after_rerank"]
    ].mean()
)

print("\n按 level 分组：")
print(
    df.groupby("level")[
        ["retrieval_hit_before_rerank", "retrieval_hit_after_rerank"]
    ].mean()
)


# # ==========================================================
# # 3. Retrieval 配置
# # ==========================================================
# TOP_K = 20

# embedder = EmbeddingClient(
#     model_name="BAAI/bge-small-en-v1.5"
# )

# vector_store = ChromaVectorStore(
#     persist_dir=r"D:\SWL_chroma_db\hotpot_supported_windows_200_bge",
#     collection_name="hotpot_supported_windows_200_bge"
# )


# # ==========================================================
# # 4. 工具函数
# # ==========================================================
# def normalize_text(text: str) -> str:
#     if text is None:
#         return ""

#     return str(text).strip().lower()


# def check_retrieval_hit(gold_context: list, retrieved_context: str) -> bool:
#     """
#     判断 retrieved_context 是否覆盖所有 gold sentence。
#     """
#     retrieved = normalize_text(retrieved_context)

#     if not gold_context or not retrieved:
#         return False

#     for gold in gold_context:
#         gold_text = normalize_text(gold.get("text", ""))

#         if not gold_text:
#             continue

#         if gold_text not in retrieved:
#             return False

#     return True


# def format_gold_context(gold_context: list) -> str:
#     parts = []

#     for gold in gold_context:
#         title = gold.get("title", "")
#         sentence_index = gold.get("sentence_index", "")
#         text = gold.get("text", "")

#         parts.append(f"[{title} #{sentence_index}] {text}")

#     return "\n\n".join(parts)


# # ==========================================================
# # 5. 读取测试问题
# # ==========================================================
# with open(TEST_QUESTIONS_FILE, "r", encoding="utf-8") as f:
#     questions = json.load(f)

# print("测试问题数量:", len(questions))


# # ==========================================================
# # 6. 逐题检索，不调用 LLM
# # ==========================================================
# rows = []

# for idx, q in enumerate(questions, start=1):
#     query = q["query"]
#     expected_answer = q.get("expected_answer", "")
#     gold_context = q.get("gold_context", [])

#     print("=" * 80)
#     print(f"正在检索第 {idx}/{len(questions)} 题")
#     print("Q:", query)

#     query_embedding = embedder.embed_query(query)

#     results = vector_store.search(
#         query_embedding=query_embedding,
#         top_k=TOP_K
#     )

#     retrieved_context = "\n\n".join(
#         item.get("text", "")
#         for item in results
#     )

#     retrieved_sources = "\n".join(
#         item.get("source", "")
#         for item in results
#     )

#     retrieved_chunk_ids = "\n".join(
#         item.get("chunk_id", "")
#         for item in results
#     )

#     retrieval_hit = check_retrieval_hit(
#         gold_context=gold_context,
#         retrieved_context=retrieved_context
#     )

#     print("Retrieval Hit:", retrieval_hit)

#     rows.append({
#         "id": q.get("id", ""),
#         "query": query,
#         "expected_answer": expected_answer,
#         "gold_context": format_gold_context(gold_context),
#         "retrieved_context": retrieved_context,
#         "retrieved_sources": retrieved_sources,
#         "retrieved_chunk_ids": retrieved_chunk_ids,
#         "retrieval_hit": retrieval_hit,
#         "type": q.get("type", ""),
#         "level": q.get("level", "")
#     })


# # ==========================================================
# # 7. 保存 CSV
# # ==========================================================
# df = pd.DataFrame(rows)

# df.to_csv(
#     OUTPUT_CSV_FILE,
#     index=False,
#     encoding="utf-8-sig"
# )


# # ==========================================================
# # 8. 输出统计
# # ==========================================================
# total = len(df)
# retrieval_hits = df["retrieval_hit"].sum()

# print("=" * 80)
# print(f"Retrieval-only CSV 已保存到: {OUTPUT_CSV_FILE}")
# print(f"总问题数: {total}")
# print(f"Retrieval Hit: {retrieval_hits}/{total} = {retrieval_hits / total:.2%}")

# print("\n按 type 分组：")
# print(df.groupby("type")["retrieval_hit"].mean())

# print("\n按 level 分组：")
# print(df.groupby("level")["retrieval_hit"].mean())