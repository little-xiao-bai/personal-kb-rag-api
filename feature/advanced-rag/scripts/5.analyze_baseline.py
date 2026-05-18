"""
分析不同 RAG baseline 实验结果：
1. BGE baseline
2. BGE + sentence window
3. BGE + sentence window + reranker

输出整体命中率，以及按题型 / 难度分组统计。
"""


import pandas as pd
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
print (ROOT_PATH)

# CSV_FILE = ROOT_PATH / "baseline_supported_200.csv"
CSV_FILE_200_bge = ROOT_PATH / "baseline_supported_200_bge_top20.csv"


df_200_bge = pd.read_csv(CSV_FILE_200_bge)

total = len(df_200_bge)

answer_hits = df_200_bge["answer_hit"].sum()
retrieval_hits = df_200_bge["retrieval_hit"].sum()

print("=" * 80)
print("分析使用 BGE embedding 的结果：")
print(f"总问题数: {total}")
print(f"Answer Hit: {answer_hits}/{total} = {answer_hits / total:.2%}")
print(f"Retrieval Hit: {retrieval_hits}/{total} = {retrieval_hits / total:.2%}")

print("失败题数:", len(df_200_bge[df_200_bge["answer_hit"] == False]))


print(df_200_bge.groupby("type")[["answer_hit", "retrieval_hit"]].mean())
print(df_200_bge.groupby("level")[["answer_hit", "retrieval_hit"]].mean())
print(df_200_bge["type"].value_counts())

print("=" * 80)
print("分析使用 BGE embedding + window 的结果：")
CSV_FILE_200_bge_window = ROOT_PATH / "baseline_supported_200_bge_window.csv"

# failed = df[df["answer_hit"] == False]
df_200_bge_window = pd.read_csv(CSV_FILE_200_bge_window)

total = len(df_200_bge_window)

answer_hits = df_200_bge_window["answer_hit"].sum()
retrieval_hits = df_200_bge_window["retrieval_hit"].sum()

print(f"总问题数: {total}")
print(f"Answer Hit: {answer_hits}/{total} = {answer_hits / total:.2%}")
print(f"Retrieval Hit: {retrieval_hits}/{total} = {retrieval_hits / total:.2%}")

print("失败题数:", len(df_200_bge_window[df_200_bge_window["answer_hit"] == False]))



print(df_200_bge_window.groupby("type")[["answer_hit", "retrieval_hit"]].mean())
print(df_200_bge_window.groupby("level")[["answer_hit", "retrieval_hit"]].mean())
print(df_200_bge_window["type"].value_counts())


print("=" * 80)
print("分析使用 BGE embedding + window +reranker 的结果：")
CSV_FILE_200_bge_window_rerank = ROOT_PATH / "baseline_supported_200_bge_window_rerank.csv"

# failed = df[df["answer_hit"] == False]
df_200_bge_window_rerank = pd.read_csv(CSV_FILE_200_bge_window_rerank)

total = len(df_200_bge_window_rerank)

answer_hits = df_200_bge_window_rerank["answer_hit"].sum()
retrieval_hits = df_200_bge_window_rerank["retrieval_hit"].sum()

print(f"总问题数: {total}")
print(f"Answer Hit: {answer_hits}/{total} = {answer_hits / total:.2%}")
print(f"Retrieval Hit: {retrieval_hits}/{total} = {retrieval_hits / total:.2%}")

print("失败题数:", len(df_200_bge_window_rerank[df_200_bge_window_rerank["answer_hit"] == False]))



print(df_200_bge_window_rerank.groupby("type")[["answer_hit", "retrieval_hit"]].mean())
print(df_200_bge_window_rerank.groupby("level")[["answer_hit", "retrieval_hit"]].mean())
print(df_200_bge_window_rerank["type"].value_counts())


# for idx, row in failed.iterrows():
#     print("=" * 100)
#     print(f"问题 ID: {row['id']}")

#     print("问题:")
#     print(row["query"])

#     print("\n标准答案:")
#     print(row["expected_answer"])

#     print("\nRAG答案:")
#     print(row["rag_answer"])

#     print("\nGold Context:")
#     print(row["gold_context"])

    # print("\nRetrieved Context:")
    # print(row["retrieved_context"])
