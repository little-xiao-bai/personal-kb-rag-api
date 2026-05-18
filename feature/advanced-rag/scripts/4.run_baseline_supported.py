"""
用途：
  跑 HotpotQA sentence-level supported 50 条问题的 Naive RAG baseline。

输入：
  data/test_questions_supported_50.json

前提：
  1. FastAPI 服务已经启动
  2. main.py 中的 ChromaVectorStore 已经指向：
       persist_dir = D:\SWL_chroma_db_supported_sentences
       collection_name = hotpot_supported_sentences
  3. /rag/query 接口可以正常调用

输出：
  feature/advanced-rag/baseline_supported_50.csv

CSV 中会保存：
  - query
  - expected_answer
  - rag_answer
  - gold_context
  - retrieved_context
  - retrieved_sources
  - answer_hit
  - retrieval_hit
  - error"""

import json
import requests
from pathlib import Path
from time import sleep

import pandas as pd

import re
import string
from collections import Counter


# ==========================================================
# 1. 路径配置
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent     
SCRIPT_DIR = Path(__file__).resolve().parent                 

print(ROOT_DIR)
print(SCRIPT_DIR)

TEST_QUESTIONS_FILE = ROOT_DIR / "data" / "test_questions_supported_200.json"
# OUTPUT_CSV_FILE = ROOT_DIR / "baseline_supported_200_bge_top20.csv"
# OUTPUT_CSV_FILE = ROOT_DIR / "baseline_supported_200_bge_window.csv"
OUTPUT_CSV_FILE = ROOT_DIR / "baseline_supported_200_bge_window_rerank.csv"


# ==========================================================
# 2. RAG API 配置
# ==========================================================
RAG_QUERY_URL = "http://127.0.0.1:8000/rag/query"

TOP_K = 25       # RAG 返回的 top_k 相关文档数量。可以根据需要调整，例如 20、25、30 等。
REQUEST_INTERVAL = 0.1  # 每个请求之间的间隔，单位秒。可以根据需要调整，例如 0.1、0.2、0.5 等，以避免过快请求导致服务器压力过大。




# ==========================================================
# 3. 工具函数
# ==========================================================
# def normalize_text(text: str) -> str:
#     """
#     简单文本归一化，用于粗略比较答案。
#     """
#     if text is None:
#         return ""

#     return text.strip().lower()


# def check_answer_hit(expected_answer: str, rag_answer: str) -> bool:
#     """
#     判断 RAG answer 是否命中 expected_answer。

#     当前是宽松判断：
#         expected_answer 出现在 rag_answer 中，就算命中。

#     例如：
#         expected_answer = "no"
#         rag_answer = "No, they are not located in the same neighborhood."
#         => True
#     """
#     expected = normalize_text(expected_answer)
#     answer = normalize_text(rag_answer)

#     if not expected or not answer:
#         return False

#     return expected in answer

def normalize_text(text: str) -> str:
    """
    HotpotQA / SQuAD 风格归一化：
    小写、去标点、去冠词、合并空格。
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def check_answer_hit(expected_answer: str, rag_answer: str) -> bool:
    """
    只升级 answer_hit，不改 retrieval_hit。

    判断顺序：
    1. yes/no 特殊处理
    2. 标准化后完全匹配
    3. 双向 substring
    4. token-level F1 >= 0.5
    """
    expected = normalize_text(expected_answer)
    answer = normalize_text(rag_answer)

    if not expected or not answer:
        return False

    # yes / no 特殊处理
    if expected == "yes":
        return (
            answer.startswith("yes")
            or "yes" in answer.split()
            or "both" in answer.split()
            or "same" in answer.split()
        )

    if expected == "no":
        return (
            answer.startswith("no")
            or "no" in answer.split()
            or "not" in answer.split()
            or "different" in answer.split()
        )

    # 完全匹配
    if expected == answer:
        return True

    # 双向包含：解决全名 vs 简写
    if expected in answer or answer in expected:
        return True

    # F1 兜底
    return f1_score(rag_answer, expected_answer) >= 0.5



def check_retrieval_hit(gold_context: list, retrieved_context: str) -> bool:
    """
    判断 retrieved_context 是否覆盖 gold_context。

    当前逻辑：
        只要所有 gold sentence 都能在 retrieved_context 中找到，
        就算 retrieval_hit = True。

    注意：
        这是严格判断。
        如果只找到部分 gold evidence，结果为 False。
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
    """
    将 gold_context list 转成方便写入 CSV 的字符串。
    """
    parts = []

    for gold in gold_context:
        title = gold.get("title", "")
        sentence_index = gold.get("sentence_index", "")
        text = gold.get("text", "")

        parts.append(f"[{title} #{sentence_index}] {text}")

    return "\n\n".join(parts)


# ==========================================================
# 4. 读取测试问题
# ==========================================================
with open(TEST_QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"读取测试问题数量: {len(questions)}")


# ==========================================================
# 5. 循环调用 /rag/query
# ==========================================================
rows = []

for idx, q in enumerate(questions, start=1):
    query = q["query"]
    expected_answer = q.get("expected_answer", "")
    gold_context = q.get("gold_context", [])

    print("=" * 80)
    print(f"正在处理第 {idx}/{len(questions)} 题")
    print("Q:", query)
    print("Expected:", expected_answer)

    payload = {
        "question": query,
        "top_k": TOP_K
    }

    try:
        response = requests.post(
            RAG_QUERY_URL,
            json=payload,
            timeout=90
        )
        response.raise_for_status()

        data = response.json()

        rag_answer = data.get("answer", "")
        sources = data.get("sources", [])

        retrieved_context = "\n\n".join(
            source.get("text_preview", "")
            for source in sources
        )

        retrieved_sources = "\n".join(
            source.get("source", "")
            for source in sources
        )

        retrieved_chunk_ids = "\n".join(
            source.get("chunk_id", "")
            for source in sources
        )

        answer_hit = check_answer_hit(
            expected_answer=expected_answer,
            rag_answer=rag_answer
        )

        retrieval_hit = check_retrieval_hit(
            gold_context=gold_context,
            retrieved_context=retrieved_context
        )

        error = data.get("error")

        print("RAG Answer:", rag_answer)
        print("Answer Hit:", answer_hit)
        print("Retrieval Hit:", retrieval_hit)

    except Exception as e:
        rag_answer = ""
        retrieved_context = ""
        retrieved_sources = ""
        retrieved_chunk_ids = ""
        answer_hit = False
        retrieval_hit = False
        error = str(e)

        print("出错:", error)

    rows.append({
        "id": q.get("id", ""),
        "query": query,
        "expected_answer": expected_answer,
        "rag_answer": rag_answer,
        "gold_context": format_gold_context(gold_context),
        "retrieved_context": retrieved_context,
        "retrieved_sources": retrieved_sources,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "answer_hit": answer_hit,
        "retrieval_hit": retrieval_hit,
        "type": q.get("type", ""),
        "level": q.get("level", ""),
        "error": error
    })

    sleep(REQUEST_INTERVAL)


# ==========================================================
# 6. 保存 CSV
# ==========================================================
df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 80)
print(f"Baseline CSV 已保存到: {OUTPUT_CSV_FILE}")


# ==========================================================
# 7. 简单统计
# ==========================================================
total = len(df)

answer_hits = df["answer_hit"].sum()
retrieval_hits = df["retrieval_hit"].sum()

print(f"总问题数: {total}")
print(f"Answer Hit: {answer_hits}/{total} = {answer_hits / total:.2%}")
print(f"Retrieval Hit: {retrieval_hits}/{total} = {retrieval_hits / total:.2%}")