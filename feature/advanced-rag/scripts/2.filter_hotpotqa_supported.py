"""
作用：
筛选 HotpotQA 中 supporting_facts 完整存在的样本，
生成可用于严格评估的测试集和 sentence-level 知识库。

输出：
1. test_questions_supported_200.json
2. hotpotqa_supported_sentences_200/

用途：
用于 sentence-level RAG baseline evaluation。
"""
import json
import re
from pathlib import Path


# ==========================================================
# 1. 路径配置
# ==========================================================
SCRIPT_DIR = Path(__file__).parent  
ROOT_DIR = SCRIPT_DIR.parent        


INPUT_FILE = ROOT_DIR / "data" / "hotpotqa_dev_fullwiki_pretty.json"

OUTPUT_QUESTIONS_FILE = ROOT_DIR / "data" / "test_questions_supported_200.json"
OUTPUT_SENTENCE_DIR = ROOT_DIR / "data" / "hotpotqa_supported_sentences_200"

OUTPUT_SENTENCE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. 工具函数
# ==========================================================
def safe_filename(text: str, max_len: int = 80) -> str:
    """
    将 title 转成安全文件名
    """
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)
    return text[:max_len]


def extract_gold_context(item: dict) -> list[dict] | None:
    """
    检查 supporting_facts 是否都能在当前 item["context"] 中找到。

    如果全部找到：
        返回 gold_context

    如果有任何一个找不到：
        返回 None
    """
    context_map = {
        title: sentences
        for title, sentences in item.get("context", [])
    }

    gold_context = []

    for title, sent_idx in item.get("supporting_facts", []):
        sentences = context_map.get(title)

        if sentences is None:
            return None

        if sent_idx < 0 or sent_idx >= len(sentences):
            return None

        gold_context.append({
            "title": title,
            "sentence_index": sent_idx,
            "text": sentences[sent_idx].strip()
        })

    return gold_context


# ==========================================================
# 3. 读取 HotpotQA
# ==========================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    hotpot_data = json.load(f)

print(f"原始样本数量: {len(hotpot_data)}")


# ==========================================================
# 4. 筛选 supporting_facts 完整存在的样本
# ==========================================================
filtered_questions = []
selected_items = []

for item in hotpot_data:
    gold_context = extract_gold_context(item)

    if gold_context is None:
        continue

    selected_items.append(item)

    filtered_questions.append({
        "id": item.get("_id"),
        "query": item.get("question"),
        "expected_answer": item.get("answer"),
        "supporting_facts": item.get("supporting_facts", []),
        "gold_context": gold_context,
        "context": item.get("context", []),   # 保留原始 context,
        "type": item.get("type"),
        "level": item.get("level")
    })

    if len(filtered_questions) >= 200:
        break

print(f"筛选后问题数量: {len(filtered_questions)}")


# ==========================================================
# 5. 保存筛选后的测试问题
# ==========================================================
with open(OUTPUT_QUESTIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        filtered_questions,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"已保存测试问题: {OUTPUT_QUESTIONS_FILE}")


# ==========================================================
# 6. 生成 sentence-level 知识库
# ==========================================================
file_count = 0

for item_idx, item in enumerate(selected_items, start=1):
    for title, sentences in item.get("context", []):
        title_safe = safe_filename(title)

        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if not sentence:
                continue

            file_name = f"supported_{item_idx}_{title_safe}_sent{sent_idx}.md"
            file_path = OUTPUT_SENTENCE_DIR / file_name

            content = f"{title}\n\n{sentence}"

            with open(file_path, "w", encoding="utf-8") as out_f:
                out_f.write(content)

            file_count += 1

print(f"已生成 sentence-level 文件数量: {file_count}")
print(f"输出目录: {OUTPUT_SENTENCE_DIR}")


# ==========================================================
# 7. 简单预览
# ==========================================================
print("\n前 3 条问题预览：")
for q in filtered_questions[:3]:
    print("-" * 50)
    print("Q:", q["query"])
    print("A:", q["expected_answer"])
    print("Gold:")
    for g in q["gold_context"]:
        print(f"  [{g['title']} #{g['sentence_index']}] {g['text']}")