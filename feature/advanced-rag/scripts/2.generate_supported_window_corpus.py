"""
generate_supported_window_corpus.py

作用：
    从 test_questions_supported_200.json 的原始 context 中，
    生成真正的 sentence window corpus。

输入：
    feature/advanced-rag/data/test_questions_supported_200.json

输出：
    feature/advanced-rag/data/hotpotqa_supported_windows_200/

window 规则：
    每个 chunk = title + 前一句 + 当前句 + 后一句
"""

import json
import re
from pathlib import Path

# ==========================================================
# 1. 路径配置
# ==========================================================

# 当前文件：
#   feature/advanced-rag/scripts/generate_supported_window_corpus.py
#
# parent.parent.parent.parent:
#   回到项目根目录 personal-kb-rag-api
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# 输入：
#   之前筛选好的 200 条测试问题
INPUT_FILE = ROOT_DIR / "feature/advanced-rag/data/test_questions_supported_200.json"

# 输出：
#   sentence window corpus
OUTPUT_DIR = ROOT_DIR / "feature/advanced-rag/data/hotpotqa_supported_windows_200"

# 自动创建目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. 文件名清洗函数
#
# 为什么需要？
#
# title 里可能有：
#   空格
#   /
#   :
#   中文
#   特殊符号
#
# Windows 文件名不允许这些字符。
# ==========================================================
def safe_filename(text: str, max_len: int = 80):
    """
    把 title 转成安全文件名
    """

    # 空格改下划线
    text = text.replace(" ", "_")

    # 去掉非法字符
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)

    # 避免文件名过长
    return text[:max_len]


# ==========================================================
# 3. 读取筛选后的 200 条问题
# ==========================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

print("测试问题数量:", len(questions))


# ==========================================================
# 4. sentence window 参数
#
# WINDOW_RADIUS = 1
#
# 表示：
#   前一句 + 当前句 + 后一句
#
# 如果改成 2：
#   前两句 + 当前句 + 后两句
# ==========================================================
WINDOW_RADIUS = 1

file_count = 0


# ==========================================================
# 5. 遍历所有问题
# ==========================================================
for q_idx, item in enumerate(questions, start=1):

    # context 格式：
    #
    # [
    #   [
    #       "Djamaâ el Kebir",
    #       [
    #           "sentence 0",
    #           "sentence 1",
    #           ...
    #       ]
    #   ]
    # ]
    #
    context = item.get("context", [])

    # 遍历每个 title
    for title, sentences in context:

        # 清洗 title 用于文件名
        title_safe = safe_filename(title)

        # 遍历该 title 下的每一句
        for sent_idx, sentence in enumerate(sentences):

            sentence = sentence.strip()

            if not sentence:
                continue

            # ==================================================
            # 计算 window 范围
            #
            # 例如：
            #
            # 当前句 = 3
            # WINDOW_RADIUS = 1
            #
            # start = 2
            # end   = 5
            #
            # 取：
            # sentence 2
            # sentence 3
            # sentence 4
            # ==================================================
            start_idx = max(0, sent_idx - WINDOW_RADIUS)
            end_idx = min(len(sentences), sent_idx + WINDOW_RADIUS + 1)

            # 提取 window 内容
            window_sentences = [
                s.strip()
                for s in sentences[start_idx:end_idx]
                if s.strip()
            ]

            # ==================================================
            # 文件命名
            #
            # 示例：
            #
            # window_1_Djama_el_Kebir_sent3_w3.md
            # ==================================================
            file_name = f"window_{q_idx}_{title_safe}_sent{sent_idx}_w3.md"
            file_path = OUTPUT_DIR / file_name

            # ==================================================
            # 文件内容
            #
            # 示例：
            #
            # Djamaâ el Kebir
            #
            # sentence 2
            #
            # sentence 3
            #
            # sentence 4
            # ==================================================
            content = (
                f"{title}\n\n"
                + "\n\n".join(window_sentences)
            )

            with open(file_path, "w", encoding="utf-8") as out_f:
                out_f.write(content)

            file_count += 1


# ==========================================================
# 6. 输出统计
# ==========================================================
print("=" * 60)
print("window 文件生成完成")
print("文件数量:", file_count)
print("输出目录:", OUTPUT_DIR)
print("=" * 60)