"""
统计 HotpotQA 问题数量

用途：
    查看 HotpotQA 原始数据集中包含多少条问题

说明：
    hotpotqa_dev_fullwiki_pretty.json 本质上是一个 JSON 数组：

    [
        { question1 },
        { question2 },
        { question3 },
        ...
    ]

    所以：
        len(data)

    就等于问题总数。

预期：
    HotpotQA dev set 通常约 7405 条
"""

import json
from pathlib import Path

# ==========================================================
# 1. 路径配置
# ==========================================================
SCRIPT_DIR = Path(__file__).parent   #d:Projects\personal-kb-rag-api\feature\advanced-rag\scripts
ROOT_DIR = SCRIPT_DIR.parent        # d:Projects\personal-kb-rag-api\feature\advanced-rag


file_path = ROOT_DIR / "data" / "hotpotqa_dev_fullwiki_pretty.json"

print(file_path)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("问题总数:", len(data))