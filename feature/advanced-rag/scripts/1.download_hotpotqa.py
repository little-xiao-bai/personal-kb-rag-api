"""
作用：
下载 HotpotQA dev fullwiki 数据集，并生成一个格式化（pretty）的 JSON 文件方便查看。

用途：
1. 从官方 URL 下载原始 HotpotQA 数据集
2. 保存到本地 data 目录
3. 将原始紧凑 JSON 转成可读性更好的格式化 JSON
4. 方便后续数据分析、调试、构造测试集
"""

import requests
from pathlib import Path
import json


# 当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent

# 原始数据保存路径
download_file = SCRIPT_DIR.parent / "data" / "hotpotqa_dev_fullwiki.json"

# 如果 data 目录不存在则自动创建
download_file.parent.mkdir(parents=True, exist_ok=True)


# HotpotQA 官方下载地址
url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"

print("Downloading HotpotQA dev fullwiki...")

# 流式下载，避免一次性占用过多内存
r = requests.get(url, stream=True)
r.raise_for_status()

# 保存原始 JSON 文件
with open(download_file, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Downloaded to {download_file}")


# 格式化后的 JSON 输出路径
output_file = SCRIPT_DIR.parent / "data" / "hotpotqa_dev_fullwiki_pretty.json"

# 读取原始 JSON
with open(download_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 保存为格式化 JSON（方便阅读）
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        indent=2,           # 缩进
        ensure_ascii=False  # 保留 unicode 字符
    )

print(f"Saved pretty JSON to {output_file}")