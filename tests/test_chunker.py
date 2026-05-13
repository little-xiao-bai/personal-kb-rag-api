import sys
import os
from pathlib import Path
import pytest

# 添加项目根目录到 sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from services.document_loader import load_document
from services.chunker import split_text  # 假设 split_text 放在 services/chunker.py

@pytest.mark.parametrize("filename", [
    "data/sample.txt",
    "data/sample.md",
    "data/sample.pdf"
])
def test_split_text(filename):
    # 构造文件绝对路径
    file_path = ROOT_DIR / filename

    # 加载文档
    doc = load_document(file_path)
    text = doc["text"]
    source = doc["source"]

    # 切分 chunk
    chunk_size = 500
    chunks = split_text(text, source, chunk_size=chunk_size)

    # 基本检查
    assert len(chunks) > 0, "未生成任何 chunk"
    for c in chunks:
        assert "chunk_id" in c
        assert "source" in c
        assert "start" in c
        assert "end" in c
        assert "text" in c
        # start/end 范围检查
        assert 0 <= c["start"] < c["end"] <= len(text)
        # chunk 文本长度不超过 chunk_size（最后一个 chunk 可以小于）
        assert len(c["text"]) <= chunk_size

    # 打印前 3 个 chunk 调试
    print(f"\n文件: {filename}")
    for i, c in enumerate(chunks[:3]):
        print(f"chunk {i} id: {c['chunk_id']}")
        print(f"来源: {c['source']}")
        print(f"起止: {c['start']}-{c['end']}")
        print(f"内容前50字符: {c['text'][:50]}\n")