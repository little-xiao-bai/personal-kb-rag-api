######################################################
# Part1: 测试加载 sample.txt 文件
# import sys
# import os
# # from pprint import pprint

# # 项目根目录
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, ROOT_DIR)


# from services.document_loader import load_text_file

# def test_load_sample():
#     # 测试加载 sample.txt 文件
#     file_path = os.path.join(ROOT_DIR, "data/sample.txt")

#     text = load_text_file(file_path)

#     assert text is not None
#     assert "测试文档" in text["text"]
#     assert text["metadata"]["file_type"] == "txt"

# # pprint(text)
# # print(text["text"])


##################################################
# Part:2 测试加载 sample.txt & sample.md 文件
# 文件路径: tests/test_document_loader_md.py
# import sys
# import os
# from pathlib import Path
# import pytest

# # 将项目根目录加入 sys.path，这样可以导入 services 包
# ROOT_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT_DIR))

# from services.document_loader import load_text_file

# @pytest.mark.parametrize("filename", [
#     "data/sample.txt",
#     "data/sample.md"  # 如果你有 Markdown 测试文件
# ])
# def test_load_text_file(filename):
#     # 加载文件
#     text = load_text_file(filename)

#     # 基本检查
#     assert text is not None, f"{filename} 返回 None"
#     assert "source" in text, "缺少 source 字段"
#     assert "text" in text, "缺少 text 字段"
#     assert "metadata" in text, "缺少 metadata 字段"

#     # metadata 检查
#     meta = text["metadata"]
#     assert "file_type" in meta, "metadata 缺少 file_type"
#     assert "chars" in meta, "metadata 缺少 chars"

#     # 文件长度检查
#     text_len = len(text["text"])
#     assert text_len == meta["chars"], f"{filename} text 长度与 metadata 不一致"

#     # 输出前 100 个字符，方便查看
#     print(f"\n文件: {filename}")
#     print("前 100 字符:", text["text"][:100])
#     print("文件类型:", meta["file_type"], "总字符数:", meta["chars"])


###############################################################
# Part3:  测试加载 sample.txt & sample.md & sample.pdf 文件
import sys
import os
from pathlib import Path
import pytest

# 将项目根目录加入 sys.path，保证可以导入 services 包
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from services.document_loader import load_document

@pytest.mark.parametrize("filename", [
    "data/sample.txt",
    "data/sample.md",
    "data/sample.pdf"
])
def test_load_document(filename):
    # 构造文件绝对路径
    file_path = ROOT_DIR / filename

    # 加载文件
    doc = load_document(file_path)

    # 基本检查
    assert doc is not None, f"{filename} 返回 None"
    assert "source" in doc, "缺少 source 字段"
    assert "text" in doc, "缺少 text 字段"
    assert "metadata" in doc, "缺少 metadata 字段"

    # metadata 检查
    meta = doc["metadata"]
    assert "file_type" in meta, "metadata 缺少 file_type"
    assert "chars" in meta, "metadata 缺少 chars"

    # 文件长度检查
    text_len = len(doc["text"])
    assert text_len == meta["chars"], f"{filename} text 长度与 metadata 不一致"

    # 打印前 100 个字符，方便调试
    print(f"\n文件: {filename}")
    print("文件类型:", meta["file_type"])
    print("总字符数:", meta["chars"])
    print("前 100 字符:\n", doc["text"][:100])