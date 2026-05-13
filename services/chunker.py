# services/chunker.py
from pathlib import Path
from typing import List, Dict

def split_text(text: str, source: str, chunk_size: int = 500, page: int | None=None) -> List[Dict]:
    """
    将文本切分为 chunk，每个 chunk 包含以下字段：
        - chunk_id : 唯一 ID
        - source   : 文档来源
        - page     : 页码，PDF 有页码；TXT / MD 为 None
        - start    : 在原文本中的起始位置
        - end      : 在原文本中的结束位置
        - text     : chunk 内的文本内容

    参数：
        text      : 要切分的文本
        source    : 文档来源路径
        chunk_size: 每个 chunk 的最大字符数，默认 500

    返回：
        chunks : List[Dict] 切分好的 chunks
    """
    chunks = []
    for start in range(0, len(text), chunk_size):
        end = start + chunk_size
        chunk_text = text[start:end]

         # 如果有 page，就把 page 放进 chunk_id，方便定位 PDF 页
        if page is not None:
            chunk_id = f"{Path(source).name}-p{page}-{len(chunks)}"
        else:
            chunk_id = f"{Path(source).name}-{len(chunks)}"

        chunks.append({
            "chunk_id": chunk_id,
            "source": source,
            "page": page,   
            "start": start,
            "end": min(end, len(text)),
            "text": chunk_text
        })
    return chunks


# ==========================
# 测试 / 示例调用
# ==========================
if __name__ == "__main__":
    import sys
    # from pprint import pprint

    # 将项目根目录加入 sys.path，保证可以导入 services 包
    ROOT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT_DIR))

    # print(ROOT_DIR)

    from services.document_loader import load_document

    # 示例：加载文件
    # file_paths = ["data/sample.txt", "data/sample.md", "data/sample.pdf"]
    file_paths = [
    ROOT_DIR / "data/sample.txt",
    ROOT_DIR / "data/sample.md",
    ROOT_DIR / "data/sample.pdf"
    ]
  

    print(file_paths)

    for f in file_paths:
        try:
            doc = load_document(f)

            # chunks = split_text(doc["text"], doc["source"], chunk_size=500)

            # ==========================
            # 如果是 PDF：按页切分
            # ==========================
            if doc["metadata"]["file_type"] == "pdf":
                chunks = []

                for page_item in doc["pages"]:
                    page_chunks = split_text(
                        text=page_item["text"],
                        source=doc["source"],
                        chunk_size=500,
                        page=page_item["page"]
                    )
                    chunks.extend(page_chunks)

            # ==========================
            # 如果是 TXT / MD：直接按全文切分
            # ==========================
            else:
                chunks = split_text(
                    text=doc["text"],
                    source=doc["source"],
                    chunk_size=500
                )


            print(f"\n文件: {f}")
            print("文件类型:", doc["metadata"]["file_type"])
            print("总 chunk 数量:", len(chunks))

            for i, c in enumerate(chunks[:3]):  # 打印前3个 chunk
                print(f"\nchunk {i} id: {c['chunk_id']}")
                print(f"来源: {c['source']}")
                print(f"页码: {c['page']}")
                print(f"起止: {c['start']}-{c['end']}")
                print(f"内容前50字符: {c['text'][:50]}")

        except Exception as e:
            print(f"加载或切分 {f} 出错: {e}")