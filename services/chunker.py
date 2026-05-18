# services/chunker.py
# ==========================
# split_text 按句子切分 + chunk_overlap 版本
# ==========================
# 说明：
# - 文本先按句子分割（使用 NLTK sent_tokenize）
# - 每个 chunk 长度接近 chunk_size，但不会切断句子
# - chunk 可以设置 chunk_overlap，保证前后 chunk 有上下文重叠
# - PDF 文件按页切分；MD/TXT 文件按全文切分
# - chunk_id 由文件名 + 页码 + 索引生成，确保唯一
# - 优点：上下文完整，检索结果更自然，LLM 生成答案质量更好
# - 缺点：稍慢于固定长度切分，chunk 数量可能略多
from pathlib import Path
from typing import List, Dict

import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('punkt_tab', download_dir=r'D:\Software\Anaconda\envs\langchain_new\nltk_data')


def split_text(text: str, source: str, chunk_size: int = 500, page: int | None=None, chunk_overlap: int = 50) -> List[Dict]:
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
    sentences = sent_tokenize(text)  # 使用 NLTK 的句子分割器

    current_chunk = ""  # 当前 chunk 的文本内容
    start_idx = 0  # 当前 chunk 在原文本中的起始位置

    for i, sent in enumerate(sentences):
        if len(current_chunk) + len(sent) > chunk_size:
            # 当前 chunk 已满，保存它
            end_idx = start_idx + len(current_chunk)
            chunk_id = f"{Path(source).name}-p{page}-{len(chunks)}" if page is not None else f"{Path(source).name}-{len(chunks)}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "start": start_idx,
                "end": end_idx,
                "text": current_chunk.strip()
            })

            # 重叠处理
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else "" # 获取当前 chunk 的最后 chunk_overlap 字符作为重叠部分
            current_chunk = overlap_text + sent   # 将重叠部分和当前句子一起放入新的 chunk
            # print("chunk_overlap类型：", type(chunk_overlap))
            start_idx = end_idx - chunk_overlap  # 新 chunk 的起始位置是上一个 chunk 的结束位置减去重叠部分的长度
        else:
            # 当前 chunk 还未满，继续添加句子
            current_chunk += " " + sent
        
        

    if current_chunk.strip():  # 添加最后一个 chunk
        end_idx = start_idx + len(current_chunk)
        chunk_id = f"{Path(source).name}-p{page or 0}-{len(chunks)}" if page is not None else f"{Path(source).name}-{len(chunks)}"
        
        chunks.append({
            "chunk_id": chunk_id,
            "source": source,
            "page": page,
            "start": start_idx,
            "end": end_idx,
            "text": current_chunk.strip()
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
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para1.md",
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para2.md",
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para3.md",
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para4.md",
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para5.md",
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para6.md",
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para7.md",
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para8.md",
    ROOT_DIR / "data" / "hotpotqa_paragraphs"/ "hotpot_1_para9.md"
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
                print(f"正在处理文档: {f.name}")
                # print(f"doc['text'] 类型: {type(doc['text'])}")
                chunks = split_text(
                    text=doc["text"],
                    source=doc["source"],
                    chunk_size=1000,
                    page=None
                )


            print(f"\n文件: {f}")
            print("文件类型:", doc["metadata"]["file_type"])
            print("总 chunk 数量:", len(chunks))

            for i, c in enumerate(chunks[:3]):  # 打印前3个 chunk
                print(f"\nchunk {i} id: {c['chunk_id']}")
                print(f"来源: {c['source']}")
                print(f"页码: {c['page']}")
                print(f"起止: {c['start']}-{c['end']}")
                print(f"内容前500字符: {c['text'][:500]}")

        except Exception as e:
            print(f"加载或切分 {f} 出错: {e}")