"""
将  HotpotQA 数据 embedding 后写入 Chroma，
用于 RAG retrieval evaluation。
"""

from pathlib import Path
import sys

# ==========================================================
# 将项目根目录加入 Python 搜索路径
#
# 作用：
#   保证可以 import services.xxx
#
# 当前文件：
#   feature/advanced-rag/embed_supported_sentences.py
#
# parent.parent.parent:
#   回到项目根目录 personal-kb-rag-api
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  
print(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))


# 导入项目组件
from services.chunker import split_text
from services.embedding import EmbeddingClient
from services.vector_store import ChromaVectorStore
from services.document_loader import load_document


# ==========================================================
# 1. 数据路径
#
# 这里读取你刚生成的 sentence-level corpus
# ==========================================================
# DATA_DIR = ROOT_DIR / "feature/advanced-rag/data/hotpotqa_supported_sentences_200"  
DATA_DIR = ROOT_DIR / "feature/advanced-rag/data/hotpotqa_supported_windows_200"


# ==========================================================
# 2. 初始化 embedding 模型
#
# 使用本地免费 embedding 模型：
#   sentence-transformers/all-MiniLM-L6-v2
#
# 输出维度：
#   384
# ==========================================================

# 擅长英文文本的轻量级模型-sentence-transformers/all-MiniLM-L6-v2
# embedder = EmbeddingClient(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

embedder = EmbeddingClient(
    model_name="BAAI/bge-small-en-v1.5"
)

# ==========================================================
# 3. 初始化 Chroma 向量库
#
# 单独开一个 sentence benchmark 数据库
#
# 避免污染你原来的：
#   chroma_db/documents
# ==========================================================
# vector_store = ChromaVectorStore(
#     persist_dir=r"D:\SWL_chroma_db\hotpot_supported_sentences_200",
#     collection_name="hotpot_supported_sentences"
# )

# vector_store = ChromaVectorStore(
#     persist_dir=r"D:\SWL_chroma_db_supported_sentences_200_bge",
#     collection_name="hotpot_supported_sentences_200_bge"
# )

vector_store = ChromaVectorStore(
    persist_dir=r"D:\SWL_chroma_db\hotpot_supported_windows_200_bge",
    collection_name="hotpot_supported_windows_200_bge"
)

# ==========================================================
# 4. 获取所有 sentence 文件
# ==========================================================
# 按数字顺序排序文件，例如 hotpot_1_para1.md, hotpot_1_para2.md ... hotpot_1_para10.md
import re

def natural_key(path):
    return [
        int(text) if text.isdigit() else text
        for text in re.split(r'(\d+)', path.name)
    ]

files = sorted(DATA_DIR.glob("*.md"), key=natural_key)

print("sentence 文件数量:", len(files))


# ==========================================================
# 5. batch embedding 配置
#
# 为什么 batch？
#
# 如果一次性 embedding 所有文件：
#   内存爆
#
# 所以：
#   每 500 个 chunks 写一次
# ==========================================================
BATCH_SIZE = 500

batch_chunks = []
processed_files = 0


# ==========================================================
# 6. 遍历 sentence 文件
# ==========================================================
for doc_file in files:

    # 加载 md 文件
    doc = load_document(str(doc_file))

    # sentence 文件本身已经很小
    # chunk_size 给大一点，避免再切碎
    chunks = split_text(
        text=doc["text"],
        source=doc["source"],
        chunk_size=1000,
        page=0
    )

    batch_chunks.extend(chunks)
    processed_files += 1

    # 满 batch 就写入
    if len(batch_chunks) >= BATCH_SIZE:

        texts = [chunk["text"] for chunk in batch_chunks]

        embeddings = embedder.embed_texts(texts)

        vector_store.add_chunks(
            chunks=batch_chunks,
            embeddings=embeddings
        )

        print(f"已处理文件数: {processed_files}")

        # 清空 batch
        batch_chunks = []


# ==========================================================
# 7. 处理最后剩余的数据
# ==========================================================
if batch_chunks:
    texts = [chunk["text"] for chunk in batch_chunks]
    embeddings = embedder.embed_texts(texts)

    vector_store.add_chunks(
        chunks=batch_chunks,
        embeddings=embeddings
    )

    print(f"已写入最后 {len(batch_chunks)} 个 chunks")


print("sentence-level embedding 完成")
print("chunk 总数:", vector_store.collection.count())
