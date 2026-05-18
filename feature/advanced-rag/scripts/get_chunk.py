"""
作用：
检查 Chroma 向量数据库中的内容。

用途：
1. 查看当前 collection 的 chunk 总数
2. 抽样查看前几个 chunk 的：
   - chunk_id
   - source（来源文件）
   - text（文本内容）
3. 调试确认 embedding / indexing 是否成功写入数据库
"""

from pathlib import Path
import sys

# 项目根目录（用于导入 services 模块）
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from services.vector_store import ChromaVectorStore

# 连接指定的 Chroma 数据库
vector_store = ChromaVectorStore(
    persist_dir=r"D:\SWL_chroma_db\hotpot_supported_windows_200_bge",
    collection_name="hotpot_supported_windows_200_bge"
)

# 获取前 5 条 chunk 示例
result = vector_store.collection.get(limit=5)

# 打印数据库中的 chunk 总数
print("count:", vector_store.collection.count())

# 打印 chunk 详情
for i in range(len(result["ids"])):
    print("=" * 80)
    print("chunk_id:", result["ids"][i])
    print("source:", result["metadatas"][i]["source"])
    print("text:")
    print(result["documents"][i])