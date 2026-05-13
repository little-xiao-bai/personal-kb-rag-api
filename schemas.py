# Pydantic 基类
# 用于定义数据模型（Data Model）
# 功能：
# 1. 自动类型校验（Type Validation）
# 2. 自动数据转换（如 str -> int）
# 3. 自动生成 JSON Schema
# 4. 给 FastAPI 自动生成 Swagger API 文档
from pydantic import BaseModel,Field

# =========================
# 健康检查响应模型
# =========================
# 对应接口：
#   GET /health
#
# 作用：
#   用于服务健康检查（Health Check）
#   前端 / 运维 / Docker / K8s 可通过此接口判断服务是否正常运行
#
# 示例：
# {
#   "status": "ok"
# }
class HealthResponse(BaseModel):
    status: str   # 服务状态（通常为 "ok"）

# =========================
# 聊天请求模型
# =========================
# 对应接口：
#   POST /chat
#
# 作用：
#   定义用户向 LLM 提问时提交的请求体
#
# 示例：
# {
#   "question": "什么是 FastAPI？",
#   "model": "deepseek-v4-flash",
#   "temperature": 0.7
# }
class ChatRequest(BaseModel):
    question: str                        # 用户问题
    model: str = "deepseek-v4-flash"     # 使用的模型 # 默认 DeepSeek Flash
    temperature: float = 0.7             # 生成文本的随机程度，范围 [0.0, 1.0]，默认 0.7


# =========================
# 聊天响应模型
# =========================
# 对应接口：
#   POST /chat
#
# 作用：
#   统一聊天接口返回结构
#
# 成功示例：
# {
#   "answer": "FastAPI 是一个现代 Python Web 框架...",
#   "model": "deepseek-v4-flash",
#   "error": null
# }
#
# 失败示例：
# {
#   "answer": null,
#   "model": "deepseek-v4-flash",
#   "error": "API 调用失败"
# }
class ChatResponse(BaseModel):
    answer: str | None = None          # AI 回复文本，失败时为 null
    model: str = "deepseek-v4-flash"   # 实际使用的模型
    error: str | None = None           # 错误信息，成功时为 null


# =========================
# RAG 来源 chunk 模型
# =========================
# 作用：
#   表示检索阶段找到的一个文档片段（chunk）
#
# 用途：
#   返回给前端，让用户看到答案来自哪些文档
#
# 示例：
# {
#   "chunk_id": "chunk_001",
#   "source": "data/sample.pdf",
#   "page": 3,
#   "text_preview": "FastAPI 是一个现代 Python Web 框架...",
#   "score": 0.92
# }
class SourceChunk(BaseModel):
    chunk_id: str                    # chunk 唯一 ID
    source: str                      # 来源文件路径
    page: int | None = None          # 页码；PDF 有页码，TXT / MD 为 null
    text_preview: str                # 文本预览（前 100 字）
    distance: float | None = None       # 相似度分数，# Chroma distance；越小越相似，失败时为 null


# =========================
# RAG 查询请求模型
# =========================
# 对应接口：
#   POST /rag/query
#
# 作用：
#   定义用户查询知识库时的请求结构
#
# 工作流程：
#   用户问题
#      ↓
#   Embedding
#      ↓
#   向量检索
#      ↓
#   返回最相关 chunks
#      ↓
#   拼接上下文
#      ↓
#   交给 LLM 生成最终答案
#
# 示例：
# {
#   "question": "什么是 FastAPI？",
#   "top_k": 3
# }
class RAGQueryRequest(BaseModel):
    question: str        # 用户问题
    top_k: int = 3       # 返回最相关的 chunk 数量，默认 3

# =========================
# RAG 查询响应模型
# =========================
# 对应接口：
#   POST /rag/query
#
# 作用：
#   返回 RAG 最终答案 + 来源文档
#
# 示例：
# {
#   "answer": "...",
#   "sources": [...],
#   "error": null
# }
##########################
# 初版，会有BUG
# class RAGQueryResponse(BaseModel):
#     answer: str | None = None         # 最终 LLM 回答
#     sources: list[SourceChunk] = []   # 检索到的相关 chunk 列表
#     error: str | None = None          # 错误信息，成功时为 null

class RAGQueryResponse(BaseModel):
    answer: str | None = None         # 最终 LLM 回答
    sources: list[SourceChunk] = Field(default_factory=list)   # 检索到的相关 chunk 列表
    error: str | None = None          # 错误信息，成功时为 null


# =========================
# 文档上传请求模型
# =========================
# 对应接口：
#   POST /documents/upload
#
# 作用：
#   定义文档上传（注册）请求结构
#
# 说明：
#   当前这里的“上传”本质上是告诉系统：
#   “这些文件需要加入知识库处理流程”
#
# 如果你现在是本地文件模式：
#   上传 = 提供本地文件路径
#
# 如果以后接前端文件上传：
#   这里可能会改成 UploadFile
#
# 示例：
# {
#   "file_paths": [
#       "data/sample.pdf",
#       "data/manual.txt"
#   ]
# }
class DocumentUploadRequest(BaseModel):
    file_paths: list[str]             # 要上传的文件路径列表

# =========================
# 上传失败文档模型
# =========================
# 作用：
#   表示单个上传失败的文件信息
#
# 用途：
#   当批量上传多个文件时，
#   某些文件可能成功，某些失败
#
# 示例：
# {
#   "file_path": "data/missing.pdf",
#   "reason": "文件不存在"
# }
class FailedDocument(BaseModel):
    file_path: str        # 上传失败的文件路径
    reason: str           # 失败原因

# =========================
# 文档上传响应模型
# =========================
# 对应接口：
#   POST /documents/upload
#
# 作用：
#   返回文档上传（注册）结果
#
# 设计思路：
#   支持批量上传
#   因此可能出现：
#   - 全部成功
#   - 部分成功
#   - 全部失败
class DocumentUploadResponse(BaseModel):
    registered_documents: list[str] = Field(default_factory=list)           # 成功注册的文档路径列表
    failed_documents: list[FailedDocument] = Field(default_factory=list)    # 上传失败的文档列表
    documents_count: int = 0    # 成功上传（注册）的文档数量
    error: str | None = None    # 全局错误信息

# =========================
# 文档索引请求模型
# =========================
# 对应接口：
#   POST /documents/index
#
# 作用：
#   定义文档入库（索引）请求结构
#
# 工作流程：
#   文件路径
#      ↓
#   加载文档
#      ↓
#   文本切分
#      ↓
#   Embedding
#      ↓
#   写入向量数据库
#
# 示例：
# {
#   "file_paths": [
#       "data/sample.pdf",
#       "data/manual.txt"
#   ],
#   "chunk_size": 500
# }
class DocumentIndexRequest(BaseModel):
    document_ids: list[str]            # 要索引的文件路径列表
    chunk_size: int = 500                # 每个 chunk 的最大字符数，默认 500


# =========================
# 文档索引响应模型
# =========================
# 对应接口：
#   POST /documents/index
#
# 作用：
#   返回文档索引结果
#
# 成功示例：
# {
#   "document_id": "doc_001",
#   "chunks_count": 12,
#   "indexed_chunks": 12,
#   "error": null
# }
#
# 失败示例：
# {
#   "document_id": null,
#   "chunks_count": 0,
#   "indexed_chunks": 0,
#   "error": "文件不存在"
# }
class DocumentIndexResponse(BaseModel):
    document_id: str | None = None          # 文档唯一 ID，成功时返回
    chunks_count: int = 0
    indexed_chunks: int = 0               # 成功索引的 chunk 数量
    error: str | None = None           # 错误信息，成功时为 null