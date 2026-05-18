# FastAPI Web 框架
from fastapi import FastAPI

# API 请求/响应数据模型（Schema）
from schemas import (
    HealthResponse,      # /health 响应模型
    ChatRequest,         # /chat 请求模型
    ChatResponse,        # /chat 响应模型
    RAGQueryRequest,     # /rag/query 请求模型
    RAGQueryResponse,    # /rag/query 响应模型
    SourceChunk,         # RAG 来源 chunk 模型
    DocumentUploadRequest,       # 文档加载请求模型
    FailedDocument,              # 文档加载失败模型
    DocumentUploadResponse,      # 文档加载相应模型
    DocumentIndexRequest,         # 文档 chunk 请求模型
    DocumentIndexResponse        # 文档 chunk 相应模型
)

# 服务层组件
from services.llm import LLMClient              # 大模型调用客户端
from services.embedding import EmbeddingClient  # Embedding 客户端
from services.vector_store import ChromaVectorStore  # 向量数据库
from services.document_loader import load_document  # 文档加载器
from services.chunker import split_text          # 文本切 chunk 工具
from services.reranker import Reranker     # Reranker 重排序器

# 系统工具
import os
from pathlib import Path

# 读取 .env 环境变量
from dotenv import load_dotenv

import traceback


# =========================
# 创建 FastAPI 应用
# =========================
# 这是整个 Web 服务的入口
app = FastAPI()

# =========================
# 项目路径
# =========================
# 当前文件所在目录
ROOT_DIR = Path(__file__).resolve().parent


# =========================
# 环境变量加载
# =========================
# 读取 .env 文件
#
# 例如：
# DEEPSEEK_API_KEY=xxxx
# DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/chat/completions
load_dotenv() # 自动加载同级目录 .env 文件

API_KEY = os.environ["DEEPSEEK_API_KEY"]         # 获取 DeepSeek API Key
BASE_URL = os.environ["DEEPSEEK_BASE_URL"]       # 获取 DeepSeek API Base URL

# =========================
# 初始化服务组件
# =========================

# -------------------------
# 1. LLM 客户端
# -------------------------
# 用于普通聊天：
# /chat
# 初始化 LLMClient
llm_client = LLMClient(
    api_key=API_KEY, 
    base_url=BASE_URL,
    model = 'deepseek-v4-flash'
)

# -------------------------
# 2. Embedding 客户端
# -------------------------
# 用于：
# 文本 -> 向量
#
# 给 RAG 检索使用
# embedder = EmbeddingClient()

# embedder = EmbeddingClient(model_name = "sentence-transformers/all-MiniLM-L6-v2")

embedder = EmbeddingClient(
    model_name="BAAI/bge-small-en-v1.5"
)

# -------------------------
# 3. Chroma 向量数据库
# -------------------------
# 用于：
# 保存 chunks
# 向量检索
# vector_store = ChromaVectorStore(
#     persist_dir=str(ROOT_DIR / "chroma_db"), 
#     collection_name="documents")

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


# -------------------------
# 4. Reranker 重排序器
# -------------------------
reranker = Reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# 简单文档注册表
# 最小版先存在内存里，服务重启后会丢失
DOCUMENT_REGISTRY = {}



# ==========================================================
# /health
# ==========================================================
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    健康检查接口

    方法：
        GET

    用途：
        检查服务是否正常运行

    示例：
        GET /health

    返回：
        {
            "status": "ok"
        }
    """
    return HealthResponse(status="ok")

# ==========================================================
# /chat
# ==========================================================
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    普通聊天接口

    方法：
        POST

    用途：
        调用大模型进行普通对话

    请求示例：
        {
            "question": "什么是 FastAPI？",
            "model": "deepseek-v4-flash",
            "temperature": 0.7
        }

    返回：
        {
            "answer": "...",
            "model": "deepseek-v4-flash",
            "error": null
        }
    """
    try:
        # 调用 LLM
        answer = llm_client.chat(question=request.question, temperature=request.temperature)
        # 成功响应
        return ChatResponse(answer=answer, model=request.model)
    except Exception as e:
        # 异常响应
        return ChatResponse(answer=None, model=request.model, error=str(e))


# ==========================================================
# /rag/query
# ==========================================================
@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(request: RAGQueryRequest):
    """
    最小 RAG 查询接口

    工作流程：

    1. 接收用户问题
    2. query -> embedding
    3. Chroma 检索 top-k chunks
    4. 构造 sources
    5. 拼接 answer
    6. 返回结果

    请求示例：
    {
    "question": "What methods does the article use to solve the problem?",
    "top_k": 3
    }

    """
    try:
        #########################################################################################
        # =========================
        # # 1. query -> embedding
        # # =========================
        # # 把用户问题转成向量
        # query_embedding = embedder.embed_query(request.question)

        
        # # =========================
        # # 2. 向量检索
        # # =========================
        # # 在 Chroma 中找最相似 chunks
        # results = vector_store.search(query_embedding=query_embedding,top_k=request.top_k)

        # # =========================
        # # 3. 构造 sources
        # # =========================
        # # 返回给前端：
        # # 来源文件
        # # 文本预览
        # # 相似度分数
        # sources = []

        # for item in results:
        #     metadata = item.get("metadata", {})

        #     sources.append(
        #         SourceChunk(
        #             chunk_id=item.get("chunk_id"),

        #             # 来源文件
        #             source=metadata.get("source", ""),

        #             # 页码（如果有）
        #             page=metadata.get("page"),

        #             # 文本预览
        #             text_preview=item.get("text", "")[:500],

        #             # 相似度距离
        #             distance=item.get("distance")
        #         )
        #     )
            
        # # # =========================
        # # # 4. 拼接检索文本
        # # # =========================
        # # # 当前是最小版本：
        # # # 不调用 LLM
        # # # 直接把检索结果拼起来
        # # retrieved_text = "\n\n".join(
        # #     [
        # #          item.get("text","")
        # #          for item in results
        # #     ]
        # # )

        # # # 最终回答
        # # answer = (
        # #      "以下是根据本地知识库检索到的相关内容：\n\n"
        # #      + retrieved_text
        # # )


        #  # # =========================
        # # # 4. 拼接检索文本
        # # # =========================
        # # # 调用LLM生成更自然的回答
        # # =========================
        # # 4.1 检查是否检索到内容
        # # =========================
        # # 如果向量库没有返回任何结果
        # # 直接返回提示
        # if not results:
        #     return RAGQueryResponse(
        #         answer="没有找到相关内容。",
        #         sources=[],
        #         error=None
        #     )
        
        #  # =========================
        # # 4.2 构造上下文 context
        # # =========================
        # # 把检索到的 chunk 文本提取出来
        # #
        # # 目的：
        # # 给 LLM 提供参考资料
        # #
        # # 为什么限制 [:1000]？
        # #
        # # 防止 chunk 太长：
        # # - prompt 太大
        # # - token 成本高
        # # - 模型注意力分散
        # context_part = []

        # for item in results:
        #     text = item.get("text", "")[:1000]   # 提取文本  # 最多保留前 1000 字符
        #     context_part.append(text)

        # # 用空行拼接成完整上下文
        # #
        # # 最终类似：
        # #
        # # chunk1内容
        # #
        # # chunk2内容
        # #
        # # chunk3内容
        # context = "\n\n".join(context_part)
        ############################################################################

        # =========================
        # 1. query -> embedding
        # =========================
        # 把用户问题转成向量
        query_embedding = embedder.embed_query(request.question)

        # =========================
        # 2. Retriever 粗召回 top20
        # =========================
        retrieved_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k*4
        )

        # =========================
        # 3. Reranker 精排 top5
        # =========================
        reranked_results = reranker.rerank(
            query=request.question,
            chunks=retrieved_results,
            top_k=request.top_k
        )

        if not reranked_results:
            return RAGQueryResponse(
                answer="I don't know.",
                sources=[],
                error=None
            )

        # =========================
        # 4. 构造 sources
        # =========================
        sources = []

        for item in reranked_results:
            metadata = item.get("metadata", {})

            sources.append(
                SourceChunk(
                    chunk_id=item.get("chunk_id"),
                    source=metadata.get("source") or item.get("source", ""),
                    page=metadata.get("page") or item.get("page"),
                    text_preview=item.get("text", "")[:500],

                    # 临时用 distance 存 reranker_score
                    # 更推荐后面在 SourceChunk 里新增 reranker_score 字段
                    distance=item.get("reranker_score")
                )
            )

        # =========================
        # 5. 构造 context
        # =========================
        context = "\n\n".join(
            item.get("text", "")[:1000]
            for item in reranked_results
        )

        # =========================
        # 4.3 构造 RAG Prompt
        # =========================
        # Prompt Engineering：
        # 明确告诉模型：
        #
        # 1. 只能基于提供资料回答
        # 2. 不允许自由发挥 / 幻觉
        # 3. 没答案就明确说明
        # prompt = f"""
        # 请只根据以下资料回答问题。
        # 如果资料中没有答案，请明确说“当前资料不足以回答”。

        # 资料：
        # {context}

        # 问题：
        # {request.question}
        # """

        prompt = f"""
        Answer in English only.

        Use ONLY the retrieved context to answer.

        If the answer cannot be determined from the context, say:
        I don't know.

        Context:
        {context}

        Question:
        {request.question}
        """

        # =========================
        # 4.4. 调用 LLM 生成答案
        # =========================
        #
        # temperature=0.3：
        # 较低随机性
        #
        # 原因：
        # RAG 更强调准确性
        # 不希望模型胡乱发挥
        answer = llm_client.chat(question=prompt, temperature=0.3)

        # =========================
        # 5. 返回成功响应
        # =========================
        return RAGQueryResponse(
            answer=answer,
            sources=sources,
            error=None
        )

    except Exception as e:
        # 返回错误
        return RAGQueryResponse(
            answer=None,
            sources=[],
            error=str(e)
        )
     





# # ==========================================================
# # /documents/index
# # ==========================================================
# @app.post("/documents/index", response_model=DocumentIndexResponse)
# def index_document(request: DocumentIndexRequest):
#     """
#     文档索引接口(支持多文档)

#     工作流程：

#     1. 接收多个文件路径
#     2. 逐个加载文档内容:  load_document()
#     3. 逐个切分成 chunks: split_text()
#     4. 合并所有生成 chunks
#     5. 统一生成 embeddings: embed_texts()
#     6. 存入 Chroma 向量库: vector_store.add_chunks()
#     7. 返回索引结果
#     """
#     try:
#         all_chunks = []
#         document_ids = []

#         # 1.遍历多个文件路径
#         for file_path_str in request.file_paths:
#             file_path = ROOT_DIR / Path(file_path_str)

#             # 检查文件是否存在
#             if not file_path.exists():
#                 raise FileNotFoundError(f"文件未找到: {f}")

#             # 2.加载文档内容
#             doc =load_document(str(file_path))
#             document_ids.append(Path(doc["source"]).name)  # 文档 ID 可以用文件名

#             # 3. PDF：按页切分，保留 page
#             if doc["metadata"]["file_type"] == "pdf":
#                 for page_item in doc["pages"]:
#                     page_chunks = split_text(
#                         text=page_item["text"],
#                         source=doc["source"],
#                         chunk_size=request.chunk_size,
#                         page=page_item["page"]
#                     )
#                     all_chunks.extend(page_chunks)
           
#             # 4. TXT / MD：按全文切分
#             else:
#                 chunks = split_text(
#                     text=doc["text"],
#                     source=doc["source"],
#                     chunk_size=request.chunk_size,
#                     page = 0
#                 )
#                 all_chunks.extend(chunks)

#         # 5. 如果没有生成任何 chunk
#         if not all_chunks:
#             return DocumentIndexResponse(
#                 document_id=", ".join(document_ids),
#                 chunks_count=0,
#                 indexed_count=0,
#                 error="没有生成任何 chunk"
#             )
        
#         # 6. 生成 embeddings
#         texts = [chunk["text"] for chunk in all_chunks]
#         embeddings = embedder.embed_texts(texts)

#         # 7. 存入 Chroma 向量库
#         vector_store.add_chunks(
#             chunks=all_chunks,
#             embeddings=embeddings)
        
#         # 8.返回结果

#         return DocumentIndexResponse(
#             document_id=", ".join(document_ids),
#             chunks_count=len(all_chunks),
#             indexed_chunks=len(all_chunks),
#             error=None
#         )
#     except Exception as e:
#         return DocumentIndexResponse(
#             document_id=None,
#             chunks_count=0,
#             indexed_chunks=0,
#             error=str(e)
#         )
    
#######################################################################
#     将读文件 和 索引分开 新增/upload,重写/index                       #                               
#######################################################################

# ==========================================================
# /documents/upload
# ==========================================================
@app.post("/documents/upload",response_model=DocumentUploadResponse)
def upload_document(request:DocumentUploadRequest):
    """
    文档上传 / 注册接口

    工作流程：

    1. 接收多个文件路径
    2. 检查文件是否存在
    3. load_document()
    4. 注册到 DOCUMENT_REGISTRY
    5. 返回成功 / 失败结果

    注意：
    本接口只负责文档进入系统，不做 chunk / embedding / index

    请求示例：
    {
        "file_paths": [
            "data/sample.md",
            "data/sample.txt",
            "data/sample.pdf"
        ]
    }

    返回：
    {
        "registered_documents": [
            "sample.md",
            "sample.txt",
            "sample.pdf"
        ],
        "failed_documents": [],
        "documents_count": 3,
        "error": null
    }
    """

    registered_documents = []
    failed_documents = []

    for file_path_str in request.file_paths:
        try:
            # 拼接项目根目录
            file_path = ROOT_DIR / Path(file_path_str)

            # 文件检查是否存在
            if not file_path.exists():
                raise FileNotFoundError("文件不存在")

            doc = load_document(str(file_path))

            # 生成 document_id
            document_id = Path(doc["source"]).name

            if document_id in DOCUMENT_REGISTRY:
                print(f"已注册，跳过: {document_id}")
                continue

            # 注册到内存 registry
            DOCUMENT_REGISTRY[document_id] = doc

            # 记录成功
            registered_documents.append(document_id)

        except Exception as e:
            failed_documents.append(
                FailedDocument(
                    file_path=file_path_str,
                    reason = str(e)
                )
            )

    return DocumentUploadResponse(
        registered_documents=registered_documents,
        failed_documents=failed_documents,
        documents_count=len(registered_documents),
        error=None
    )

@app.post("/documents/index",response_model=DocumentIndexResponse)
def index_document(request:DocumentIndexRequest):
    """
    文档索引接口

    只负责：
    1. 根据 document_id 从 DOCUMENT_REGISTRY 找文档
    2. split_text()
    3. embed_texts()
    4. vector_store.add_chunks()
    5. 返回索引结果

    请求示例：
    {
        "document_ids": [
            "sample.md",
            "sample.txt",
            "sample.pdf"
        ],
        "chunk_size": 500
    }

    返回：
    {
        "registered_documents": [
            "data/sample.md",
            "data/sample.txt",
            "data/sample.pdf"
        ],
        "failed_documents": [],
        "documents_count": 3,
        "error": null
    }
    """

    try:
        all_chunks = []
        indexed_document_ids = []
        

        for document_id in request.document_ids:
            # 每个文档单独判断
            document_added = False

            # 1. 检查是否已注册
            if document_id not in DOCUMENT_REGISTRY:
                raise ValueError(f"document_id 未注册，请先调用 /documents/upload: {document_id}")
            
            print(f"\n开始处理文档: {document_id}")

            doc = DOCUMENT_REGISTRY[document_id]

            # PDF: 按页切分，保留 page
            if doc["metadata"]["file_type"] == "pdf":
                print("PDF 文档")
                
                for page_item in doc["pages"]:
                    page_chunks = split_text(
                        text = page_item["text"],
                        source=doc["source"],
                        chunk_size=request.chunk_size,
                        page=page_item["page"]
                    )
                    # 如果当前页没有 chunk
                    if not page_chunks:
                        continue

                    # 用第一页 chunk 测试是否已索引
                    test_chunk_id = page_chunks[0]["chunk_id"]

                    existing = vector_store.collection.get(ids=[test_chunk_id])   # 查询 Chroma 看这个 chunk_id 是否已经存在了

                    if existing["ids"]:
                        print(f"已索引，跳过: {document_id} page {page_item['page']}")
                        continue
                    
                    # 当前页加入索引队列
                    all_chunks.extend(page_chunks)
                    document_added = True
                    
            

            # TXT / MD：按全文切分
            else:
                print("TXT / MD 文档")

                chunks = split_text(
                    text=doc["text"],
                    source=doc["source"],
                    chunk_size=request.chunk_size,
                    page=0
                )

                if not chunks:
                    continue

                test_chunk_id = chunks[0]["chunk_id"]

                existing = vector_store.collection.get(ids=[test_chunk_id])

                if existing and existing.get("ids"):
                    print(f"已索引，跳过: {document_id}")
                    continue

                all_chunks.extend(chunks)
                document_added = True

            # 只有真的加入了 chunk 才记录
            if document_added:
                indexed_document_ids.append(document_id)

        # ==================================================
        # 没有任何新 chunk
        # ==================================================
        if not all_chunks:
            return DocumentIndexResponse(
            document_id=", ".join(indexed_document_ids),
            chunks_count=0,
            indexed_chunks=0,
            error="没有新的 chunk 需要索引（可能都已存在）"
        )

        print("\n开始生成 embeddings...")
        # 2. embedding
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = embedder.embed_texts(texts)

        print("开始写入 Chroma...")
        # 3. 写入向量库
        vector_store.add_chunks(
            chunks=all_chunks,
            embeddings=embeddings,
            )
        
        # 4. 返回成功
        return DocumentIndexResponse(
            document_id=", ".join(indexed_document_ids),
            chunks_count=len(all_chunks),
            indexed_chunks=len(all_chunks),
            error=None
        )
    except Exception as e:
        print("⚠️ /documents/index 出错！")
        traceback.print_exc()   # 打印完整 traceback 到控制台
        return DocumentIndexResponse(
            document_id=None,
            chunks_count=0,
            indexed_chunks=0,
            error=str(e)
        )


# ==========================================================
# /chunks
# ==========================================================
@app.get("/chunks")
def list_chunks(limit: int = 20, offset: int = 0):
    """
    分页查看当前 Chroma 中的 chunk 接口

    方法：
        GET

    用途：
        分页查看当前向量库中已有的 chunk

    适用场景：
        - 检查 index 是否成功
        - 查看当前有哪些 chunk
        - 调试 chunk_id 命名
        - 统计 chunk 数量
        - 避免一次性读取全部 chunk 导致 too many SQL variables

    参数：
        limit  : 每次返回多少个 chunk，默认 20
        offset : 从第几个 chunk 开始返回，默认 0

    请求示例：
        GET /chunks
        GET /chunks?limit=20&offset=0
        GET /chunks?limit=20&offset=20

    返回示例：
        {
            "total_chunks": 1076,
            "limit": 20,
            "offset": 0,
            "chunks": [
                {
                    "chunk_id": "hotpot_1_para1.md-p0-0",
                    "metadata": {
                        "source": "...",
                        "page": 0,
                        "start": 0,
                        "end": 500
                    },
                    "text_preview": "chunk 前 200 字符..."
                }
            ]
        }
    """

    try:
        # 只统计总数，不一次性读取所有 chunk
        total_chunks = vector_store.collection.count()

        # 分页读取 chunk，避免一次性 get 全部导致 SQLite 报错
        result = vector_store.collection.get(
            limit=limit,
            offset=offset
        )

        chunks = []

        for i in range(len(result["ids"])):
            chunks.append({
                "chunk_id": result["ids"][i],
                "metadata": result["metadatas"][i],
                "text_preview": result["documents"][i][:200]
            })

        return {
            "total_chunks": total_chunks,
            "limit": limit,
            "offset": offset,
            "chunks": chunks
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# ==========================================================
# /chunks/{chunk_id}
# ==========================================================
@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    """
    查看指定 chunk 的完整内容接口

    方法：
        GET

    用途：
        根据 chunk_id 查询 Chroma 向量库中的完整 chunk 内容

    适用场景：
        - 调试 RAG 检索结果
        - 查看某个 chunk 的完整文本
        - 检查 chunk 切分是否合理
        - 检查 chunk overlap 是否正确
        - 排查检索质量问题

    请求示例：
        GET /chunks/hotpot_7_para1.md-p0-0

    返回示例（成功）：
        {
            "chunk_id": "hotpot_7_para1.md-p0-0",
            "text": "完整 chunk 内容...",
            "metadata": {
                "source": "data/hotpotqa_paragraphs/hotpot_7_para1.md",
                "page": 0,
                "start": 0,
                "end": 500
            }
        }

    返回示例（失败）：
        {
            "error": "chunk 不存在"
        }
    """

    try:
        # 从 Chroma 根据 chunk_id 查询
        result = vector_store.collection.get(ids=[chunk_id])

        # 如果没找到
        if not result or not result.get("ids"):
            return {
                "error": "chunk 不存在"
            }

        # 返回 chunk 完整内容
        return {
            "chunk_id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0]
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# ==========================================================
# 本地启动
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    # 启动 FastAPI 服务
    uvicorn.run(
        "main:app",

        # 本地地址
        host="127.0.0.1",

        # 端口
        port=8000,

        # 热更新
        reload=True
    )
    
