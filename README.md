# Naive RAG - 项目说明（初学者版）

> 本项目实现一个最小可运行的 RAG（Retrieval-Augmented Generation）系统，
> 支持本地文档上传、索引、向量化以及基于 DeepSeek 的问答生成。

---

## 01｜项目目录结构

```
personal-kb-rag-api/
│
├─ main.py                   # FastAPI 启动入口，定义所有 API 路由
├─ schemas.py                # Pydantic 请求/响应数据模型
├─ .env                      # 环境变量（API Key等）
├─ requirements.txt           # 项目依赖（如果有）
├─ README.md                 # 本文档
│
├─ data/                      # 测试文档
│   ├─ sample.txt
│   ├─ sample.md
│   └─ sample.pdf
│
├─ chroma_db/                 # Chroma 向量数据库持久化目录
│
├─ services/                  # 核心服务逻辑
│   ├─ __init__.py            # Python 包标记文件（当前为空）
│   ├─ document_loader.py     # 文档读取（TXT/MD/PDF）
│   ├─ chunker.py             # 文本切分为 chunks
│   ├─ embedding.py           # 文本向量化（SentenceTransformer）
│   ├─ vector_store.py        # Chroma 封装（add_chunks/query）
│   └─ llm.py                 # DeepSeek API 客户端封装
│
└─ tests/                     # 单元测试
    ├─ test_document_loader.py
    └─ test_chunker.py
```

---

## 02｜安装依赖

```bash
pip install fastapi uvicorn requests python-dotenv
pip install sentence-transformers chromadb pymupdf
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

---

## 03｜配置环境变量

项目根目录创建 `.env` 文件：

```text
DEEPSEEK_API_KEY=你的真实APIKey
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
```

---

## 04｜启动 FastAPI 服务

```bash
cd personal-kb-rag-api
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 05｜健康检查接口 /health

- **请求**: `GET /health`
- **响应**:

```json
{
  "status": "ok"
}
```

---

## 06｜聊天接口 /chat

- **请求**: `POST /chat`
- **请求体**:

```json
{
  "question": "什么是 FastAPI？",
  "model": "deepseek-v4-flash",
  "temperature": 0.7
}
```

- **响应体**:

```json
{
  "answer": "FastAPI 是一个现代 Python Web 框架...",
  "model": "deepseek-v4-flash",
  "error": null
}
```

- **失败示例**:

```json
{
  "answer": null,
  "model": "deepseek-v4-flash",
  "error": "API Key 未配置"
}
```

---

## 07｜文档上传接口 /documents/upload

- **请求**: `POST /documents/upload`
- **请求体**:

```json
{
  "file_paths": [
    "data/sample.pdf",
    "data/sample.md"
  ]
}
```

- **响应体**:

```json
{
  "registered_documents": [
    "sample.pdf",
    "sample.md"
  ],
  "failed_documents": [
    {
      "file_path": "data/not_exist.pdf",
      "reason": "文件不存在"
    }
  ],
  "documents_count": 2,
  "error": null
}
```

- **说明**:

  - 只负责将文档进入系统注册，不做 chunk / embedding / index
  - 支持批量上传
  - 成功/失败文件均有记录

---

## 08｜文档索引接口 /documents/index

- **请求**: `POST /documents/index`
- **请求体**:

```json
{
  "document_ids": ["sample.pdf", "sample.md"],
  "chunk_size": 500
}
```

- **工作流程**:

1. 根据 `document_id` 从 `DOCUMENT_REGISTRY` 获取文档
2. `split_text()` 按 `chunk_size` 切分文本
3. `embed_texts()` 获取向量
4. 写入 Chroma 向量库
5. 返回索引统计结果

- **响应体**:

```json
{
  "document_id": "sample.pdf, sample.md",
  "chunks_count": 72,
  "indexed_chunks": 72,
  "error": null
}
```

- **失败示例**:

```json
{
  "document_id": null,
  "chunks_count": 0,
  "indexed_chunks": 0,
  "error": "document_id 未注册"
}
```

---

## 09｜RAG 查询接口 /rag/query

- **请求**: `POST /rag/query`
- **请求体**:

```json
{
  "question": "What is FastAPI?",
  "top_k": 3
}
```

- **响应体**:

```json
{
  "answer": "FastAPI 是一个现代 Python Web 框架...",
  "sources": [
    {
      "chunk_id": "sample.pdf-p2-0",
      "source": "data/sample.pdf",
      "page": 2,
      "text_preview": "FastAPI 是一个现代 Python Web 框架...",
      "distance": 1.48
    }
  ],
  "error": null
}
```

- **说明**:

  - 返回 LLM 答案和 top-k chunks
  - `distance` 越小越相关
  - PDF chunks 保留 page 信息

- **失败示例**:

```json
{
  "answer": "当前资料不足以回答。",
  "sources": [],
  "error": null
}
```

---

## 10｜文档加载与 Chunk 测试

- 使用 `tests/` 文件夹的 pytest 测试：
  - `test_document_loader.py`
  - `test_chunker.py`
- 验证：
  - Document 字典完整性 (`source`, `text`, `metadata`)
  - PDF 分页信息 `pages` 是否保留
  - chunk 字段完整性 (`chunk_id`, `source`, `page`, `start`, `end`, `text`)
  - chunk 长度 <= `chunk_size`

- 运行方式：

```bash
pytest tests/
```

---

## 11｜工程经验

1. 接口职责单一
   - `/documents/upload` → 文档注册
   - `/documents/index` → 建立向量索引
   - `/rag/query` → 检索 + 回答
2. PDF 按页切分，TXT/MD 全文切分
3. 支持批量文件，部分失败不会影响整体
4. Pydantic 自动校验请求体字段
5. 避免 Python mutable default（使用 `Field(default_factory=list)`）
6. `distance` 字段语义正确（越小越相关）

---

## 12｜常见失败排查

- 文件未找到 → 检查 `file_paths`
- 文档未注册 → 先调用 `/documents/upload`
- API Key 未配置 → 检查 `.env`
- chunk 无法生成 → 检查 `split_text()` 参数
- 查询无结果 → 文档未索引或 top-k 设置过小

---

## 13｜运行总结

```bash
cd personal-kb-rag-api
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 14｜一句话总结

> `/documents/upload` → 注册文档  
> `/documents/index` → 建立向量索引  
> `/rag/query` → 查询 + LLM 回答  
> 职责清晰、PDF 支持 page 元数据、支持批量上传和部分失败，完整覆盖 Naive RAG MVP。