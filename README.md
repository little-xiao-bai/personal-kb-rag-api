# Personal KB RAG API

> 从零构建的 Retrieval-Augmented Generation（RAG）项目：从 Naive RAG MVP 演进到 Advanced RAG。

---

# 项目简介

这是一个基于 **FastAPI + Chroma + Embedding + DeepSeek** 构建的完整 RAG（Retrieval-Augmented Generation）项目。

项目最初从一个：

```text
最小可运行 Naive RAG MVP
```

开始：

```text
用户问题
    ↓
Embedding
    ↓
向量检索
    ↓
Top-K Chunks
    ↓
LLM 回答
```

随后逐步升级为：

```text
Advanced RAG
```

最终架构：

```text
用户问题
    ↓
BGE Query Embedding
    ↓
Chroma Retrieve Top100
    ↓
CrossEncoder Rerank Top25
    ↓
Context Construction
    ↓
DeepSeek Grounded Generation
    ↓
Answer + Sources
```

---

# 项目目标

本项目目标：

---

## 1. 理解 RAG 完整闭环

从零实现：

- 文档加载
- 文本切分
- Embedding
- 向量数据库
- Retrieval
- Prompt Construction
- LLM Generation
- API 封装

---

## 2. 理解 FastAPI 在 LLM 系统中的作用

实现：

- API routing
- request validation
- response schema
- service abstraction
- production API design

---

## 3. 系统优化 Retriever

从：

```text
Naive Retrieval
```

升级到：

```text
Advanced Retrieval
```

包括：

- 更强 embedding
- sentence window retrieval
- reranking
- retrieval benchmarking

---

## 4. 构建可扩展工程结构

项目不是 notebook demo。

而是：

```text
工程化 API 项目
```

---

# 技术栈

## Backend

- FastAPI
- Uvicorn
- Pydantic

---

## LLM

- DeepSeek API

---

## Embedding

Naive：

- Sentence Transformers

Advanced：

- BAAI/bge-small-en-v1.5

---

## Vector Database

- ChromaDB

---

## Document Processing

- PyMuPDF
- TXT / MD loader

---

## Evaluation

- HotpotQA
- Retrieval Benchmark
- CSV Analysis

---

## Advanced Retrieval

- Sentence Window Retrieval
- CrossEncoder Reranker

---

# 项目演进

# Phase 0：基础准备

先完成：

- FastAPI
- Pydantic
- HTTP 请求流程
- LLM API 调用
- Embedding 基础
- Chroma 基础

目标：

理解：

```text
一个 RAG 系统到底由哪些组件组成
```

---

# Phase 1：Naive RAG MVP

实现最小可运行版本：

流程：

```text
Question
    ↓
Embedding
    ↓
Chroma Top-K
    ↓
DeepSeek
    ↓
Answer
```

功能：

- /health
- /chat
- /documents/upload
- /documents/index
- /rag/query

---

特点：

- 文档上传
- 文档索引
- chunking
- embedding
- vector retrieval
- DeepSeek generation

---

这是：

```text
最小闭环 MVP
```

---

# Phase 2：评估体系

发现：

```text
回答错
```

但不知道原因。

于是加入：

评估：

---

## End-to-End

看：

```text
最终答案对不对
```

指标：

```text
Answer Hit
```

---

## Retrieval-only

看：

```text
Retriever 有没有找到证据
```

指标：

```text
Retrieval Hit
```

---

这样可以明确：

```text
Retriever 问题
还是 Generator 问题
```

---

# Phase 3：Advanced Retrieval

对 Retriever 做优化。

---

## 更强 Embedding

从：

```text
基础 embedding
```

升级：

```text
BGE
```

模型：

```text
BAAI/bge-small-en-v1.5
```

---

## Sentence Window Retrieval

问题：

单句：

```text
上下文太弱
```

方案：

```text
前一句 + 当前句 + 后一句
```

---

## Reranker

问题：

向量检索：

```text
快
但粗
```

方案：

```text
retrieve top100
→ rerank top25
```

---

模型：

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

# Phase 4：FastAPI 集成

最终把实验成果接回：

```text
POST /rag/query
```

生产流程：

```text
query
→ embed_query()
→ retrieve top100
→ rerank top25
→ context build
→ DeepSeek
→ response
```

---

# 当前成果

最终：

```text
Naive RAG
→
Advanced RAG
```

性能提升：

Retrieval：

```text
82.5% → 94.5%
```

Answer：

```text
86.5% → 92.5%
```

---

# 项目结构概览

```text
personal-kb-rag-api/
│
├── main.py
├── schemas.py
├── services/
│
├── tests/
│
└── feature/
    └── advanced-rag/
```

说明：

---

## 根目录

运行中的：

```text
FastAPI API 项目
```

---

## feature/advanced-rag

实验目录：

包含：

- benchmark
- HotpotQA
- retrieval evaluation
- reranker experiments

详细见：

```text
feature/advanced-rag/README.md
```

---

# 系统架构

# Naive RAG 架构

初始版本：

```text
用户问题
    ↓
Query Embedding
    ↓
Chroma Vector Search
    ↓
Top-K Chunks
    ↓
Prompt Construction
    ↓
DeepSeek
    ↓
Answer
```

---

组件职责：

---

## Document Loader

负责：

读取本地文档。

支持：

- TXT
- MD
- PDF

输出统一结构：

```python
{
    "source": "...",
    "text": "...",
    "metadata": {...}
}
```

---

## Chunker

负责：

文本切分。

输出：

```python
{
    "chunk_id": "...",
    "source": "...",
    "page": ...,
    "start": ...,
    "end": ...,
    "text": "..."
}
```

---

## Embedding

负责：

文本：

```text
text
```

转：

```text
vector
```

---

## Vector Store

负责：

Chroma 封装。

支持：

- add_chunks()
- search()

---

## LLM Client

负责：

DeepSeek API 调用。

职责：

- request payload
- HTTP request
- response parsing
- error handling

---

# Advanced RAG 架构

当前生产级版本：

```text
用户问题
        ↓
BGE Query Embedding
        ↓
Chroma Retrieve Top100
        ↓
CrossEncoder Rerank Top25
        ↓
Context Construction
        ↓
DeepSeek Grounded Generation
        ↓
Answer + Sources
```

---

相比 Naive：

新增：

- stronger embedding
- sentence window retrieval
- reranker
- retrieval optimization

---

# 项目目录结构

```text
personal-kb-rag-api/
│
├── main.py                   # FastAPI API 入口
├── schemas.py               # Pydantic schemas
├── .env                     # 环境变量
├── requirements.txt
├── README.md
│
├── data/
│   ├── sample.txt
│   ├── sample.md
│   └── sample.pdf
│
├── services/
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── reranker.py
│   └── llm.py
│
├── tests/
│   ├── test_document_loader.py
│   └── test_chunker.py
│
└── feature/
    └── advanced-rag/
```

---

# 安装依赖

## pip 安装

```bash
pip install fastapi uvicorn requests python-dotenv
pip install sentence-transformers chromadb pymupdf
```

---

或：

```bash
pip install -r requirements.txt
```

---

# 环境变量配置

项目根目录创建：

```text
.env
```

内容：

```text
DEEPSEEK_API_KEY=你的真实APIKey
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
```

---

# 启动项目

进入项目目录：

```bash
cd personal-kb-rag-api
```

启动：

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

---

访问：

Swagger：

```text
http://127.0.0.1:8000/docs
```

ReDoc：

```text
http://127.0.0.1:8000/redoc
```

---

# API 接口

# GET /health

健康检查。

请求：

```http
GET /health
```

响应：

```json
{
  "status": "ok"
}
```

---

# POST /chat

普通 LLM 聊天接口。

不使用：

```text
RAG
```

---

请求：

```json
{
  "question": "什么是 FastAPI？",
  "model": "deepseek-v4-flash",
  "temperature": 0.7
}
```

---

响应：

```json
{
  "answer": "...",
  "model": "deepseek-v4-flash",
  "error": null
}
```

---

# POST /documents/upload

文档注册接口。

作用：

```text
把文档加入系统
```

不做：

```text
embedding
index
```

---

请求：

```json
{
  "file_paths": [
    "data/sample.pdf",
    "data/sample.md"
  ]
}
```

---

响应：

```json
{
  "registered_documents": [
    "sample.pdf",
    "sample.md"
  ],
  "failed_documents": [],
  "documents_count": 2,
  "error": null
}
```

---

职责：

- 文件读取
- 注册
- 批量支持
- 错误记录

---

# POST /documents/index

文档索引接口。

作用：

建立向量索引。

---

流程：

```text
document_id
→ load
→ chunk
→ embedding
→ add to Chroma
```

---

请求：

```json
{
  "document_ids": ["sample.pdf"],
  "chunk_size": 500
}
```

---

响应：

```json
{
  "registered_documents": [
    "sample.pdf"
  ],
  "failed_documents": [],
  "documents_count": 1,
  "error": null
}
```

---

# POST /rag/query

核心 RAG 查询接口。

当前已升级为：

```text
Advanced RAG
```

---

流程：

```text
query
→ embed_query
→ retrieve top100
→ rerank top25
→ context build
→ DeepSeek
→ answer
```

---

请求：

```json
{
  "question": "What is FastAPI?",
  "top_k": 25
}
```

---

响应：

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "...",
      "source": "...",
      "page": 0,
      "text_preview": "...",
      "distance": 0.23
    }
  ],
  "error": null
}
```

---

说明：

返回：

- 最终答案
- source chunks
- metadata

---

# 测试

使用：

```bash
pytest tests/
```

覆盖：

- document loader
- chunking

---

# 常见问题排查

## 文档未找到

检查：

```text
file_paths
```

---

## 文档未注册

先调用：

```text
/documents/upload
```

---

## 查询没结果

检查：

- 是否索引
- top_k 是否太小
- embedding 是否正确

---

## DeepSeek 超时

常见：

```text
Read timed out
```

原因：

- 网络
- API 波动
- timeout 太短

---

解决：

增大：

```python
timeout=30
```

---

## Retriever 效果差

检查：

- embedding model
- chunk strategy
- sentence window
- top-k
- reranker

---

# Advanced RAG 实验结果

完整实验记录见：

```text
feature/advanced-rag/README.md
```

本节给出核心结果概览。

---

# Benchmark 数据集

使用：

```text
HotpotQA
```

实验子集：

```text
200 supported questions
```

组成：

```text
bridge: 175
comparison: 25
difficulty: hard
```

---

# 实验方案

共测试：

---

## 1. Baseline BGE

流程：

```text
问题
→ BGE embedding
→ Chroma retrieval
→ DeepSeek
```

---

## 2. BGE + Sentence Window

流程：

```text
问题
→ BGE embedding
→ sentence-window retrieval
→ DeepSeek
```

---

## 3. BGE + Sentence Window + Reranker

流程：

```text
问题
→ retrieve top100
→ rerank top25
→ DeepSeek
```

---

# 实验结果

| 方法 | Answer Hit | Retrieval Hit |
|------|-----------:|--------------:|
| BGE Baseline | 86.5% | 82.5% |
| BGE + Sentence Window | 92.0% | 91.0% |
| BGE + Window + Reranker | 92.5% | 94.5% |

---

# 结果分析

## Sentence Window 是最大提升项

提升：

```text
Retrieval:
82.5% → 91.0%
```

原因：

单句：

```text
上下文太弱
```

例如：

```text
It was built in 1097.
```

Embedding 缺乏主体信息。

window 后：

```text
title + surrounding context
```

语义完整。

---

## Reranker 有效

最终：

```text
91.0% → 94.5%
```

原因：

向量检索：

```text
快
但粗
```

CrossEncoder：

```text
慢
但准
```

组合：

```text
高 recall + 高 precision
```

---

## Retriever 是核心瓶颈

很多错误本质：

```text
Retriever 没找到证据
```

结论：

```text
先优化 Retriever
再优化 Generator
```

---

# 错误分析

## 类型 1

现象：

```text
retrieval_hit = True
answer_hit = False
```

说明：

Retriever 正常。

问题：

```text
Generator / LLM
```

---

## 类型 2

现象：

```text
retrieval_hit = False
answer_hit = False
```

说明：

```text
Retriever recall 不足
```

---

## 类型 3

现象：

```text
retrieval_hit = False
answer_hit = True
```

说明：

模型：

- 用部分证据
- 用已有知识
- 严格指标限制

---

# 当前系统能力

当前项目已经从：

```text
Naive RAG MVP
```

升级为：

```text
Advanced RAG
```

具备：

---

## 文档处理

支持：

- TXT
- MD
- PDF

---

## Embedding

支持：

- Sentence Transformers
- BGE

---

## Retrieval

支持：

- Chroma vector retrieval
- sentence window retrieval
- top-k tuning

---

## Ranking

支持：

- CrossEncoder reranking

---

## Generation

支持：

- DeepSeek grounded generation

---

## API

支持：

- FastAPI production API

---

## Evaluation

支持：

- retrieval benchmark
- answer benchmark
- HotpotQA experiments

---

# 当前生产级 Pipeline

最终：

```text
用户问题
        ↓
BGE Query Embedding
        ↓
Chroma Retrieve Top100
        ↓
CrossEncoder Rerank Top25
        ↓
Context Construction
        ↓
DeepSeek Grounded Generation
        ↓
Answer + Sources
```

---

# 项目亮点

## 从零实现

不是：

```text
Notebook demo
```

而是：

```text
完整工程项目
```

---

## FastAPI 工程化

具备：

- schema validation
- API routing
- service abstraction
- modular design

---

## Retriever 优化

完成：

- embedding upgrade
- chunk strategy optimization
- reranker integration

---

## Benchmark 驱动优化

不是：

```text
感觉变好了
```

而是：

```text
benchmark 验证
```

---

## 实验可复现

feature/advanced-rag 提供：

- scripts
- datasets
- csv
- analysis

---

# 下一步 Roadmap

未来可以继续：

---

## Hybrid Retrieval

组合：

```text
BM25 + Vector Search
```

---

## Query Rewrite

先改写 query：

```text
更适合检索
```

---

## Multi-query Retrieval

一个 query：

生成多个：

```text
query1
query2
query3
```

---

## 更强 Reranker

当前：

```text
MiniLM
```

未来：

更强 CrossEncoder。

---

## Citation Grounding

答案更精确绑定：

```text
source chunks
```

---

## 更完整指标

加入：

```text
EM
F1
Recall@K
MRR
nDCG
```

---

# 最终总结

这是一个完整的：

```text
从零到生产级 RAG
```

实践项目。

成长路径：

```text
FastAPI 基础
    ↓
Naive RAG MVP
    ↓
Evaluation Framework
    ↓
Retriever Optimization
    ↓
Sentence Window
    ↓
Reranker
    ↓
Advanced RAG
```

最终提升：

---

Retrieval：

```text
82.5% → 94.5%
```

---

Answer：

```text
86.5% → 92.5%
```

---

一句话：

```text
从最小可运行 RAG，演进到可评估、可优化、可扩展的 Advanced RAG 系统。
```