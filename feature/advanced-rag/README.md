# Advanced RAG 实验记录

> 基于 HotpotQA 的检索增强生成（RAG）优化实验，用于将原始 Naive RAG 系统升级为更强的 Advanced RAG。

---

# 项目说明

本目录记录了将原始 **Naive RAG** 系统逐步升级为 **Advanced RAG** 的完整实验过程。

本实验的核心目标是：

```text
提升 RAG 在复杂多跳问答场景下的检索与回答能力
```

重点优化方向：

- 更强的 Embedding 模型
- 更合理的 Chunk 策略
- Sentence Window Retrieval
- Retrieval-only Benchmark
- Reranker 精排
- HotpotQA 实验评估
- FastAPI 生产级集成

---

# 背景

最初项目实现的是一个最小可运行的 Naive RAG 系统。

整体流程：

```text
用户问题
    ↓
Query Embedding
    ↓
Chroma 向量检索
    ↓
Top-K Chunks
    ↓
DeepSeek
    ↓
最终回答
```

这个版本已经具备完整闭环：

- 文档上传
- 文档切分
- Embedding
- Chroma 检索
- Prompt 构造
- LLM 回答
- Sources 返回

即：

```text
MVP Naive RAG
```

对于简单知识库问答已经可用。

---

## 为什么继续升级？

在真实复杂问答中，尤其是：

```text
多跳推理（multi-hop QA）
```

Naive RAG 暴露明显问题。

典型失败：

### 1. 检索不到关键证据

Retriever 无法召回真正需要的 supporting facts。

表现：

```text
LLM 说不知道
```

本质：

```text
Retriever recall 不足
```

---

### 2. 单句上下文太弱

例如：

```text
It was built in 1097.
```

问题：

```text
谁 built？
什么 built？
```

句子本身缺乏主体。

Embedding 表示不稳定。

---

### 3. Bridge Question 表现差

例如：

```text
X 和 Y 哪个更早？
```

需要：

```text
检索 X 的信息
+
检索 Y 的信息
+
比较
```

这类问题对 Retriever 要求非常高。

---

### 4. Top-K 噪声污染 Prompt

例如：

```text
Top20
```

里只有：

```text
2~3 个相关 chunk
```

剩下：

```text
17 个噪声
```

导致：

- Prompt 被污染
- token 浪费
- 模型注意力分散

---

因此：

需要系统升级。

---

# 实验目标

本实验的核心目标：

---

## 1. 提升 Retriever Recall

核心问题：

```text
真正需要的证据能否被召回？
```

目标：

```text
提高 Retrieval Hit
```

---

## 2. 提升最终 Answer Accuracy

核心问题：

```text
最终回答是否正确？
```

目标：

```text
提高 Answer Hit
```

---

## 3. 分离 Retriever 与 Generator

传统 RAG：

```text
Retriever + LLM 混在一起
```

问题：

失败时不知道是谁的问题。

所以：

拆成两类评估：

---

### Retrieval-only

只看：

```text
Retriever
```

---

### End-to-end

看：

```text
Retriever + Generator
```

---

这样可以明确定位问题来源。

---

## 4. 构建更强生产级 RAG Pipeline

最终目标：

将：

```text
Naive RAG
```

升级为：

```text
Advanced RAG
```

最终接回：

```text
FastAPI API
```

---

# 数据集

# HotpotQA

本实验使用：

```text
HotpotQA
```

论文：

> Yang et al., HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering (EMNLP 2018)

官网：

https://hotpotqa.github.io/

下载地址：

http://curtis.ml.cmu.edu/datasets/hotpot/

---

## 为什么选择 HotpotQA？

HotpotQA 非常适合 RAG 实验。

---

### 多跳推理

不是：

```text
法国首都是哪里？
```

而是：

```text
X 和 Y 哪个更早建立？
```

需要：

```text
事实 A
+
事实 B
+
跨文档推理
```

---

### Supporting Facts

HotpotQA 提供：

```text
gold evidence
```

这非常重要。

因为可以直接判断：

```text
Retriever 有没有找对证据
```

---

### Bridge Questions

例如：

```text
通过 A 找到 B
再回答问题
```

非常贴近真实 RAG 场景。

---

# 当前实验子集

完整 HotpotQA：

```text
7000+
```

完整跑太慢。

所以构造小 benchmark：

```text
200 条 supported questions
```

组成：

```text
bridge: 175
comparison: 25
```

难度：

```text
hard
```

---

## 为什么只用 200 条？

原因：

- 快速实验
- API 成本低
- 易于调试
- 易于重复 benchmark
- 更适合 Retriever 调参

---

# 实验整体流程

完整流程：

```text
HotpotQA 原始数据
        ↓
下载
        ↓
筛选 supported questions
        ↓
构造 benchmark
        ↓
生成 sentence corpus
        ↓
生成 sentence-window corpus
        ↓
Embedding 入 Chroma
        ↓
Baseline RAG 实验
        ↓
Retrieval-only Benchmark
        ↓
Reranker 实验
        ↓
结果分析
        ↓
FastAPI 集成
```

---

# 目录结构

```text
feature/advanced-rag/
│
├── README.md
│
├── data/
│   ├── hotpotqa_dev_fullwiki.json
│   ├── hotpotqa_dev_fullwiki_pretty.json
│   ├── test_questions_supported_200.json
│   │
│   ├── hotpotqa_supported_sentences_200/
│   └── hotpotqa_supported_windows_200/
│
├── data_retrieval/
│   └── retrieval_only_bge_window_top20.csv
│
├── scripts/
│   ├── 1.download_hotpotqa.py
│   ├── 2.filter_hotpotqa_supported.py
│   ├── 2.generate_supported_window_corpus.py
│   ├── 3.embed_supported_sentences.py
│   ├── 4.run_baseline_supported.py
│   ├── 5.analyze_baseline.py
│   ├── 6.eval_retrieval_only.py
│   ├── count_hotpotqa_questions.py
│   └── get_chunk.py
│
├── baseline_supported_200.csv
├── baseline_supported_200_bge_top20.csv
├── baseline_supported_200_bge_window.csv
└── baseline_supported_200_bge_window_rerank.csv
```

---

# 脚本说明

# 1.download_hotpotqa.py

作用：

下载 HotpotQA 原始数据集。

输出：

```text
hotpotqa_dev_fullwiki.json
```

说明：

这是原始官方数据。

包含：

- question
- answer
- context
- supporting_facts
- type
- level

---

# 2.filter_hotpotqa_supported.py

作用：

从 HotpotQA 中筛选：

```text
supported questions
```

构造实验 benchmark。

输出：

```text
test_questions_supported_200.json
```

每条数据包含：

- question
- expected_answer
- gold_context
- type
- level

---

## 为什么做这个？

原始 HotpotQA：

```text
太大
太杂
不适合快速实验
```

所以先筛：

```text
可控 benchmark
```

方便：

- 快速重复实验
- 对比不同 Retriever
- 降低 API 成本

---

# 2.generate_supported_window_corpus.py

作用：

生成：

```text
sentence window corpus
```

窗口策略：

```text
前一句
+
当前句
+
后一句
```

即：

```text
window size = 3
```

---

例如：

原始句：

```text
It was built in 1097.
```

问题：

上下文太弱。

---

window 后：

```text
Djamaâ el Kebir

The Great Mosque...
An inscription says...
It was built in 1097.
```

---

## 为什么有效？

Embedding 不仅看当前句。

还能看到：

- 主体
- 局部语义
- 相邻上下文

提高语义表达质量。

---

# 3.embed_supported_sentences.py

作用：

把语料 embedding 写入 Chroma。

Embedding 模型：

```text
BAAI/bge-small-en-v1.5
```

---

职责：

### 1. 加载语料

读取：

```text
hotpotqa_supported_sentences_200
```

或：

```text
hotpotqa_supported_windows_200
```

---

### 2. Embedding

把文本：

```text
text
```

转：

```text
vector
```

---

### 3. 写入 Chroma

保存：

- ids
- documents
- metadata
- embeddings

---

## 为什么单独做 embedding？

避免：

每次 benchmark：

```text
重复 embedding
```

节省时间。

---

# 4.run_baseline_supported.py

作用：

完整 RAG benchmark。

流程：

```text
问题
→ query embedding
→ retrieval
→ optional rerank
→ DeepSeek
→ answer
→ CSV
```

---

输出：

例如：

```text
baseline_supported_200.csv
baseline_supported_200_bge_top20.csv
baseline_supported_200_bge_window.csv
baseline_supported_200_bge_window_rerank.csv
```

---

CSV 记录：

- question
- expected_answer
- rag_answer
- gold_context
- retrieved_context
- answer_hit
- retrieval_hit

---

## 为什么保留 CSV？

方便：

后续分析：

```text
错误题
类型统计
对比实验
```

---

# 5.analyze_baseline.py

作用：

分析 benchmark 结果。

统计：

- Answer Hit
- Retrieval Hit

---

支持分组：

### 按 type

例如：

```text
bridge
comparison
```

---

### 按 level

例如：

```text
hard
```

---

用途：

快速比较不同实验。

例如：

```text
BGE vs Window vs Reranker
```

---

# 6.eval_retrieval_only.py

作用：

只评估：

```text
Retriever
```

不调用：

```text
DeepSeek
```

---

流程：

```text
问题
→ query embedding
→ retrieval
→ compare with gold evidence
```

---

为什么重要？

传统 RAG：

失败时：

不知道：

```text
Retriever 问题
还是 LLM 问题
```

Retrieval-only 可以明确：

```text
只看 Retriever
```

---

优点：

- 快
- 稳定
- 无 API 波动
- 零生成成本
- 适合调 top-k
- 适合调 reranker

---

# count_hotpotqa_questions.py

工具脚本。

作用：

统计：

```text
HotpotQA 问题数量
```

例如：

- bridge
- comparison
- hard

---

# get_chunk.py

调试工具。

作用：

查看：

```text
Chroma chunk 内容
```

例如：

- chunk_id
- source
- metadata
- text

---

用途：

调试：

```text
embedding 是否正确
chunk 内容是否合理
window 是否真的生成
```

---

# 实验设计

# 实验 1：Baseline BGE

目标：

建立 baseline。

流程：

```text
问题
→ BGE embedding
→ Chroma top-k
→ DeepSeek
```

---

配置：

```text
Embedding:
BAAI/bge-small-en-v1.5
```

---

结果：

```text
Answer Hit: 86.5%
Retrieval Hit: 82.5%
```

---

发现：

bridge 问题表现较差。

说明：

```text
Retriever recall 不足
```

---

# 实验 2：Sentence Window

目标：

解决：

```text
单句上下文太弱
```

---

流程：

```text
问题
→ BGE embedding
→ sentence-window retrieval
→ DeepSeek
```

---

思路：

不要 embedding：

```text
单句
```

而是：

```text
局部上下文
```

---

结果：

```text
Answer Hit: 92.0%
Retrieval Hit: 91.0%
```

---

提升：

```text
Retrieval +8.5%
```

---

结论：

Sentence Window 非常有效。

---

# 实验 3：Retrieval-only Benchmark

目标：

单独看 Retriever。

---

流程：

```text
问题
→ embedding
→ retrieval
→ compare with gold evidence
```

---

不调用：

```text
LLM
```

---

意义：

判断：

```text
Retriever 到底行不行
```

---

适合调：

- top-k
- embedding
- reranker
- chunk strategy

---

# 实验 4：Reranker

目标：

减少：

```text
Top-K 噪声
```

---

思路：

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

所以：

组合。

---

流程：

```text
问题
→ retrieve top100
→ rerank
→ top25
→ DeepSeek
```

---

Reranker：

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

为什么不是：

```text
retrieve 20 → rerank 5
```

因为：

如果：

```text
gold evidence 根本没进 top20
```

rerank 没意义。

所以：

先大召回。

再精排。

---

# 实验结果

# 实验结果对比

| 方法 | Answer Hit | Retrieval Hit |
|------|-----------:|--------------:|
| BGE Baseline | 86.5% | 82.5% |
| BGE + Sentence Window | 92.0% | 91.0% |
| BGE + Sentence Window + Reranker | 92.5% | 94.5% |

---

# 实验结果分析

## Baseline BGE

结果：

```text
Answer Hit: 86.5%
Retrieval Hit: 82.5%
```

说明：

基础 BGE embedding 已经有不错效果。

但问题：

```text
Retriever recall 不够高
```

尤其：

```text
bridge question
```

表现较差。

原因：

bridge 类型需要：

```text
多个 supporting facts 同时召回
```

难度更高。

---

## Sentence Window

结果：

```text
Answer Hit: 92.0%
Retrieval Hit: 91.0%
```

相比 baseline：

```text
Retrieval:
82.5% → 91.0%

Answer:
86.5% → 92.0%
```

---

提升原因：

单句：

```text
语义信息不足
```

例如：

```text
It was built in 1097.
```

Embedding 难理解。

Window 后：

```text
title + surrounding context
```

语义完整。

因此：

Retriever 更容易召回正确 chunk。

---

## Reranker

最终配置：

```text
retrieve_top_k = 100
rerank_top_k = 25
```

结果：

```text
Answer Hit: 92.5%
Retrieval Hit: 94.5%
```

相比 Window：

```text
Retrieval:
91.0% → 94.5%
```

---

提升原因：

向量检索：

```text
粗召回
```

容易混入噪声。

Reranker：

```text
精排
```

更擅长：

```text
query-chunk relevance 判断
```

因此：

正确 chunk 被排到更前面。

---

# 指标说明

# Answer Hit

定义：

最终答案是否正确。

用于：

```text
End-to-end RAG 效果
```

即：

```text
Retriever + Generator 综合表现
```

---

例如：

问题：

```text
Who was older?
```

expected：

```text
Charles Nungesser
```

模型回答：

```text
Charles Eugène Nungesser
```

本质：

```text
对
```

---

# Retrieval Hit

定义：

所有 gold evidence 是否都被召回。

严格判断。

要求：

```text
所有 supporting facts 都必须出现
```

否则：

```text
False
```

---

用于：

```text
Retriever recall
```

---

注意：

这是严格指标。

所以：

```text
Retriever 不全
但模型回答对
```

也可能出现。

---

# 错误分析

错误分析的核心思想：

先判断：

```text
Retriever 有没有找到证据
```

再判断：

```text
LLM 有没有正确回答
```

---

# 类型 1：Retrieve 全，但回答错

现象：

```text
retrieval_hit = True
answer_hit = False
```

说明：

Retriever：

```text
正常
```

问题：

```text
LLM / Generator
```

---

实际观察：

把：

```text
retrieved_context
```

直接发给 GPT。

GPT 可以回答。

说明：

```text
证据足够
DeepSeek 没答出来
```

---

原因：

可能：

- Prompt 不够好
- DeepSeek 推理不足
- Generator 不稳定

---

结论：

这是：

```text
Generator 问题
```

不是 Retriever。

---

# 类型 2：Retrieve 不全，回答错

现象：

```text
retrieval_hit = False
answer_hit = False
```

说明：

Retriever：

```text
漏召回
```

---

本质：

```text
没有关键 supporting facts
```

所以：

LLM 无法回答。

---

结论：

这是：

```text
Retriever recall 问题
```

---

# 类型 3：Retrieve 不全，但回答对

现象：

```text
retrieval_hit = False
answer_hit = True
```

说明：

虽然严格指标判：

```text
Retriever 不全
```

但：

模型依然回答正确。

---

可能原因：

---

## 1. 部分证据足够

并不一定：

```text
所有 gold evidence
```

都必须。

有时：

```text
部分 supporting facts
```

已经够回答。

---

## 2. 模型利用常识

例如：

模型已有背景知识。

即使 context 不完整：

也能猜对。

---

## 3. retrieval_hit 太严格

当前判断：

```text
必须全部 gold evidence 命中
```

这本身很严格。

---

结论：

```text
retrieval_hit != answer correctness
```

---

# 核心发现

# 1. Sentence Window 是最大提升项

提升：

```text
82.5% → 91.0%
```

说明：

局部上下文极其重要。

---

原因：

单句：

```text
信息不足
```

window：

```text
语义完整
```

---

这是当前最大收益优化。

---

# 2. Retriever Recall 是核心

很多失败：

本质：

```text
Retriever 没找到
```

如果 evidence 缺失：

LLM 再强也没用。

---

结论：

```text
先优化 Retriever
再优化 Generator
```

---

# 3. Reranker 有效，但有前提

错误配置：

```text
retrieve 20
rerank 5
```

问题：

如果：

```text
gold evidence 不在 top20
```

rerank 无法救。

---

正确配置：

```text
retrieve 100
rerank 25
```

原则：

```text
先高 recall
再高 precision
```

---

# 4. Retrieval-only Benchmark 非常有价值

优点：

- 快
- 稳
- 成本低
- 无 API 波动

非常适合：

```text
Retriever 调参
```

---

比：

```text
每次都调用 LLM
```

高效得多。

---

# 最终生产级 Pipeline

最终接回 FastAPI 的 Advanced RAG：

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

# FastAPI 集成

最终 Advanced RAG 已接回：

```text
main.py
```

核心接口：

```text
POST /rag/query
```

流程：

```text
query
→ embed_query()
→ vector_store.search(top100)
→ reranker.rerank(top25)
→ context build
→ DeepSeek generate
→ response
```

说明：

实验成果已经从：

```text
offline benchmark
```

升级为：

```text
production API pipeline
```

---

# 下一步优化方向

未来可继续探索：

---

## Hybrid Retrieval

组合：

```text
BM25 + Vector Search
```

解决：

```text
纯语义检索漏 lexical match
```

---

## Query Rewrite

先让模型改写 query：

例如：

```text
复杂问题 → 更适合检索的问题
```

---

## Multi-query Retrieval

一个问题：

生成多个 query：

```text
query1
query2
query3
```

提升 recall。

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

更精确：

```text
答案对应 source chunk
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

本实验成功将：

```text
Naive RAG
```

升级为：

```text
Advanced RAG
```

最终结果：

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

核心贡献：

- 更强 embedding
- sentence window retrieval
- reranker
- retrieval-only benchmark
- systematic error analysis
- FastAPI production integration

---

最终形成：

```text
可复用的 Advanced RAG Pipeline
```