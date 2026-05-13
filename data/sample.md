# 测试文档 - 深度示例

这是一个用于测试 `load_text_file` 函数的 Markdown 文档。内容稍微丰富一些，包含多段文字和列表，以验证读取文本长度、前 100 个字符统计，以及 metadata 信息。

---

## 1. 文档简介

本文件演示如何在项目中使用 Markdown 文档作为 RAG 数据源。  
我们希望 LLM 能够读取文件内容，并生成回答或摘要。

---

## 2. 使用说明

1. 将本文件放在项目 `data/` 目录下  
2. 使用 `load_text_file("data/sample.md")` 读取内容  
3. 函数返回字典，包含：
   - `source`：文件路径
   - `text`：文件完整文本
   - `metadata`：字典，包含文件类型和字符长度

---

## 3. 示例文本

Lorem ipsum dolor sit amet, consectetur adipiscing elit.  
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.  
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.  

---

## 4. 列表示例

- 第一个项目：验证 Markdown 列表解析  
- 第二个项目：测试多行文本处理  
- 第三个项目：确保文本长度统计准确  

---

## 5. 结语

这个文档仅用于测试 RAG 文档加载功能。  
文本长度适中，足够测试前 100 个字符、总长度统计以及文件类型识别。