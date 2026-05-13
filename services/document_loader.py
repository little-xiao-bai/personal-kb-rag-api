from pathlib import Path
import fitz # PyMuPDF 用于处理 PDF 文件

# =========================
# 加载 TXT / MD 文件
# =========================
def load_text_file(file_path: str) -> dict:
    """
    加载 TXT 或 MD 文本文件内容

    输入:
        文件路径

    输出:
        标准文档字典:
        {
            "source": 文件路径,
            "text": 文件内容,
            "metadata": {
                "file_type": 文件类型,
                "chars": 字符数
            }
        }
    """

    # 把字符串路径转换为 Path 对象
    path = Path(file_path)

    # 检查文件是否存在
    if not path.is_file():
        raise FileNotFoundError(f"文件未找到: {file_path}")
    

    # 读取整个文本文件 # UTF-8 是常用编码
    text = path.read_text(encoding="utf-8")
    
    # 返回统一结构的文档字典
    return {
        "source":str(path),
        "text": text,
        "metadata": {
            "file_type": path.suffix.lstrip('.').lower(), # 获取文件扩展名作为类型
            "chars": len(text) # 计算字符数
        }
    }

# # =========================
# # 加载 PDF 文件
# # =========================
# def load_pdf_file(file_path: str) -> dict:
#     """
#     加载 PDF 文件内容

#     输入:
#         PDF 文件路径

#     输出:
#         标准文档字典
#     """

#     # 转 Path 对象
#     path = Path(file_path)

#     # 文件存在性检查
#     if not path.is_file():
#         raise FileNotFoundError(f"文件未找到: {file_path}")
    
#     # 初始化文本内容
#     text = ""

#     # 打开 PDF
#     with fitz.open(file_path) as pdf:
#         # 遍历每一页
#         for page in pdf:
#             # 提取页面文字 # 每页后加换行
#             text += page.get_text() + "\n"

#     # 去掉首尾空白字符
#     text = text.strip()
    
#     # 返回统一结构
#     return {
#         "source":str(path),
#         "text": text,
#         "metadata": {
#             "file_type": "pdf",
#             "chars": len(text)
#         }
#     }

# =========================
# 加载 PDF 文件
# =========================
def load_pdf_file(file_path: str) -> dict:
    """
    加载 PDF 文件内容，并保留页码信息

    输入:
        PDF 文件路径

    输出:
        标准文档字典:
        {
            "source": 文件路径,
            "text": PDF 全文,
            "pages": [
                {
                    "page": 页码,
                    "text": 当前页文本
                }
            ],
            "metadata": {
                "file_type": "pdf",
                "chars": 字符数,
                "pages": 页数
            }
        }
    """

    # 转 Path 对象
    path = Path(file_path)

    # 文件存在性检查
    if not path.is_file():
        raise FileNotFoundError(f"文件未找到: {file_path}")

    # 保存 PDF 全文
    text = ""

    # 保存每一页的文本和页码
    pages = []

    # 打开 PDF
    with fitz.open(file_path) as pdf:
        # 遍历每一页
        # enumerate(..., start=1) 表示页码从 1 开始
        for page_number, page in enumerate(pdf, start=1):
            # 提取当前页文本
            page_text = page.get_text().strip()

            # 保存当前页
            pages.append({
                "page": page_number,
                "text": page_text
            })

            # 拼接到全文
            text += page_text + "\n\n"

    # 去掉首尾空白字符
    text = text.strip()

    # 返回统一结构
    return {
        "source": str(path),
        "text": text,
        "pages": pages,
        "metadata": {
            "file_type": "pdf",
            "chars": len(text),
            "pages": len(pages)
        }
    }

# =========================
# 通用文档加载入口
# =========================
def load_document(file_path: str) -> dict:
    """
    根据文件类型自动选择加载器

    支持:
        txt
        md
        pdf

    输入:
        文件路径

    输出:
        标准文档字典
    """
    # 获取文件扩展名 # example.txt -> .txt
    ext = Path(file_path).suffix.lower()

    # TXT / Markdown
    if ext in ['.txt', '.md']:
        return load_text_file(file_path)
    
    # PDF
    elif ext == '.pdf':
        return load_pdf_file(file_path)
    
    # 不支持的文件
    else:
        raise ValueError(f"不支持的文件类型: {ext}")