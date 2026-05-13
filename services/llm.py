from pydantic import BaseModel, Field
from typing import List
import requests

# =========================
# LLM 消息模型
# =========================
# 对应 OpenAI / DeepSeek Chat API 中的 message 格式：
# {
#   "role": "user",
#   "content": "你好"
# }
class LLMMessage(BaseModel):
    # 消息角色
    # system = 系统提示词
    # user = 用户输入
    # assistant = AI 回复

    role: str = Field(description="消息角色，如 system、user、assistant")
    content: str = Field(description="消息内容")

# =========================
# DeepSeek Chat API 请求体模型
# =========================
# 最终会变成：
#
# {
#   "model": "deepseek-v4-flash",
#   "messages": [...],
#   "temperature": 0.7,
#   "stream": false
# }
class DeepSeekChatPayload(BaseModel):
    # 使用哪个模型
    model: str = Field(default="deepseek-v4-flash", description="使用的模型名称")

    # 对话消息列表
    messages: List[LLMMessage] = Field(description="消息列表")

    # 温度（控制生成随机性）# 越高越发散 # 越低越稳定
    temperature: float = Field(default = 0.7, description="生成文本的随机程度")

    # 是否流式输出 # False = 一次性返回完整结果 # True = 边生成边返回结果
    stream: bool = Field(default=False, description="是否启用流式响应")

# =========================
# LLM 客户端
# =========================
# 作用：
# 封装 DeepSeek API 调用逻辑
#
# 外部只需要：
#
# client = LLMClient(...)
# client.chat("你好")
#
# 不需要关心 HTTP 请求细节
class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None ,model: str = 'deepseek-v4-flash',timeout: int = 10, simulate: bool = False):
        """
        初始化 LLM 客户端

        api_key:
            DeepSeek API Key

        base_url:
            API 地址，例如：
            https://api.deepseek.com/chat/completions

        model:
            使用的模型名称

        timeout:
            请求超时时间（秒）

        simulate:
            是否启用模拟模式
            True 时不会真的调用 API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.simulate = simulate

    # =========================
    # 对外公开方法
    # =========================
    def chat(self, question: str, temperature: float = 0.7) -> str:
        """
        对外聊天接口

        输入：
            用户问题

        输出：
            AI 回复文本
        """

        # 模拟模式
        # 用于测试，不消耗 API
        if self.simulate:
            return f"[模拟响应] 你问了: {question}，温度设置为: {temperature}"
        
        # 真正调用 API
        return self._call(question, temperature)
    
    # =========================
    # 内部 API 调用方法
    # =========================
    def _call(self, question: str, temperature: float = 0.7) -> str:
        """
        内部方法：
        构造请求 -> 发送 HTTP 请求 -> 解析响应
        """

        # 检查配置
        if not self.api_key or not self.base_url:
            raise ValueError("API Key 和 Base URL 必须提供")
        
        # 构造请求体
        payload = DeepSeekChatPayload(
            model=self.model,
            messages=[
                LLMMessage(role="system", content="You are a helpful assistant."),
                LLMMessage(role="user", content=question)
            ],
            temperature=temperature
        )

        # HTTP 请求头
        headers = {
            "Authorization":f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            # 发起 POST 请求
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload.model_dump(),
                timeout=self.timeout
            )

            # 调试日志
            print(f"请求 URL: {self.base_url}")
            print(f"请求 Headers: {headers}")
            print(f"HTTP 状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            # 如果状态码不是 200，会抛异常
            response.raise_for_status()

            # JSON 转 Python dict
            data = response.json()

            # 提取 AI 回复
            return data['choices'][0]['message']['content']
        
        # 返回结构异常
        except (KeyError, IndexError, TypeError):
            return "[错误] 未找到 content 字段"
        
        # 网络错误 / 超时 / 连接失败
        except requests.exceptions.RequestException as e:
            return f"[请求错误] {str(e)}"
        
        # JSON 解析失败
        except ValueError:
            return "[错误] 无法解析响应 JSON"
