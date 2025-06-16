"""OpenAI 模型实现。"""

from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from loguru import logger
import openai

from novamind.core.models import BaseModel as BaseLLM, ModelResponse


class OpenAIConfig(BaseModel):
    """OpenAI 配置。"""
    
    model_name: str = Field(default="gpt-3.5-turbo")
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization: Optional[str] = None
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = None
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)


class OpenAIModel(BaseLLM):
    """OpenAI 模型实现。"""
    
    def __init__(
        self,
        config: Optional[OpenAIConfig] = None,
        **kwargs: Any,
    ):
        """初始化 OpenAI 模型。
        
        参数:
            config: OpenAI 配置
            **kwargs: 附加参数
        """
        self.config = config or OpenAIConfig(**kwargs)
        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            organization=self.config.organization,
        )
        logger.info(f"初始化 OpenAI 模型: {self.config.model_name}")
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """生成响应。
        
        参数:
            messages: 消息列表
            **kwargs: 附加参数
            
        返回:
            模型响应
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                **kwargs,
            )
            
            return ModelResponse(
                content=response.choices[0].message.content,
                metadata={
                    "model": self.config.model_name,
                    "usage": response.usage.model_dump(),
                    "finish_reason": response.choices[0].finish_reason,
                },
            )
        except Exception as e:
            logger.error(f"生成响应时出错: {e}")
            raise
            
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[ModelResponse, None]:
        """流式生成响应。
        
        参数:
            messages: 消息列表
            **kwargs: 附加参数
            
        返回:
            模型响应生成器
        """
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stream=True,
                **kwargs,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield ModelResponse(
                        content=chunk.choices[0].delta.content,
                        metadata={
                            "model": self.config.model_name,
                            "finish_reason": chunk.choices[0].finish_reason,
                        },
                    )
        except Exception as e:
            logger.error(f"流式生成响应时出错: {e}")
            raise
            
    def __del__(self) -> None:
        """清理资源。"""
        if hasattr(self, "client"):
            self.client.close() 