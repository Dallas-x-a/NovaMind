"""
NovaMind模型接口定义

提供统一的模型接口、配置管理和注册系统。
支持各种大语言模型的统一调用和管理。

主要功能：
- 统一模型接口：标准化的模型调用接口
- 配置管理：灵活的模型参数配置
- 模型注册：动态模型注册和管理
- 流式生成：支持流式文本生成
- 消息格式化：统一的消息格式处理
- 多模型支持：支持多种大语言模型
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    """
    模型响应结果
    
    标准化的模型输出格式，包含响应内容、元数据和使用统计
    """
    
    content: str                                    # 响应内容 - 模型生成的文本
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 元数据 - 额外的响应信息
    usage: Dict[str, int] = Field(default_factory=dict)     # 使用统计 - token使用情况
    finish_reason: Optional[str] = None             # 完成原因 - 生成结束的原因


class ModelMessage(BaseModel):
    """
    模型输入/输出消息
    
    定义与模型交互的消息格式，支持角色、内容、名称等
    """
    
    role: str                                       # 角色 - user/assistant/system
    content: str                                    # 消息内容 - 实际的文本内容
    name: Optional[str] = None                      # 消息名称 - 可选的标识符
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 元数据 - 额外的消息信息


class ModelConfig(BaseModel):
    """
    模型配置
    
    定义模型的各种参数和配置选项，支持不同模型的特定参数
    """
    
    model_name: str                                 # 模型名称 - 唯一标识符
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)  # 温度参数 - 控制随机性
    max_tokens: Optional[int] = Field(default=None, gt=0)    # 最大token数 - 输出长度限制
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)        # top_p参数 - 核采样参数
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)  # 频率惩罚 - 减少重复
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)   # 存在惩罚 - 鼓励新话题
    stop: Optional[List[str]] = None                # 停止词 - 生成停止条件
    api_key: Optional[str] = None                   # API密钥 - 认证密钥
    api_base: Optional[str] = None                  # API基础URL - 服务地址
    custom: Dict[str, Any] = Field(default_factory=dict)     # 自定义参数 - 模型特定参数


class BaseModel(ABC):
    """
    所有模型的基类
    
    定义模型的标准接口，包括初始化、生成、配置等功能
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ):
        """
        初始化模型
        
        Args:
            config: 模型配置
            **kwargs: 附加参数
        """
        self.config = config or ModelConfig(model_name="default")
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            
    @property
    def model_name(self) -> str:
        """
        获取模型名称
        
        Returns:
            str: 模型名称
        """
        return self.config.model_name
        
    @abstractmethod
    async def generate(
        self,
        messages: List[ModelMessage],
        **kwargs: Any,
    ) -> ModelResponse:
        """
        生成响应 - 抽象方法，子类必须实现
        
        Args:
            messages: 输入消息列表
            **kwargs: 附加参数
            
        Returns:
            ModelResponse: 模型响应
        """
        pass
        
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[ModelMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[ModelResponse, None]:
        """
        生成流式响应 - 抽象方法，子类必须实现
        
        Args:
            messages: 输入消息列表
            **kwargs: 附加参数
            
        Yields:
            ModelResponse: 模型响应块
        """
        pass
        
    def format_messages(
        self,
        messages: List[Union[str, Dict[str, Any], ModelMessage]],
    ) -> List[ModelMessage]:
        """
        格式化消息为模型输入格式
        
        将各种格式的消息统一转换为ModelMessage格式
        
        Args:
            messages: 各种格式的消息列表
            
        Returns:
            List[ModelMessage]: 格式化的ModelMessage对象列表
            
        Raises:
            TypeError: 当消息类型不支持时
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, str):
                formatted.append(ModelMessage(role="user", content=msg))
            elif isinstance(msg, dict):
                formatted.append(ModelMessage(**msg))
            elif isinstance(msg, ModelMessage):
                formatted.append(msg)
            else:
                raise TypeError(f"不支持的消息类型: {type(msg)}")
        return formatted
        
    def update_config(self, **kwargs: Any) -> None:
        """
        更新模型配置
        
        Args:
            **kwargs: 配置更新
        """
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            
    def to_dict(self) -> Dict[str, Any]:
        """
        将模型转换为字典
        
        Returns:
            Dict[str, Any]: 模型的字典表示
        """
        return {
            "model_name": self.model_name,
            "config": self.config.model_dump(),
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """
        从字典创建模型
        
        Args:
            data: 模型数据字典
            
        Returns:
            BaseModel: 模型实例
        """
        config = ModelConfig(**data["config"])
        return cls(config=config)


class ModelRegistry:
    """
    模型注册表 - 管理所有注册的模型
    
    提供模型类的注册、注销和查找功能，
    支持动态模型管理
    """
    
    def __init__(self):
        """
        初始化注册表
        """
        self._models: Dict[str, type[BaseModel]] = {}  # 模型类字典，以模型名称为键
        
    def register(self, name: str, model_class: type[BaseModel]) -> None:
        """
        注册模型类
        
        Args:
            name: 模型名称
            model_class: 要注册的模型类
            
        Raises:
            ValueError: 当模型名称已存在时
        """
        if name in self._models:
            raise ValueError(f"模型 {name} 已注册")
        self._models[name] = model_class
        
    def unregister(self, name: str) -> None:
        """
        注销模型类
        
        Args:
            name: 要注销的模型名称
            
        Raises:
            ValueError: 当模型不存在时
        """
        if name not in self._models:
            raise ValueError(f"模型 {name} 未找到")
        del self._models[name]
        
    def get_model_class(self, name: str) -> type[BaseModel]:
        """
        按名称获取模型类
        
        Args:
            name: 要获取的模型名称
            
        Returns:
            type[BaseModel]: 请求的模型类
            
        Raises:
            ValueError: 当模型不存在时
        """
        if name not in self._models:
            raise ValueError(f"模型 {name} 未找到")
        return self._models[name]
        
    def create_model(
        self,
        name: str,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        创建模型实例
        
        Args:
            name: 要创建的模型名称
            config: 模型配置
            **kwargs: 附加参数
            
        Returns:
            BaseModel: 模型实例
        """
        model_class = self.get_model_class(name)
        return model_class(config=config, **kwargs)
        
    def list_models(self) -> List[str]:
        """
        列出所有注册的模型
        
        Returns:
            List[str]: 模型名称列表
        """
        return list(self._models.keys())


class ModelManager:
    """
    模型管理器 - 统一管理模型实例
    
    提供模型实例的创建、缓存和管理功能，
    支持模型的生命周期管理
    """
    
    def __init__(self):
        """
        初始化模型管理器
        """
        self.registry = ModelRegistry()  # 模型注册表
        self._instances: Dict[str, BaseModel] = {}  # 模型实例缓存
        self._initialize_default_models()  # 初始化默认模型
        
    def _initialize_default_models(self):
        """
        初始化默认模型
        
        注册系统预定义的模型类型
        """
        # 这里可以注册默认的模型类型
        pass
        
    def register_model(self, name: str, model_class: type[BaseModel]) -> None:
        """
        注册模型类
        
        Args:
            name: 模型名称
            model_class: 要注册的模型类
        """
        self.registry.register(name, model_class)
        
    def get_model(self, config: ModelConfig) -> BaseModel:
        """
        获取模型实例
        
        根据配置获取或创建模型实例，支持实例缓存
        
        Args:
            config: 模型配置
            
        Returns:
            BaseModel: 模型实例
        """
        # 检查缓存
        cache_key = f"{config.model_name}_{hash(str(config.model_dump()))}"
        if cache_key in self._instances:
            return self._instances[cache_key]
            
        # 创建新实例
        try:
            model = self.registry.create_model(config.model_name, config)
            self._instances[cache_key] = model
            return model
        except ValueError as e:
            # 如果模型未注册，使用默认模型
            from .models import MockModel
            model = MockModel(config)
            self._instances[cache_key] = model
            return model
        
    def create_model(self, name: str, **kwargs: Any) -> BaseModel:
        """
        创建模型实例
        
        Args:
            name: 模型名称
            **kwargs: 模型参数
            
        Returns:
            BaseModel: 模型实例
        """
        config = ModelConfig(model_name=name, **kwargs)
        return self.get_model(config)
        
    def list_available_models(self) -> List[str]:
        """
        列出可用的模型
        
        Returns:
            List[str]: 可用模型名称列表
        """
        return self.registry.list_models()
        
    def remove_model(self, name: str) -> None:
        """
        移除模型
        
        从注册表中移除模型类，并清理相关实例
        
        Args:
            name: 要移除的模型名称
        """
        # 清理相关实例
        instances_to_remove = [
            key for key in self._instances.keys()
            if key.startswith(f"{name}_")
        ]
        for key in instances_to_remove:
            del self._instances[key]
            
        # 从注册表移除
        self.registry.unregister(name)
        
    def clear_models(self) -> None:
        """
        清空所有模型
        
        清理所有模型实例和注册信息
        """
        self._instances.clear()
        # 注意：这里不清理注册表，因为可能还需要重新创建实例


class MockModel(BaseModel):
    """
    模拟模型 - 用于测试和开发
    
    提供基本的模型功能，用于在没有真实模型时的测试
    """
    
    async def generate(
        self,
        messages: List[ModelMessage],
        **kwargs: Any,
    ) -> ModelResponse:
        """
        生成模拟响应
        
        Args:
            messages: 输入消息列表
            **kwargs: 附加参数
            
        Returns:
            ModelResponse: 模拟响应
        """
        # 简单的模拟响应
        last_message = messages[-1].content if messages else ""
        response_content = f"这是对 '{last_message}' 的模拟响应。"
        
        return ModelResponse(
            content=response_content,
            metadata={"model": "mock", "temperature": self.config.temperature},
            usage={"prompt_tokens": len(str(messages)), "completion_tokens": len(response_content)}
        )
        
    async def generate_stream(
        self,
        messages: List[ModelMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[ModelResponse, None]:
        """
        生成模拟流式响应
        
        Args:
            messages: 输入消息列表
            **kwargs: 附加参数
            
        Yields:
            ModelResponse: 模拟响应块
        """
        last_message = messages[-1].content if messages else ""
        response_content = f"这是对 '{last_message}' 的模拟流式响应。"
        
        # 模拟流式输出
        words = response_content.split()
        for i, word in enumerate(words):
            yield ModelResponse(
                content=word + (" " if i < len(words) - 1 else ""),
                metadata={"model": "mock", "chunk_index": i},
                usage={"prompt_tokens": len(str(messages)), "completion_tokens": i + 1}
            )


# 全局模型管理器实例
model_manager = ModelManager()

# 注册默认模型
model_manager.register_model("mock", MockModel) 