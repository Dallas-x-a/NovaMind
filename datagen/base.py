"""Novamind 数据生成系统基础接口。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class DataGeneratorConfig(BaseModel):
    """数据生成器配置。"""
    
    seed: Optional[int] = None
    max_iterations: int = 10
    temperature: float = 0.7
    batch_size: int = 1
    validation_threshold: float = 0.8


class DataGenerator(ABC, Generic[T]):
    """数据生成器基类。"""
    
    def __init__(self, config: Optional[DataGeneratorConfig] = None):
        """初始化数据生成器。
        
        参数:
            config: 生成器配置
        """
        self.config = config or DataGeneratorConfig()
        
    @abstractmethod
    async def generate(self, **kwargs: Any) -> List[T]:
        """生成数据。
        
        参数:
            **kwargs: 生成参数
            
        返回:
            生成的数据列表
        """
        pass
        
    @abstractmethod
    async def validate(self, data: T) -> bool:
        """验证生成的数据。
        
        参数:
            data: 要验证的数据
            
        返回:
            数据是否有效
        """
        pass
        
    @abstractmethod
    async def filter(self, data_list: List[T]) -> List[T]:
        """过滤生成的数据。
        
        参数:
            data_list: 要过滤的数据列表
            
        返回:
            过滤后的数据列表
        """
        pass
        
    @abstractmethod
    async def save(self, data_list: List[T], path: str) -> None:
        """保存生成的数据。
        
        参数:
            data_list: 要保存的数据列表
            path: 保存路径
        """
        pass
        
    @abstractmethod
    async def load(self, path: str) -> List[T]:
        """加载生成的数据。
        
        参数:
            path: 数据路径
            
        返回:
            加载的数据列表
        """
        pass 