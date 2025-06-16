"""Novamind 存储系统基础接口。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


class BaseStorage(ABC, Generic[T]):
    """存储系统基类。"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """获取存储的数据。
        
        参数:
            key: 数据键
            
        返回:
            存储的数据，如果不存在则返回 None
        """
        pass
        
    @abstractmethod
    async def set(self, key: str, value: T) -> None:
        """设置存储的数据。
        
        参数:
            key: 数据键
            value: 要存储的数据
        """
        pass
        
    @abstractmethod
    async def delete(self, key: str) -> None:
        """删除存储的数据。
        
        参数:
            key: 数据键
        """
        pass
        
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查数据是否存在。
        
        参数:
            key: 数据键
            
        返回:
            数据是否存在
        """
        pass
        
    @abstractmethod
    async def list_keys(self, pattern: Optional[str] = None) -> list[str]:
        """列出所有键。
        
        参数:
            pattern: 可选的键模式匹配
            
        返回:
            键列表
        """
        pass
        
    @abstractmethod
    async def clear(self) -> None:
        """清空所有数据。"""
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """关闭存储连接。"""
        pass 