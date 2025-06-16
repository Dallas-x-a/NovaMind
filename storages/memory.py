"""Novamind 内存存储实现。"""

from typing import Any, Dict, Generic, Optional, TypeVar
import re
from loguru import logger

from novamind.storages.base import BaseStorage

T = TypeVar("T")


class InMemoryStorage(BaseStorage[T]):
    """内存存储实现。"""
    
    def __init__(self):
        """初始化内存存储。"""
        self._storage: Dict[str, T] = {}
        logger.info("初始化内存存储")
        
    async def get(self, key: str) -> Optional[T]:
        """获取存储的数据。"""
        return self._storage.get(key)
        
    async def set(self, key: str, value: T) -> None:
        """设置存储的数据。"""
        self._storage[key] = value
        logger.debug(f"设置数据: {key}")
        
    async def delete(self, key: str) -> None:
        """删除存储的数据。"""
        if key in self._storage:
            del self._storage[key]
            logger.debug(f"删除数据: {key}")
            
    async def exists(self, key: str) -> bool:
        """检查数据是否存在。"""
        return key in self._storage
        
    async def list_keys(self, pattern: Optional[str] = None) -> list[str]:
        """列出所有键。"""
        if pattern is None:
            return list(self._storage.keys())
            
        # 将 glob 模式转换为正则表达式
        regex_pattern = pattern.replace(".", "\\.").replace("*", ".*").replace("?", ".")
        regex = re.compile(f"^{regex_pattern}$")
        
        return [key for key in self._storage.keys() if regex.match(key)]
        
    async def clear(self) -> None:
        """清空所有数据。"""
        self._storage.clear()
        logger.info("清空存储")
        
    async def close(self) -> None:
        """关闭存储连接。"""
        self.clear()
        logger.info("关闭内存存储")
        
    def __len__(self) -> int:
        """获取存储的数据数量。"""
        return len(self._storage)
        
    def __contains__(self, key: str) -> bool:
        """检查键是否存在。"""
        return key in self._storage 