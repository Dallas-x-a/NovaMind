"""Novamind JSON 文件存储实现。"""

import json
import os
from typing import Any, Dict, Generic, Optional, TypeVar
from loguru import logger

from novamind.storages.base import BaseStorage

T = TypeVar("T")


class JSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器。"""
    
    def default(self, obj: Any) -> Any:
        """处理特殊类型的序列化。"""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return super().default(obj)


class JSONStorage(BaseStorage[T]):
    """JSON 文件存储实现。"""
    
    def __init__(self, file_path: str):
        """初始化 JSON 存储。
        
        参数:
            file_path: JSON 文件路径
        """
        self.file_path = file_path
        self._storage: Dict[str, T] = {}
        self._load()
        logger.info(f"初始化 JSON 存储: {file_path}")
        
    def _load(self) -> None:
        """从文件加载数据。"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._storage = json.load(f)
                logger.debug(f"从文件加载数据: {self.file_path}")
            except Exception as e:
                logger.error(f"加载文件时出错: {e}")
                self._storage = {}
                
    async def _save(self) -> None:
        """保存数据到文件。"""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._storage, f, ensure_ascii=False, indent=2, cls=JSONEncoder)
            logger.debug(f"保存数据到文件: {self.file_path}")
        except Exception as e:
            logger.error(f"保存文件时出错: {e}")
            
    async def get(self, key: str) -> Optional[T]:
        """获取存储的数据。"""
        return self._storage.get(key)
        
    async def set(self, key: str, value: T) -> None:
        """设置存储的数据。"""
        self._storage[key] = value
        await self._save()
        logger.debug(f"设置数据: {key}")
        
    async def delete(self, key: str) -> None:
        """删除存储的数据。"""
        if key in self._storage:
            del self._storage[key]
            await self._save()
            logger.debug(f"删除数据: {key}")
            
    async def exists(self, key: str) -> bool:
        """检查数据是否存在。"""
        return key in self._storage
        
    async def list_keys(self, pattern: Optional[str] = None) -> list[str]:
        """列出所有键。"""
        if pattern is None:
            return list(self._storage.keys())
            
        # 将 glob 模式转换为正则表达式
        import re
        regex_pattern = pattern.replace(".", "\\.").replace("*", ".*").replace("?", ".")
        regex = re.compile(f"^{regex_pattern}$")
        
        return [key for key in self._storage.keys() if regex.match(key)]
        
    async def clear(self) -> None:
        """清空所有数据。"""
        self._storage.clear()
        await self._save()
        logger.info("清空存储")
        
    async def close(self) -> None:
        """关闭存储连接。"""
        await self._save()
        logger.info("关闭 JSON 存储")
        
    def __len__(self) -> int:
        """获取存储的数据数量。"""
        return len(self._storage)
        
    def __contains__(self, key: str) -> bool:
        """检查键是否存在。"""
        return key in self._storage 