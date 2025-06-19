"""
NovaMind智能体内存管理系统

提供智能体的记忆存储、检索和管理功能。
支持重要性评分、标签分类和自动清理机制。

主要功能：
- 记忆存储：支持多种类型记忆的存储和管理
- 智能检索：基于查询、标签、重要性的记忆检索
- 自动清理：防止内存溢出的自动清理机制
- 记忆分类：对话记忆、情节记忆、语义记忆等
- 重要性评分：基于重要性的记忆优先级管理
- 标签系统：支持记忆的分类和标签管理
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from loguru import logger


class MemoryItem(BaseModel):
    """
    单个记忆项
    
    定义记忆的基本结构，包含内容、时间戳、元数据等信息
    """
    
    content: Any                                    # 记忆内容 - 实际存储的数据
    timestamp: datetime = Field(default_factory=datetime.utcnow)  # 时间戳 - 创建时间
    metadata: Dict[str, Any] = Field(default_factory=dict)        # 元数据 - 额外信息
    importance: float = Field(default=0.5, ge=0.0, le=1.0)        # 重要性评分(0-1) - 记忆优先级
    tags: List[str] = Field(default_factory=list)                 # 标签列表 - 分类标签


class Memory(BaseModel):
    """
    智能体内存管理系统
    
    提供记忆的增删改查功能，支持自动清理和限制管理
    """
    
    items: List[MemoryItem] = Field(default_factory=list)  # 记忆项列表 - 存储所有记忆
    max_items: int = Field(default=1000, gt=0)             # 最大记忆项数量 - 防止内存溢出
    max_tokens: Optional[int] = Field(default=None, gt=0)  # 最大token数量 - 基于token的限制
    
    def add(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        添加新的记忆项
        
        Args:
            content: 要记忆的内容
            metadata: 附加元数据
            importance: 重要性评分(0-1)
            tags: 标签列表
        """
        item = MemoryItem(
            content=content,
            metadata=metadata or {},
            importance=importance,
            tags=tags or [],
        )
        
        self.items.append(item)
        self._prune()  # 自动清理
        logger.debug(f"添加记忆项: {item}")
        
    def get(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        """
        检索记忆项
        
        Args:
            query: 搜索查询 - 在记忆内容中搜索
            tags: 按标签过滤 - 必须包含所有指定标签
            limit: 返回的最大项数 - 限制返回结果数量
            min_importance: 最小重要性评分 - 过滤低重要性记忆
            
        Returns:
            List[MemoryItem]: 匹配的记忆项列表
        """
        items = self.items
        
        # 按重要性过滤
        items = [i for i in items if i.importance >= min_importance]
        
        # 按标签过滤
        if tags:
            items = [
                i for i in items
                if all(tag in i.tags for tag in tags)
            ]
            
        # 按查询过滤(简单的文本搜索)
        if query:
            items = [
                i for i in items
                if query.lower() in str(i.content).lower()
            ]
            
        # 按重要性和时间戳排序
        items.sort(key=lambda x: (-x.importance, x.timestamp))
        
        # 应用限制
        if limit:
            items = items[:limit]
            
        return items
        
    def update(
        self,
        index: int,
        content: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        更新记忆项
        
        Args:
            index: 要更新的项索引
            content: 新内容
            metadata: 新元数据
            importance: 新重要性评分
            tags: 新标签
            
        Raises:
            IndexError: 当索引无效时
        """
        if not 0 <= index < len(self.items):
            raise IndexError(f"无效的记忆索引: {index}")
            
        item = self.items[index]
        
        if content is not None:
            item.content = content
        if metadata is not None:
            item.metadata.update(metadata)
        if importance is not None:
            item.importance = importance
        if tags is not None:
            item.tags = tags
            
        logger.debug(f"更新记忆项 {index}: {item}")
        
    def delete(self, index: int) -> None:
        """
        删除记忆项
        
        Args:
            index: 要删除的项索引
            
        Raises:
            IndexError: 当索引无效时
        """
        if not 0 <= index < len(self.items):
            raise IndexError(f"无效的记忆索引: {index}")
            
        del self.items[index]
        logger.debug(f"删除记忆项 {index}")
        
    def clear(self) -> None:
        """
        清空所有记忆项
        
        Args:
            self: 当前内存实例
        """
        self.items.clear()
        logger.debug("清空所有记忆项")
        
    def _prune(self) -> None:
        """
        如果超出限制则清理记忆项
        
        Args:
            self: 当前内存实例
        """
        # 按项数量清理
        if len(self.items) > self.max_items:
            # 按重要性和时间戳排序
            self.items.sort(key=lambda x: (-x.importance, x.timestamp))
            # 保留最重要的项
            self.items = self.items[:self.max_items]
            logger.debug(f"清理记忆至 {self.max_items} 项")
            
        # TODO: 当设置max_tokens时实现基于token的清理
        
    def to_dict(self) -> Dict[str, Any]:
        """
        将内存转换为字典
        
        Returns:
            Dict[str, Any]: 内存的字典表示
        """
        return {
            "items": [item.model_dump() for item in self.items],
            "max_items": self.max_items,
            "max_tokens": self.max_tokens,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """
        从字典创建内存
        
        Args:
            data: 内存数据字典
            
        Returns:
            Memory: 内存实例
        """
        items = [MemoryItem(**item) for item in data["items"]]
        return cls(
            items=items,
            max_items=data["max_items"],
            max_tokens=data["max_tokens"],
        )


class ConversationMemory(Memory):
    """
    对话记忆 - 专门用于存储对话历史
    
    Args:
        max_items: 最大对话记忆数量
    """
    
    def __init__(self, max_items: int = 100):
        """
        初始化对话记忆
        
        Args:
            max_items: 最大对话记忆数量
        """
        super().__init__(max_items=max_items)
        
    def add_conversation(self, user_input: str, agent_response: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加对话记忆
        
        Args:
            user_input: 用户输入
            agent_response: 智能体响应
            metadata: 附加元数据
        """
        conversation = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": datetime.utcnow()
        }
        
        self.add(
            content=conversation,
            metadata=metadata or {},
            importance=0.7,  # 对话记忆通常具有中等重要性
            tags=["conversation"]
        )
        
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的对话
        
        Args:
            limit: 返回的对话数量
            
        Returns:
            List[Dict[str, Any]]: 最近的对话列表
        """
        items = self.get(tags=["conversation"], limit=limit)
        return [item.content for item in items]
        
    def get_conversation_context(self, query: str, limit: int = 5) -> str:
        """
        获取对话上下文
        
        根据查询获取相关的对话历史，用于上下文重建
        
        Args:
            query: 查询内容
            limit: 返回的对话数量
            
        Returns:
            str: 格式化的对话上下文
        """
        items = self.get(query=query, tags=["conversation"], limit=limit)
        
        context_parts = []
        for item in items:
            conv = item.content
            context_parts.append(f"用户: {conv['user_input']}")
            context_parts.append(f"智能体: {conv['agent_response']}")
            
        return "\n".join(context_parts)


class EpisodicMemory(Memory):
    """
    情节记忆 - 存储事件和经历
    
    Args:
        max_items: 最大情节记忆数量
    """
    
    def __init__(self, max_items: int = 500):
        """
        初始化情节记忆
        
        Args:
            max_items: 最大情节记忆数量
        """
        super().__init__(max_items=max_items)
        
    def add_episode(self, event: str, outcome: str, 
                   importance: float = 0.8, tags: Optional[List[str]] = None) -> None:
        """
        添加情节记忆
        
        Args:
            event: 事件描述
            outcome: 事件结果
            importance: 重要性评分
            tags: 标签列表
        """
        episode = {
            "event": event,
            "outcome": outcome,
            "timestamp": datetime.utcnow()
        }
        
        self.add(
            content=episode,
            importance=importance,
            tags=["episode"] + (tags or [])
        )
        
    def get_similar_episodes(self, event_description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取相似的情节记忆
        
        Args:
            event_description: 事件描述
            limit: 返回的记忆数量
            
        Returns:
            List[Dict[str, Any]]: 相似的情节记忆列表
        """
        items = self.get(query=event_description, tags=["episode"], limit=limit)
        return [item.content for item in items]


class SemanticMemory(Memory):
    """
    语义记忆 - 存储概念和知识
    
    Args:
        max_items: 最大语义记忆数量
    """
    
    def __init__(self, max_items: int = 1000):
        """
        初始化语义记忆
        
        Args:
            max_items: 最大语义记忆数量
        """
        super().__init__(max_items=max_items)
        
    def add_concept(self, concept: str, definition: str, 
                   examples: Optional[List[str]] = None,
                   importance: float = 0.9) -> None:
        """
        添加概念记忆
        
        Args:
            concept: 概念名称
            definition: 概念定义
            examples: 示例列表
            importance: 重要性评分
        """
        concept_data = {
            "concept": concept,
            "definition": definition,
            "examples": examples or [],
            "timestamp": datetime.utcnow()
        }
        
        self.add(
            content=concept_data,
            importance=importance,
            tags=["concept", "semantic"]
        )
        
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索概念
        
        Args:
            query: 搜索查询
            limit: 返回的概念数量
            
        Returns:
            List[Dict[str, Any]]: 匹配的概念列表
        """
        items = self.get(query=query, tags=["concept"], limit=limit)
        return [item.content for item in items]


class MemoryManager:
    """
    内存管理器 - 统一管理所有类型的内存
    
    Args:
        self: 当前内存管理器实例
    """
    
    def __init__(self):
        """
        初始化内存管理器
        """
        self.conversation_memory = ConversationMemory()  # 对话记忆
        self.episodic_memory = EpisodicMemory()          # 情节记忆
        self.semantic_memory = SemanticMemory()          # 语义记忆
        
    def add_conversation(self, user_input: str, agent_response: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加对话记忆
        
        Args:
            user_input: 用户输入
            agent_response: 智能体响应
            metadata: 附加元数据
        """
        self.conversation_memory.add_conversation(user_input, agent_response, metadata)
        
    def add_episode(self, event: str, outcome: str, 
                   importance: float = 0.8, tags: Optional[List[str]] = None) -> None:
        """
        添加情节记忆
        
        Args:
            event: 事件描述
            outcome: 事件结果
            importance: 重要性评分
            tags: 标签列表
        """
        self.episodic_memory.add_episode(event, outcome, importance, tags)
        
    def add_concept(self, concept: str, definition: str, 
                   examples: Optional[List[str]] = None,
                   importance: float = 0.9) -> None:
        """
        添加概念记忆
        
        Args:
            concept: 概念名称
            definition: 概念定义
            examples: 示例列表
            importance: 重要性评分
        """
        self.semantic_memory.add_concept(concept, definition, examples, importance)
        
    def get_context(self, query: str, memory_types: List[str] = None) -> Dict[str, Any]:
        """
        获取综合上下文
        
        从所有类型的内存中检索相关信息
        
        Args:
            query: 查询内容
            memory_types: 要搜索的内存类型列表
            
        Returns:
            Dict[str, Any]: 综合上下文信息
        """
        if memory_types is None:
            memory_types = ["conversation", "episodic", "semantic"]
            
        context = {}
        
        if "conversation" in memory_types:
            context["conversations"] = self.conversation_memory.get_recent_conversations(5)
            
        if "episodic" in memory_types:
            context["episodes"] = self.episodic_memory.get_similar_episodes(query, 3)
            
        if "semantic" in memory_types:
            context["concepts"] = self.semantic_memory.search_concepts(query, 5)
            
        return context
        
    def clear_all(self) -> None:
        """
        清空所有内存
        
        Args:
            self: 当前内存管理器实例
        """
        self.conversation_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息
        
        Returns:
            Dict[str, Any]: 各类型内存的统计信息
        """
        return {
            "conversation": {
                "total_items": len(self.conversation_memory.items),
                "max_items": self.conversation_memory.max_items
            },
            "episodic": {
                "total_items": len(self.episodic_memory.items),
                "max_items": self.episodic_memory.max_items
            },
            "semantic": {
                "total_items": len(self.semantic_memory.items),
                "max_items": self.semantic_memory.max_items
            }
        } 