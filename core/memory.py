"""Memory management for Novamind agents."""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from loguru import logger


class MemoryItem(BaseModel):
    """Single memory item."""
    
    content: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)


class Memory(BaseModel):
    """Memory management system for agents."""
    
    items: List[MemoryItem] = Field(default_factory=list)
    max_items: int = Field(default=1000, gt=0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    
    def add(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Add a new memory item.
        
        Args:
            content: Content to remember
            metadata: Additional metadata
            importance: Importance score (0-1)
            tags: List of tags
        """
        item = MemoryItem(
            content=content,
            metadata=metadata or {},
            importance=importance,
            tags=tags or [],
        )
        
        self.items.append(item)
        self._prune()
        logger.debug(f"Added memory item: {item}")
        
    def get(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        """Retrieve memory items.
        
        Args:
            query: Search query
            tags: Filter by tags
            limit: Maximum number of items to return
            min_importance: Minimum importance score
            
        Returns:
            List of matching memory items
        """
        items = self.items
        
        # Filter by importance
        items = [i for i in items if i.importance >= min_importance]
        
        # Filter by tags
        if tags:
            items = [
                i for i in items
                if all(tag in i.tags for tag in tags)
            ]
            
        # Filter by query (simple text search for now)
        if query:
            items = [
                i for i in items
                if query.lower() in str(i.content).lower()
            ]
            
        # Sort by importance and timestamp
        items.sort(key=lambda x: (-x.importance, x.timestamp))
        
        # Apply limit
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
        """Update a memory item.
        
        Args:
            index: Index of item to update
            content: New content
            metadata: New metadata
            importance: New importance score
            tags: New tags
        """
        if not 0 <= index < len(self.items):
            raise IndexError(f"Invalid memory index: {index}")
            
        item = self.items[index]
        
        if content is not None:
            item.content = content
        if metadata is not None:
            item.metadata.update(metadata)
        if importance is not None:
            item.importance = importance
        if tags is not None:
            item.tags = tags
            
        logger.debug(f"Updated memory item {index}: {item}")
        
    def delete(self, index: int) -> None:
        """Delete a memory item.
        
        Args:
            index: Index of item to delete
        """
        if not 0 <= index < len(self.items):
            raise IndexError(f"Invalid memory index: {index}")
            
        del self.items[index]
        logger.debug(f"Deleted memory item {index}")
        
    def clear(self) -> None:
        """Clear all memory items."""
        self.items.clear()
        logger.debug("Cleared all memory items")
        
    def _prune(self) -> None:
        """Prune memory items if limits are exceeded."""
        # Prune by item count
        if len(self.items) > self.max_items:
            # Sort by importance and timestamp
            self.items.sort(key=lambda x: (-x.importance, x.timestamp))
            # Keep most important items
            self.items = self.items[:self.max_items]
            logger.debug(f"Pruned memory to {self.max_items} items")
            
        # TODO: Implement token-based pruning when max_tokens is set
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "items": [item.model_dump() for item in self.items],
            "max_items": self.max_items,
            "max_tokens": self.max_tokens,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create memory from dictionary."""
        items = [MemoryItem(**item) for item in data["items"]]
        return cls(
            items=items,
            max_items=data["max_items"],
            max_tokens=data["max_tokens"],
        ) 