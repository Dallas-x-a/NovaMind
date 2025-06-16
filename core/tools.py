"""Novamind 工具接口实现。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class ToolMetadata(BaseModel):
    """Metadata for tools."""
    
    name: str
    description: str
    version: str = "0.1.0"
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel, Generic[T]):
    """工具执行结果。"""
    
    success: bool = Field(default=True)
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC, Generic[T]):
    """工具基类。"""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """初始化工具。
        
        参数:
            name: 工具名称
            description: 工具描述
            parameters: 工具参数定义
            **kwargs: 附加参数
        """
        self.metadata = ToolMetadata(
            name=name,
            description=description,
            parameters=parameters or {},
            **kwargs,
        )
        
    @property
    def name(self) -> str:
        """获取工具名称。"""
        return self.metadata.name
        
    @property
    def description(self) -> str:
        """获取工具描述。"""
        return self.metadata.description
        
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult[T]:
        """执行工具。
        
        参数:
            **kwargs: 工具参数
            
        返回:
            工具执行结果
        """
        pass
        
    def validate_parameters(self, **kwargs: Any) -> bool:
        """验证工具参数。
        
        参数:
            **kwargs: 要验证的参数
            
        返回:
            参数是否有效
        """
        required_params = {
            k: v for k, v in self.metadata.parameters.items()
            if v.get("required", False)
        }
        
        for param, spec in required_params.items():
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
                
            param_type = spec.get("type", str)
            if not isinstance(kwargs[param], param_type):
                raise TypeError(
                    f"Invalid type for parameter {param}: "
                    f"expected {param_type}, got {type(kwargs[param])}"
                )
                
        return True
        
    def get_parameter_schema(self) -> Dict[str, Any]:
        """获取工具参数模式。"""
        return self.metadata.parameters
        
    def cleanup(self) -> None:
        """清理工具资源。"""
        pass
        
    def __str__(self) -> str:
        """工具字符串表示。"""
        return f"{self.name} ({self.description})"
        
    def __repr__(self) -> str:
        """详细工具字符串表示。"""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"description='{self.description}', "
            f"version='{self.metadata.version}'"
            ")"
        )


class ToolRegistry:
    """注册表管理工具。"""
    
    def __init__(self):
        """初始化注册表。"""
        self._tools: Dict[str, BaseTool] = {}
        
    def register(self, tool: BaseTool) -> None:
        """注册工具。
        
        参数:
            tool: 要注册的工具
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        self._tools[tool.name] = tool
        
    def unregister(self, tool_name: str) -> None:
        """注销工具。
        
        参数:
            tool_name: 要注销工具的名称
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool {tool_name} not found")
        del self._tools[tool_name]
        
    def get_tool(self, tool_name: str) -> BaseTool:
        """按名称获取工具。
        
        参数:
            tool_name: 要获取工具的名称
            
        返回:
            请求的工具
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool {tool_name} not found")
        return self._tools[tool_name]
        
    def list_tools(self) -> List[str]:
        """列出所有注册的工具。
        
        返回:
            工具名称列表
        """
        return list(self._tools.keys())
        
    def get_tool_metadata(self, tool_name: str) -> ToolMetadata:
        """获取工具的元数据。
        
        参数:
            tool_name: 要获取元数据的工具名称
            
        返回:
            工具元数据
        """
        return self.get_tool(tool_name).metadata 