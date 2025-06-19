"""
NovaMind工具系统实现

提供统一的工具接口、注册管理和执行框架。
支持智能体使用各种工具进行任务处理。

主要功能：
- 统一工具接口：标准化的工具定义和执行接口
- 工具注册管理：动态工具注册、查找和管理
- 参数验证：自动参数类型和有效性检查
- 工具链执行：支持多个工具的串行执行
- 错误处理：统一的错误处理和结果格式
- 元数据管理：工具的描述、版本、标签等信息
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class ToolMetadata(BaseModel):
    """
    工具元数据定义
    
    描述工具的基本信息，包括名称、描述、版本、作者等
    """
    
    name: str                                    # 工具名称 - 唯一标识符
    description: str                             # 工具描述 - 功能说明
    version: str = "0.1.0"                      # 工具版本 - 版本号
    author: Optional[str] = None                # 工具作者 - 开发者信息
    tags: List[str] = Field(default_factory=list)  # 工具标签 - 分类标签
    parameters: Dict[str, Any] = Field(default_factory=dict)  # 参数定义 - 参数模式


class ToolResult(BaseModel, Generic[T]):
    """
    工具执行结果
    
    标准化的工具执行结果格式，包含成功状态、数据和错误信息
    """
    
    success: bool = Field(default=True)         # 执行是否成功 - 布尔标志
    data: Optional[T] = None                    # 执行结果数据 - 泛型数据
    error: Optional[str] = None                 # 错误信息 - 错误描述
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 元数据 - 额外信息


class BaseTool(ABC, Generic[T]):
    """
    工具基类 - 所有工具的基础类
    
    定义了工具的标准接口，包括初始化、执行、参数验证等功能
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        初始化工具
        
        Args:
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
        """
        获取工具名称
        
        Returns:
            str: 工具名称
        """
        return self.metadata.name
        
    @property
    def description(self) -> str:
        """
        获取工具描述
        
        Returns:
            str: 工具描述
        """
        return self.metadata.description
        
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult[T]:
        """
        执行工具 - 抽象方法，子类必须实现
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            ToolResult[T]: 工具执行结果
        """
        pass
        
    def validate_parameters(self, **kwargs: Any) -> bool:
        """
        验证工具参数
        
        检查参数的类型和必需性，确保参数符合工具要求
        
        Args:
            **kwargs: 要验证的参数
            
        Returns:
            bool: 参数是否有效
            
        Raises:
            ValueError: 缺少必需参数时
            TypeError: 参数类型不匹配时
        """
        required_params = {
            k: v for k, v in self.metadata.parameters.items()
            if v.get("required", False)
        }
        
        for param, spec in required_params.items():
            if param not in kwargs:
                raise ValueError(f"缺少必需参数: {param}")
                
            param_type = spec.get("type", str)
            if not isinstance(kwargs[param], param_type):
                raise TypeError(
                    f"参数 {param} 类型无效: "
                    f"期望 {param_type}, 实际 {type(kwargs[param])}"
                )
                
        return True
        
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        获取工具参数模式
        
        Returns:
            Dict[str, Any]: 参数定义字典
        """
        return self.metadata.parameters
        
    def cleanup(self) -> None:
        """
        清理工具资源
        
        在工具不再使用时释放相关资源
        """
        pass
        
    def __str__(self) -> str:
        """
        工具字符串表示
        
        Returns:
            str: 工具的字符串表示
        """
        return f"{self.name} ({self.description})"
        
    def __repr__(self) -> str:
        """
        详细工具字符串表示
        
        Returns:
            str: 工具的详细字符串表示
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"description='{self.description}', "
            f"version='{self.metadata.version}'"
            ")"
        )


class ToolRegistry:
    """
    工具注册表 - 管理所有注册的工具
    
    提供工具的注册、注销、查找和管理功能，
    支持按名称、标签等方式查找工具
    """
    
    def __init__(self):
        """
        初始化注册表
        """
        self._tools: Dict[str, BaseTool] = {}  # 工具字典，以工具名称为键
        
    def register(self, tool: BaseTool) -> None:
        """
        注册工具
        
        Args:
            tool: 要注册的工具
            
        Raises:
            ValueError: 当工具名称已存在时
        """
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name} 已注册")
        self._tools[tool.name] = tool
        
    def unregister(self, tool_name: str) -> None:
        """
        注销工具
        
        Args:
            tool_name: 要注销工具的名称
            
        Raises:
            ValueError: 当工具不存在时
        """
        if tool_name not in self._tools:
            raise ValueError(f"工具 {tool_name} 未找到")
        del self._tools[tool_name]
        
    def get_tool(self, tool_name: str) -> BaseTool:
        """
        按名称获取工具
        
        Args:
            tool_name: 要获取工具的名称
            
        Returns:
            BaseTool: 请求的工具
            
        Raises:
            ValueError: 当工具不存在时
        """
        if tool_name not in self._tools:
            raise ValueError(f"工具 {tool_name} 未找到")
        return self._tools[tool_name]
        
    def list_tools(self) -> List[str]:
        """
        列出所有注册的工具
        
        Returns:
            List[str]: 工具名称列表
        """
        return list(self._tools.keys())
        
    def get_tool_metadata(self, tool_name: str) -> ToolMetadata:
        """
        获取工具的元数据
        
        Args:
            tool_name: 要获取元数据的工具名称
            
        Returns:
            ToolMetadata: 工具元数据
        """
        return self.get_tool(tool_name).metadata
    
    def get_tools_by_tag(self, tag: str) -> List[BaseTool]:
        """
        根据标签获取工具
        
        Args:
            tag: 标签名称
            
        Returns:
            List[BaseTool]: 具有指定标签的工具列表
        """
        return [
            tool for tool in self._tools.values()
            if tag in tool.metadata.tags
        ]
    
    def search_tools(self, query: str) -> List[BaseTool]:
        """
        搜索工具
        
        根据查询字符串在工具名称和描述中搜索
        
        Args:
            query: 搜索查询字符串
            
        Returns:
            List[BaseTool]: 匹配的工具列表
        """
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self._tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                matching_tools.append(tool)
                
        return matching_tools
    
    def get_tool_count(self) -> int:
        """
        获取注册的工具数量
        
        Returns:
            int: 工具数量
        """
        return len(self._tools)


class ToolExecutor:
    """
    工具执行器 - 负责执行工具和工具链
    
    提供工具的安全执行环境，支持单个工具和工具链的执行
    """
    
    def __init__(self, registry: ToolRegistry):
        """
        初始化工具执行器
        
        Args:
            registry: 工具注册表
        """
        self.registry = registry
        
    async def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        执行单个工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            ToolResult: 执行结果
            
        Raises:
            ValueError: 当工具不存在时
        """
        try:
            tool = self.registry.get_tool(tool_name)
            
            # 验证参数
            tool.validate_parameters(**kwargs)
            
            # 执行工具
            result = await tool.execute(**kwargs)
            
            return result
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"tool_name": tool_name}
            )
    
    async def execute_tool_chain(self, tool_chain: List[Dict[str, Any]]) -> List[ToolResult]:
        """
        执行工具链
        
        按顺序执行多个工具，前一个工具的输出可以作为后一个工具的输入
        
        Args:
            tool_chain: 工具链定义，每个元素包含工具名称和参数
            
        Returns:
            List[ToolResult]: 每个工具的执行结果
        """
        results = []
        previous_result = None
        
        for tool_config in tool_chain:
            tool_name = tool_config["tool"]
            params = tool_config.get("params", {})
            
            # 如果前一个工具有结果，可以传递给下一个工具
            if previous_result and previous_result.success:
                params["previous_result"] = previous_result.data
                
            result = await self.execute_tool(tool_name, **params)
            results.append(result)
            previous_result = result
            
            # 如果某个工具失败，可以选择停止或继续
            if not result.success:
                break
                
        return results


class CalculatorTool(BaseTool):
    """
    计算器工具 - 基础数学计算
    
    提供基本的数学运算功能，包括加减乘除、幂运算等
    """
    
    def __init__(self):
        """
        初始化计算器工具
        """
        super().__init__(
            name="calculator",
            description="基础数学计算工具，支持加减乘除、幂运算等",
            parameters={
                "operation": {
                    "type": str,
                    "required": True,
                    "description": "运算类型：add, subtract, multiply, divide, power"
                },
                "a": {
                    "type": (int, float),
                    "required": True,
                    "description": "第一个操作数"
                },
                "b": {
                    "type": (int, float),
                    "required": True,
                    "description": "第二个操作数"
                }
            },
            version="1.0.0",
            author="NovaMind Team",
            tags=["math", "calculation", "basic"]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        执行计算操作
        
        Args:
            **kwargs: 包含operation, a, b参数
            
        Returns:
            ToolResult: 计算结果
        """
        try:
            self.validate_parameters(**kwargs)
            
            operation = kwargs["operation"]
            a = kwargs["a"]
            b = kwargs["b"]
            
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    raise ValueError("除数不能为零")
                result = a / b
            elif operation == "power":
                result = a ** b
            else:
                raise ValueError(f"不支持的运算类型: {operation}")
                
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "operation": operation,
                    "operands": [a, b]
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"operation": kwargs.get("operation", "unknown")}
            )


class WebSearchTool(BaseTool):
    """
    网络搜索工具 - 网络信息检索
    
    提供网络搜索功能，可以搜索网页、新闻、图片等信息
    """
    
    def __init__(self):
        """
        初始化网络搜索工具
        """
        super().__init__(
            name="web_search",
            description="网络搜索工具，支持网页、新闻、图片搜索",
            parameters={
                "query": {
                    "type": str,
                    "required": True,
                    "description": "搜索查询字符串"
                },
                "search_type": {
                    "type": str,
                    "required": False,
                    "default": "web",
                    "description": "搜索类型：web, news, images"
                },
                "max_results": {
                    "type": int,
                    "required": False,
                    "default": 10,
                    "description": "最大结果数量"
                }
            },
            version="1.0.0",
            author="NovaMind Team",
            tags=["web", "search", "information"]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        执行网络搜索
        
        Args:
            **kwargs: 包含query, search_type, max_results参数
            
        Returns:
            ToolResult: 搜索结果
        """
        try:
            self.validate_parameters(**kwargs)
            
            query = kwargs["query"]
            search_type = kwargs.get("search_type", "web")
            max_results = kwargs.get("max_results", 10)
            
            # 这里应该集成实际的搜索API
            # 示例实现
            results = [
                {
                    "title": f"搜索结果 {i}",
                    "url": f"https://example.com/result{i}",
                    "snippet": f"这是关于 '{query}' 的搜索结果 {i}"
                }
                for i in range(1, min(max_results + 1, 6))
            ]
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "query": query,
                    "search_type": search_type,
                    "result_count": len(results)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"query": kwargs.get("query", "unknown")}
            )


class ToolFactory:
    """
    工具工厂 - 创建和管理工具实例
    
    提供标准化的工具创建方法，简化工具的实例化过程
    """
    
    @staticmethod
    def create_calculator() -> CalculatorTool:
        """
        创建计算器工具
        
        Returns:
            CalculatorTool: 计算器工具实例
        """
        return CalculatorTool()
    
    @staticmethod
    def create_web_search() -> WebSearchTool:
        """
        创建网络搜索工具
        
        Returns:
            WebSearchTool: 网络搜索工具实例
        """
        return WebSearchTool()
    
    @staticmethod
    def create_tool_registry() -> ToolRegistry:
        """
        创建工具注册表
        
        Returns:
            ToolRegistry: 工具注册表实例
        """
        return ToolRegistry() 