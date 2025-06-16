"""Novamind 网络搜索工具实现。"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from loguru import logger
from duckduckgo_search import AsyncDDGS

from novamind.core.tools import BaseTool, ToolResult


class WebSearchResult(BaseModel):
    """网络搜索结果。"""
    
    title: str = Field(description="结果标题")
    link: str = Field(description="结果链接")
    snippet: str = Field(description="结果摘要")
    source: str = Field(description="结果来源")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class WebSearchTool(BaseTool):
    """网络搜索工具。"""
    
    def __init__(
        self,
        max_results: int = 5,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        **kwargs: Any,
    ):
        """初始化网络搜索工具。
        
        参数:
            max_results: 最大结果数
            region: 搜索区域
            safesearch: 安全搜索级别
            **kwargs: 附加参数
        """
        super().__init__(
            name="web_search",
            description="使用 DuckDuckGo 执行网络搜索",
            parameters={
                "query": {
                    "type": "string",
                    "description": "搜索查询",
                    "required": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大结果数",
                    "default": max_results,
                },
                "region": {
                    "type": "string",
                    "description": "搜索区域",
                    "default": region,
                },
                "safesearch": {
                    "type": "string",
                    "description": "安全搜索级别",
                    "default": safesearch,
                },
            },
            **kwargs,
        )
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch
        self.client = AsyncDDGS()
        logger.info("初始化网络搜索工具")
        
    async def execute(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: Optional[str] = None,
        safesearch: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行网络搜索。
        
        参数:
            query: 搜索查询
            max_results: 最大结果数
            region: 搜索区域
            safesearch: 安全搜索级别
            **kwargs: 附加参数
            
        返回:
            搜索结果
        """
        try:
            # 验证参数
            if not query:
                return ToolResult(
                    success=False,
                    error="搜索查询不能为空",
                )
                
            # 执行搜索
            results = []
            async for r in self.client.text(
                query,
                max_results=max_results or self.max_results,
                region=region or self.region,
                safesearch=safesearch or self.safesearch,
            ):
                results.append(
                    WebSearchResult(
                        title=r["title"],
                        link=r["link"],
                        snippet=r["body"],
                        source=r.get("source", "未知"),
                        metadata={
                            "rank": len(results) + 1,
                            "hostname": r.get("hostname", ""),
                        },
                    )
                )
                
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "query": query,
                    "num_results": len(results),
                },
            )
        except Exception as e:
            logger.error(f"执行网络搜索时出错: {e}")
            return ToolResult(
                success=False,
                error=str(e),
            )
            
    def cleanup(self) -> None:
        """清理资源。"""
        if hasattr(self, "client"):
            self.client.close() 