"""Novamind 核心代理实现。"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from loguru import logger

from novamind.core.config import Config
from novamind.core.tools import BaseTool
from novamind.core.memory import Memory
from novamind.core.models import BaseModel as BaseLLM


class AgentState(BaseModel):
    """代理状态管理。"""
    
    memory: Memory = Field(default_factory=Memory)
    tools: List[BaseTool] = Field(default_factory=list)
    config: Config = Field(default_factory=Config)
    model: Optional[BaseLLM] = None
    name: str = "agent"
    description: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class Agent:
    """Novamind 主代理类。"""
    
    def __init__(
        self,
        name: str = "agent",
        model: Optional[Union[str, BaseLLM]] = None,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[Config] = None,
        description: Optional[str] = None,
    ):
        """初始化代理。
        
        参数:
            name: 代理名称
            model: 使用的 LLM 模型（字符串名称或模型实例）
            tools: 代理可用的工具列表
            config: 代理配置
            description: 代理描述
        """
        self.state = AgentState(
            name=name,
            description=description,
            tools=tools or [],
            config=config or Config(),
        )
        
        if isinstance(model, str):
            self.state.model = self._load_model(model)
        else:
            self.state.model = model
            
        logger.info(f"初始化代理: {name}")
        
    def _load_model(self, model_name: str) -> BaseLLM:
        """通过名称加载模型。"""
        # 实现将稍后添加
        raise NotImplementedError("模型加载尚未实现")
        
    async def run(
        self,
        task: str,
        environment: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """运行代理执行任务。
        
        参数:
            task: 任务描述或提示
            environment: 可选的环境上下文
            **kwargs: 附加参数
            
        返回:
            包含代理响应和元数据的字典
        """
        if not self.state.model:
            raise ValueError("未指定代理模型")
            
        # 准备上下文
        context = {
            "task": task,
            "environment": environment,
            "tools": self.state.tools,
            "memory": self.state.memory,
            **kwargs,
        }
        
        # 执行任务
        try:
            response = await self.state.model.generate(context)
            self.state.memory.add(task, response)
            return {
                "response": response,
                "metadata": {
                    "agent": self.state.name,
                    "model": self.state.model.__class__.__name__,
                    "tools_used": [t.name for t in self.state.tools],
                }
            }
        except Exception as e:
            logger.error(f"运行代理时出错: {e}")
            raise
            
    def add_tool(self, tool: BaseTool) -> None:
        """向代理添加工具。"""
        self.state.tools.append(tool)
        logger.info(f"添加工具: {tool.name}")
        
    def remove_tool(self, tool_name: str) -> None:
        """从代理中移除工具。"""
        self.state.tools = [t for t in self.state.tools if t.name != tool_name]
        logger.info(f"移除工具: {tool_name}")
        
    def update_config(self, **kwargs: Any) -> None:
        """更新代理配置。"""
        for key, value in kwargs.items():
            setattr(self.state.config, key, value)
        logger.info(f"更新配置: {kwargs}")
        
    def get_state(self) -> Dict[str, Any]:
        """获取当前代理状态。"""
        return self.state.model_dump() 