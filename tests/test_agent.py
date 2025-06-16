"""Novamind 代理测试。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from novamind.core.agent import Agent, AgentState
from novamind.core.tools import BaseTool, ToolResult
from novamind.core.models import BaseModel as BaseLLM, ModelResponse
from novamind.core.environment import Environment


class MockTool(BaseTool):
    """模拟工具。"""
    
    def __init__(self, name: str = "mock_tool"):
        """初始化模拟工具。"""
        super().__init__(
            name=name,
            description="用于测试的模拟工具",
        )
        self.executed = False
        
    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行模拟工具。"""
        self.executed = True
        return ToolResult(
            success=True,
            data={"executed": True},
        )


class MockModel(BaseLLM):
    """模拟模型。"""
    
    def __init__(self):
        """初始化模拟模型。"""
        self.generated = False
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """生成模拟响应。"""
        self.generated = True
        return ModelResponse(
            content="模拟响应",
            metadata={"model": "mock"},
        )
        
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ):
        """生成模拟流式响应。"""
        self.generated = True
        yield ModelResponse(
            content="模拟流式响应",
            metadata={"model": "mock"},
        )


@pytest.fixture
def mock_model():
    """创建模拟模型。"""
    return MockModel()


@pytest.fixture
def mock_tool():
    """创建模拟工具。"""
    return MockTool()


@pytest.fixture
def agent(mock_model, mock_tool):
    """创建测试代理。"""
    return Agent(
        name="test_agent",
        model=mock_model,
        tools=[mock_tool],
        description="测试代理",
    )


@pytest.fixture
def environment():
    """创建测试环境。"""
    return Environment(
        variables={"test_var": "test_value"},
        constraints={"max_tokens": 100},
    )


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """测试代理初始化。"""
    assert agent.state.name == "test_agent"
    assert agent.state.model is not None
    assert len(agent.state.tools) == 1
    assert agent.state.description == "测试代理"


@pytest.mark.asyncio
async def test_agent_run(agent, environment):
    """测试代理运行。"""
    response = await agent.run("测试任务", environment=environment)
    assert response["response"].content == "模拟响应"
    assert agent.state.model.generated
    assert agent.state.tools[0].executed


@pytest.mark.asyncio
async def test_agent_tool_management(agent):
    """测试代理工具管理。"""
    # 添加工具
    new_tool = MockTool("new_tool")
    agent.add_tool(new_tool)
    assert len(agent.state.tools) == 2
    
    # 移除工具
    agent.remove_tool("mock_tool")
    assert len(agent.state.tools) == 1
    assert agent.state.tools[0].name == "new_tool"


@pytest.mark.asyncio
async def test_agent_config_update(agent):
    """测试代理配置更新。"""
    agent.update_config(temperature=0.8)
    assert agent.state.config.temperature == 0.8


@pytest.mark.asyncio
async def test_agent_error_handling(agent):
    """测试代理错误处理。"""
    # 模拟模型错误
    agent.state.model.generate = AsyncMock(side_effect=Exception("模型错误"))
    
    with pytest.raises(Exception):
        await agent.run("测试任务")


@pytest.mark.asyncio
async def test_agent_memory_management(agent):
    """测试代理内存管理。"""
    # 运行任务
    await agent.run("测试任务 1")
    await agent.run("测试任务 2")
    
    # 检查内存
    assert len(agent.state.memory.messages) == 2


@pytest.mark.asyncio
async def test_agent_environment_interaction(agent, environment):
    """测试代理环境交互。"""
    # 设置环境变量
    environment.set_variable("test_key", "test_value")
    
    # 运行任务
    response = await agent.run("测试任务", environment=environment)
    
    # 验证环境变量
    assert environment.get_variable("test_key") == "test_value"
    assert response["metadata"]["agent"] == "test_agent" 