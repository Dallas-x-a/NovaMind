"""Novamind 基础代理示例。

这个示例展示了如何使用 Novamind 创建一个基础的智能代理，该代理可以：
1. 使用 OpenAI 模型进行对话
2. 使用网络搜索工具获取信息
3. 在特定环境下执行任务
4. 处理多个任务并管理资源

使用方法：
1. 确保设置了 OPENAI_API_KEY 环境变量
2. 运行脚本：python basic_agent.py
"""

import asyncio
import os
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger

from novamind.core.agent import Agent
from novamind.core.environment import Environment
from novamind.models.openai import OpenAIModel, OpenAIConfig
from novamind.tools.web_search import WebSearchTool


class AgentError(Exception):
    """代理相关错误的基类。"""
    pass


class ConfigurationError(AgentError):
    """配置相关错误。"""
    pass


class ExecutionError(AgentError):
    """执行相关错误。"""
    pass


async def create_agent() -> Agent:
    """创建并配置代理。

    Returns:
        Agent: 配置好的代理实例

    Raises:
        ConfigurationError: 当配置无效时
    """
    try:
        # 验证环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("未设置 OPENAI_API_KEY 环境变量")

        # 创建 OpenAI 模型
        model = OpenAIModel(
            config=OpenAIConfig(
                model_name="gpt-3.5-turbo",
                api_key=api_key,
            ),
        )

        # 创建网络搜索工具
        search_tool = WebSearchTool(max_results=3)

        # 创建代理
        return Agent(
            name="research_assistant",
            model=model,
            tools=[search_tool],
            description="一个能够进行网络搜索并提供信息的智能研究助手",
        )

    except Exception as e:
        raise ConfigurationError(f"创建代理时出错: {str(e)}")


async def execute_task(agent: Agent, task: str, env: Environment) -> Dict[str, Any]:
    """执行单个任务。

    Args:
        agent (Agent): 代理实例
        task (str): 要执行的任务
        env (Environment): 执行环境

    Returns:
        Dict[str, Any]: 包含响应和元数据的字典

    Raises:
        ExecutionError: 当任务执行失败时
    """
    try:
        logger.info(f"执行任务: {task}")
        response = await agent.run(task, environment=env)
        return response
    except Exception as e:
        raise ExecutionError(f"执行任务 '{task}' 时出错: {str(e)}")


async def main():
    """运行基础代理示例。"""
    agent = None
    try:
        # 加载环境变量
        load_dotenv()

        # 创建代理
        agent = await create_agent()

        # 创建环境
        env = Environment(
            variables={
                "max_iterations": 3,
                "temperature": 0.7,
            },
            constraints={
                "max_tokens": 1000,
                "timeout": 30,
            },
        )

        # 定义任务
        tasks = [
            "什么是量子计算？",
            "量子计算的主要应用领域有哪些？",
            "量子计算面临的主要挑战是什么？",
        ]

        # 执行任务
        for task in tasks:
            try:
                response = await execute_task(agent, task, env)
                print(f"\n任务: {task}")
                print(f"响应: {response['response'].content}")
                print(f"元数据: {response['metadata']}")
            except ExecutionError as e:
                logger.error(f"任务执行失败: {str(e)}")
                continue

            # 添加小延迟以避免速率限制
            await asyncio.sleep(1)

    except ConfigurationError as e:
        logger.error(f"配置错误: {str(e)}")
    except Exception as e:
        logger.error(f"发生未预期的错误: {str(e)}")
    finally:
        # 清理资源
        if agent and hasattr(agent, 'tools'):
            for tool in agent.tools:
                if hasattr(tool, 'cleanup'):
                    tool.cleanup()
        if agent and hasattr(agent, 'model'):
            del agent.model


if __name__ == "__main__":
    asyncio.run(main()) 