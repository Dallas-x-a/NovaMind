"""
NovaMind框架高级智能体系统

企业级多智能体系统，具有基于角色的协作、
状态管理和高级通信协议。

主要功能：
- 多智能体协作系统
- 基于角色的任务分配
- 异步消息通信
- 性能监控和指标收集
- 内存管理和状态跟踪
- 心跳检测和健康监控
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from datetime import datetime
import json
import logging

from pydantic import BaseModel, Field
from loguru import logger

from .models import ModelManager, ModelConfig
from .memory import MemoryManager, ConversationMemory
from .tools import ToolRegistry, ToolExecutor
from .config import Config
from .monitor import PerformanceMetrics


class AgentRole(str, Enum):
    """
    智能体角色定义 - 用于不同的职责分工
    
    不同的角色具有不同的能力和职责，支持智能体间的专业化协作
    """
    COORDINATOR = "coordinator"    # 协调者 - 负责任务分配和协调，管理整个系统的工作流程
    EXECUTOR = "executor"          # 执行者 - 负责具体任务执行，处理实际的工作内容
    ANALYZER = "analyzer"          # 分析者 - 负责数据分析和处理，提供洞察和决策支持
    VALIDATOR = "validator"        # 验证者 - 负责结果验证和检查，确保输出质量
    RESEARCHER = "researcher"      # 研究者 - 负责信息搜索和研究，收集相关数据
    CREATOR = "creator"            # 创建者 - 负责内容创作和生成，如文本、图像等
    SPECIALIST = "specialist"      # 专家 - 负责特定领域的专业任务，具有深度专业知识
    GENERALIST = "generalist"      # 通才 - 负责通用任务处理，具有广泛的知识面


class AgentState(str, Enum):
    """
    智能体状态定义 - 用于生命周期管理
    
    跟踪智能体的当前状态，支持状态转换和生命周期管理
    """
    IDLE = "idle"                  # 空闲状态 - 等待任务分配，可以接受新任务
    BUSY = "busy"                  # 忙碌状态 - 正在处理任务，暂时无法接受新任务
    THINKING = "thinking"          # 思考状态 - 正在分析和规划，进行决策过程
    EXECUTING = "executing"        # 执行状态 - 正在执行具体操作，处理实际工作
    WAITING = "waiting"            # 等待状态 - 等待外部响应或资源，暂停处理
    ERROR = "error"                # 错误状态 - 发生错误，需要处理或恢复
    TERMINATED = "terminated"      # 终止状态 - 已停止运行，不再接受任务


class MessageType(str, Enum):
    """
    消息类型定义 - 用于智能体间通信
    
    不同类型的消息具有不同的处理逻辑和优先级
    """
    TASK = "task"                  # 任务消息 - 分配任务给智能体执行
    RESULT = "result"              # 结果消息 - 返回任务执行结果
    REQUEST = "request"            # 请求消息 - 请求信息或服务
    RESPONSE = "response"          # 响应消息 - 响应请求，提供信息或服务
    NOTIFICATION = "notification"  # 通知消息 - 系统通知，如状态变更、事件提醒
    ERROR = "error"                # 错误消息 - 错误报告，包含错误信息和上下文
    HEARTBEAT = "heartbeat"        # 心跳消息 - 健康检查，确认智能体存活状态


@dataclass
class Message:
    """
    结构化消息类 - 用于智能体间通信
    
    提供标准化的消息格式，支持元数据、优先级和过期时间
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 消息唯一标识符
    sender: str = ""                                              # 发送者智能体ID
    recipient: str = ""                                           # 接收者智能体ID
    message_type: MessageType = MessageType.TASK                 # 消息类型
    content: Any = None                                          # 消息内容，可以是任意类型
    metadata: Dict[str, Any] = field(default_factory=dict)      # 元数据，包含额外信息
    timestamp: datetime = field(default_factory=datetime.now)   # 消息创建时间戳
    priority: int = 1                                            # 消息优先级，数字越小优先级越高
    expires_at: Optional[datetime] = None                       # 消息过期时间，None表示永不过期
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将消息转换为字典格式
        
        Returns:
            Dict[str, Any]: 包含消息所有字段的字典
        """
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


class AgentCapability(BaseModel):
    """
    智能体能力定义
    
    描述智能体具备的能力、工具和技能，用于任务匹配和分配
    """
    name: str                                                     # 能力名称，如"文本生成"、"数据分析"
    description: str                                              # 能力描述，详细说明能力的功能和用途
    tools: List[str] = Field(default_factory=list)               # 可用工具列表，智能体可以使用的工具名称
    skills: List[str] = Field(default_factory=list)              # 技能列表，智能体具备的技能标签
    performance_metrics: Dict[str, float] = Field(default_factory=dict)  # 性能指标，记录能力的历史表现


class AgentConfig(BaseModel):
    """
    智能体配置
    
    定义智能体的基本配置参数，包括模型、能力、性能和安全设置
    """
    name: str                                                     # 智能体名称，用于标识和显示
    role: AgentRole                                               # 智能体角色，决定其职责和功能
    model_config: ModelConfig                                     # 模型配置，定义使用的AI模型参数
    capabilities: List[AgentCapability] = Field(default_factory=list)  # 能力列表，智能体具备的所有能力
    max_concurrent_tasks: int = 5                                 # 最大并发任务数，限制同时处理的任务数量
    timeout: int = 300                                            # 超时时间(秒)，任务执行的最大时间限制
    retry_attempts: int = 3                                       # 重试次数，任务失败时的重试次数
    memory_size: int = 1000                                       # 内存大小，对话历史的最大记录数量
    enable_monitoring: bool = True                                # 是否启用监控，控制性能指标收集
    security_level: str = "standard"                              # 安全级别，控制访问权限和安全策略


class Agent(ABC):
    """
    基础智能体类 - 具有高级功能
    
    提供智能体的核心功能，包括：
    - 消息处理：异步消息接收、处理和转发
    - 任务执行：抽象任务执行接口，支持子类实现
    - 状态管理：智能体状态跟踪和转换
    - 性能监控：实时性能指标收集和报告
    - 内存管理：对话历史和状态持久化
    - 心跳检测：定期健康检查和状态报告
    """
    
    def __init__(self, config: AgentConfig):
        """
        初始化智能体
        
        Args:
            config: 智能体配置对象，包含所有必要的参数
        """
        self.config = config                                      # 配置信息
        self.id = str(uuid.uuid4())                              # 智能体唯一ID
        self.state = AgentState.IDLE                             # 初始状态为空闲
        self.memory = ConversationMemory(config.memory_size)     # 对话内存管理器
        self.model_manager = ModelManager()                      # 模型管理器
        self.tool_registry = ToolRegistry()                      # 工具注册表
        self.tool_executor = ToolExecutor()                      # 工具执行器
        self.metrics = PerformanceMetrics()                      # 性能指标收集器
        self.task_queue = asyncio.Queue()                        # 任务队列，存储待处理的消息
        self.active_tasks: Set[str] = set()                      # 活跃任务集合，跟踪正在执行的任务
        self.message_history: List[Message] = []                 # 消息历史，记录所有收发的消息
        self.peers: Dict[str, 'Agent'] = {}                      # 对等智能体字典，存储可通信的智能体
        
        # 初始化模型
        self.model = self.model_manager.get_model(config.model_config)
        
        # 设置日志记录
        self.logger = logger.bind(agent_id=self.id, agent_name=config.name)
        
        # 性能跟踪
        self.start_time = datetime.now()                         # 启动时间
        self.task_count = 0                                      # 任务计数
        self.success_count = 0                                   # 成功计数
        self.error_count = 0                                     # 错误计数
        
    async def start(self):
        """启动智能体，开始消息处理和心跳循环"""
        self.logger.info(f"启动智能体 {self.config.name}")
        self.state = AgentState.IDLE
        # 创建消息处理循环
        asyncio.create_task(self._message_loop())
        # 创建心跳循环
        asyncio.create_task(self._heartbeat_loop())
        
    async def stop(self):
        """停止智能体，结束所有处理循环"""
        self.logger.info(f"停止智能体 {self.config.name}")
        self.state = AgentState.TERMINATED
        
    async def send_message(self, message: Message) -> bool:
        """
        向其他智能体发送消息
        
        Args:
            message: 要发送的消息对象
            
        Returns:
            bool: 发送是否成功
        """
        try:
            if message.recipient in self.peers:
                # 如果接收者在对等智能体列表中，直接发送
                await self.peers[message.recipient].receive_message(message)
                self.message_history.append(message)
                return True
            else:
                self.logger.warning(f"未找到接收者 {message.recipient}")
                return False
        except Exception as e:
            self.logger.error(f"发送消息失败: {e}")
            return False
            
    async def receive_message(self, message: Message):
        """
        接收来自其他智能体的消息
        
        Args:
            message: 接收到的消息对象
        """
        self.message_history.append(message)
        await self.task_queue.put(message)
        
    async def _message_loop(self):
        """
        主消息处理循环
        
        持续监听任务队列，处理接收到的消息
        """
        while self.state != AgentState.TERMINATED:
            try:
                # 等待消息，超时时间为1秒
                message = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                await self._process_message(message)
            except asyncio.TimeoutError:
                # 超时继续循环
                continue
            except Exception as e:
                self.logger.error(f"消息循环错误: {e}")
                
    async def _process_message(self, message: Message):
        """
        处理接收到的消息
        
        Args:
            message: 要处理的消息对象
        """
        self.logger.debug(f"处理消息: {message.id}")
        
        # 根据消息类型进行不同处理
        if message.message_type == MessageType.TASK:
            await self._handle_task(message)
        elif message.message_type == MessageType.REQUEST:
            await self._handle_request(message)
        elif message.message_type == MessageType.NOTIFICATION:
            await self._handle_notification(message)
        else:
            self.logger.warning(f"未知消息类型: {message.message_type}")
            
    async def _handle_task(self, message: Message):
        """
        处理任务消息
        
        Args:
            message: 任务消息对象
        """
        self.logger.info(f"处理任务: {message.content}")
        self.state = AgentState.EXECUTING
        
        try:
            # 执行任务
            result = await self._execute_task(message.content, message.id)
            
            # 发送结果消息
            await self._send_response(
                message, 
                result, 
                MessageType.RESULT
            )
            
            self.success_count += 1
            
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            self.error_count += 1
            
            # 发送错误消息
            await self._send_response(
                message, 
                str(e), 
                MessageType.ERROR
            )
        finally:
            self.state = AgentState.IDLE
            
    async def _handle_request(self, message: Message):
        """
        处理请求消息
        
        Args:
            message: 请求消息对象
        """
        # 子类可以重写此方法处理特定请求
        pass
        
    async def _handle_notification(self, message: Message):
        """
        处理通知消息
        
        Args:
            message: 通知消息对象
        """
        self.logger.info(f"收到通知: {message.content}")
        
    async def _send_response(self, original_message: Message, content: Any, message_type: MessageType):
        """
        发送响应消息
        
        Args:
            original_message: 原始消息对象
            content: 响应内容
            message_type: 响应消息类型
        """
        response = Message(
            sender=self.id,
            recipient=original_message.sender,
            message_type=message_type,
            content=content,
            metadata={"original_message_id": original_message.id}
        )
        
        await self.send_message(response)
        
    async def _heartbeat_loop(self):
        """
        心跳循环 - 定期发送心跳消息
        
        每30秒向所有对等智能体发送心跳消息，确认存活状态
        """
        while self.state != AgentState.TERMINATED:
            try:
                # 每30秒发送一次心跳
                await asyncio.sleep(30)
                
                if self.state != AgentState.TERMINATED:
                    heartbeat = Message(
                        sender=self.id,
                        message_type=MessageType.HEARTBEAT,
                        content={"status": self.state.value, "timestamp": datetime.now().isoformat()}
                    )
                    
                    # 向所有对等智能体发送心跳
                    for peer_id in self.peers:
                        heartbeat.recipient = peer_id
                        await self.send_message(heartbeat)
                        
            except Exception as e:
                self.logger.error(f"心跳循环错误: {e}")
                
    @abstractmethod
    async def _execute_task(self, task: Any, task_id: str) -> Any:
        """
        执行任务 - 抽象方法，子类必须实现
        
        Args:
            task: 任务内容
            task_id: 任务ID
            
        Returns:
            Any: 任务执行结果
        """
        pass
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取智能体性能指标
        
        Returns:
            Dict[str, Any]: 包含各种性能指标的字典
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "agent_id": self.id,
            "agent_name": self.config.name,
            "state": self.state.value,
            "uptime_seconds": uptime,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / max(self.task_count, 1),
            "active_tasks": len(self.active_tasks),
            "memory_usage": len(self.message_history),
            "peer_count": len(self.peers)
        }


class LLMAgent(Agent):
    """
    基于大语言模型的智能体实现
    
    专门用于处理自然语言任务，如文本生成、对话、分析等
    """
    
    def __init__(self, config: AgentConfig):
        """
        初始化LLM智能体
        
        Args:
            config: 智能体配置对象
        """
        super().__init__(config)
        
    async def _execute_task(self, task: str, task_id: str) -> str:
        """
        执行基于LLM的任务
        
        Args:
            task: 任务内容，通常是自然语言描述
            task_id: 任务ID
            
        Returns:
            str: LLM生成的响应文本
        """
        self.logger.info(f"LLM智能体执行任务: {task}")
        
        try:
            # 使用模型生成响应
            response = await self.model.generate(task)
            
            # 记录到内存
            self.memory.add_conversation(task, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM任务执行失败: {e}")
            raise


class MultiAgentSystem:
    """
    多智能体系统 - 管理多个智能体的协作
    
    提供智能体注册、任务分配、系统监控等功能
    """
    
    def __init__(self, name: str):
        """
        初始化多智能体系统
        
        Args:
            name: 系统名称
        """
        self.name = name                                          # 系统名称
        self.agents: Dict[str, Agent] = {}                       # 智能体字典，存储所有注册的智能体
        self.coordinator: Optional[Agent] = None                 # 协调者智能体，负责任务分配
        self.logger = logger.bind(system_name=name)              # 系统日志记录器
        
    def add_agent(self, agent: Agent):
        """
        添加智能体到系统
        
        Args:
            agent: 要添加的智能体对象
        """
        self.agents[agent.id] = agent
        self.logger.info(f"添加智能体: {agent.config.name} (ID: {agent.id})")
        
        # 建立对等关系
        for other_agent in self.agents.values():
            if other_agent.id != agent.id:
                agent.peers[other_agent.id] = other_agent
                other_agent.peers[agent.id] = agent
                
    def set_coordinator(self, agent: Agent):
        """
        设置协调者智能体
        
        Args:
            agent: 要设置为协调者的智能体
        """
        self.coordinator = agent
        self.logger.info(f"设置协调者: {agent.config.name}")
        
    async def start_all(self):
        """启动所有智能体"""
        self.logger.info("启动多智能体系统")
        for agent in self.agents.values():
            await agent.start()
            
    async def stop_all(self):
        """停止所有智能体"""
        self.logger.info("停止多智能体系统")
        for agent in self.agents.values():
            await agent.stop()
            
    async def execute_task(self, task: str, target_agent: Optional[str] = None) -> str:
        """
        执行任务
        
        Args:
            task: 任务内容
            target_agent: 目标智能体ID，如果为None则自动选择
            
        Returns:
            str: 任务分配结果描述
        """
        if target_agent:
            # 如果指定了目标智能体，直接发送任务
            if target_agent in self.agents:
                agent = self.agents[target_agent]
                message = Message(
                    sender="system",
                    recipient=target_agent,
                    message_type=MessageType.TASK,
                    content=task
                )
                await agent.receive_message(message)
                return f"任务已发送到智能体: {agent.config.name}"
            else:
                raise ValueError(f"未找到智能体: {target_agent}")
        else:
            # 如果没有指定目标智能体，使用协调者或自动选择
            if self.coordinator:
                # 使用协调者分配任务
                message = Message(
                    sender="system",
                    recipient=self.coordinator.id,
                    message_type=MessageType.TASK,
                    content=task
                )
                await self.coordinator.receive_message(message)
                return f"任务已发送到协调者: {self.coordinator.config.name}"
            else:
                # 自动选择合适的智能体
                agent = self._select_agent_for_task(task)
                message = Message(
                    sender="system",
                    recipient=agent.id,
                    message_type=MessageType.TASK,
                    content=task
                )
                await agent.receive_message(message)
                return f"任务已发送到智能体: {agent.config.name}"
                
    def _select_agent_for_task(self, task: str) -> Agent:
        """
        为任务选择合适的智能体
        
        Args:
            task: 任务内容
            
        Returns:
            Agent: 选中的智能体对象
        """
        # 简单的选择策略：选择第一个空闲的智能体
        for agent in self.agents.values():
            if agent.state == AgentState.IDLE:
                return agent
                
        # 如果没有空闲的智能体，返回第一个
        return list(self.agents.values())[0]
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        获取系统性能指标
        
        Returns:
            Dict[str, Any]: 包含系统整体性能指标的字典
        """
        total_tasks = sum(agent.task_count for agent in self.agents.values())
        total_success = sum(agent.success_count for agent in self.agents.values())
        
        return {
            "system_name": self.name,
            "agent_count": len(self.agents),
            "total_tasks": total_tasks,
            "total_success": total_success,
            "overall_success_rate": total_success / max(total_tasks, 1),
            "agent_metrics": {
                agent_id: agent.get_metrics() 
                for agent_id, agent in self.agents.items()
            }
        } 