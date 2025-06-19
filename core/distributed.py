"""
Distributed Agent Support for NovaMind

支持分布式Agent远程注册、gRPC/HTTP通信、消息路由、跨节点任务分发、健康检查。
"""
import asyncio
import uuid
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import httpx

from .agent import Agent, Message, MessageType

class AgentCommProtocol(str, Enum):
    HTTP = "http"
    GRPC = "grpc"

@dataclass
class RemoteAgentInfo:
    name: str
    address: str  # e.g. http://host:port or grpc://host:port
    protocol: AgentCommProtocol = AgentCommProtocol.HTTP
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = 0.0
    is_alive: bool = True

class DistributedAgentRegistry:
    """分布式Agent注册与发现"""
    def __init__(self):
        self.remote_agents: Dict[str, RemoteAgentInfo] = {}
        self.heartbeat_interval = 10
        self.logger = logger.bind(module="DistributedAgentRegistry")

    def register_remote_agent(self, info: RemoteAgentInfo):
        self.remote_agents[info.name] = info
        self.logger.info(f"Registered remote agent: {info.name} at {info.address}")

    def unregister_remote_agent(self, name: str):
        if name in self.remote_agents:
            del self.remote_agents[name]
            self.logger.info(f"Unregistered remote agent: {name}")

    def get_remote_agent(self, name: str) -> Optional[RemoteAgentInfo]:
        return self.remote_agents.get(name)

    async def send_message(self, agent_name: str, message: Message) -> bool:
        info = self.get_remote_agent(agent_name)
        if not info:
            self.logger.warning(f"Remote agent {agent_name} not found")
            return False
        if info.protocol == AgentCommProtocol.HTTP:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(f"{info.address}/agent/message", json=message.to_dict())
                    return resp.status_code == 200
            except Exception as e:
                self.logger.error(f"Failed to send HTTP message: {e}")
                return False
        # gRPC可扩展
        return False

    async def heartbeat_loop(self):
        while True:
            for name, info in list(self.remote_agents.items()):
                alive = await self.ping_agent(info)
                info.is_alive = alive
                info.last_heartbeat = asyncio.get_event_loop().time()
            await asyncio.sleep(self.heartbeat_interval)

    async def ping_agent(self, info: RemoteAgentInfo) -> bool:
        if info.protocol == AgentCommProtocol.HTTP:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{info.address}/agent/ping")
                    return resp.status_code == 200
            except Exception:
                return False
        # gRPC可扩展
        return False 