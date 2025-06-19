"""
NovaMind分布式智能体通信的gRPC服务实现

提供分布式智能体系统的核心通信功能，包括：
- 智能体注册和发现：支持智能体的动态注册和注销
- 心跳检测和健康监控：定期检查智能体存活状态
- 任务分发和结果收集：异步任务分配和结果管理
- 负载均衡和故障恢复：智能的任务分配和故障处理
- 分布式通信：基于gRPC的高性能网络通信
"""

import asyncio
import grpc
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid

# gRPC相关导入
from concurrent import futures
import novamind_pb2
import novamind_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """
    智能体信息 - 用于注册和管理
    
    存储分布式系统中智能体的完整信息，包括能力、状态、位置等
    """
    agent_id: str                 # 智能体唯一标识符
    name: str                     # 智能体名称，用于显示和识别
    capabilities: List[str]       # 智能体能力列表，如["text_generation", "data_analysis"]
    endpoint: str                 # 智能体服务端点，如"192.168.1.100:8080"
    status: str                   # 智能体状态，如"active", "busy", "offline"
    last_heartbeat: datetime      # 最后心跳时间，用于健康检查
    load: float                   # 当前负载，0.0-1.0之间的数值
    metadata: Dict[str, Any]      # 元数据信息，存储额外的配置和状态


class AgentRegistry:
    """
    中央智能体注册中心 - 管理分布式系统中的所有智能体
    
    提供智能体的注册、注销、发现和健康监控功能
    """
    
    def __init__(self):
        """
        初始化智能体注册中心
        """
        self.agents: Dict[str, AgentInfo] = {}  # 智能体信息字典，以agent_id为键
        self.heartbeat_timeout = 30              # 心跳超时时间(秒)，超过此时间认为智能体失效
        
    def register_agent(self, agent_info: AgentInfo) -> bool:
        """
        注册新的智能体
        
        Args:
            agent_info: 智能体信息对象
            
        Returns:
            bool: 注册是否成功
        """
        try:
            self.agents[agent_info.agent_id] = agent_info
            logger.info(f"智能体 {agent_info.name} ({agent_info.agent_id}) 已注册")
            return True
        except Exception as e:
            logger.error(f"注册智能体失败: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        注销智能体
        
        Args:
            agent_id: 要注销的智能体ID
            
        Returns:
            bool: 注销是否成功
        """
        if agent_id in self.agents:
            agent_name = self.agents[agent_id].name
            del self.agents[agent_id]
            logger.info(f"智能体 {agent_name} ({agent_id}) 已注销")
            return True
        return False
    
    def update_heartbeat(self, agent_id: str) -> bool:
        """
        更新智能体心跳时间
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            bool: 更新是否成功
        """
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.now()
            return True
        return False
    
    def get_available_agents(self, capabilities: List[str] = None) -> List[AgentInfo]:
        """
        获取可用的智能体，支持能力过滤
        
        Args:
            capabilities: 所需能力列表，如果为None则返回所有可用智能体
            
        Returns:
            List[AgentInfo]: 可用的智能体列表
        """
        now = datetime.now()
        available = []
        
        for agent in self.agents.values():
            # 检查智能体是否存活
            if (now - agent.last_heartbeat).seconds > self.heartbeat_timeout:
                continue
                
            # 如果指定了能力要求，检查智能体是否具备
            if capabilities:
                if not all(cap in agent.capabilities for cap in capabilities):
                    continue
                    
            available.append(agent)
        
        return available
    
    def cleanup_dead_agents(self):
        """
        清理失效的智能体
        
        移除超过心跳超时时间的智能体，保持注册中心的清洁
        """
        now = datetime.now()
        dead_agents = []
        
        for agent_id, agent in self.agents.items():
            if (now - agent.last_heartbeat).seconds > self.heartbeat_timeout:
                dead_agents.append(agent_id)
        
        for agent_id in dead_agents:
            self.unregister_agent(agent_id)


class NovaMindServiceServicer(novamind_pb2_grpc.NovaMindServiceServicer):
    """
    NovaMind分布式系统的gRPC服务实现
    
    提供所有gRPC接口的具体实现，包括智能体管理、任务处理等
    """
    
    def __init__(self):
        """
        初始化gRPC服务
        """
        self.registry = AgentRegistry()          # 智能体注册中心
        self.task_queue = asyncio.Queue()        # 任务队列，存储待处理的任务
        self.results = {}                        # 结果存储，以task_id为键存储执行结果
        
    def RegisterAgent(self, request, context):
        """
        注册新的智能体 - gRPC接口实现
        
        Args:
            request: 注册请求对象，包含智能体信息
            context: gRPC上下文
            
        Returns:
            RegisterAgentResponse: 注册响应对象
        """
        try:
            agent_info = AgentInfo(
                agent_id=request.agent_id,
                name=request.name,
                capabilities=list(request.capabilities),
                endpoint=request.endpoint,
                status="active",
                last_heartbeat=datetime.now(),
                load=request.load,
                metadata=json.loads(request.metadata) if request.metadata else {}
            )
            
            success = self.registry.register_agent(agent_info)
            
            return novamind_pb2.RegisterAgentResponse(
                success=success,
                message="智能体注册成功" if success else "注册失败"
            )
        except Exception as e:
            logger.error(f"注册错误: {e}")
            return novamind_pb2.RegisterAgentResponse(
                success=False,
                message=str(e)
            )
    
    def UnregisterAgent(self, request, context):
        """
        注销智能体 - gRPC接口实现
        
        Args:
            request: 注销请求对象
            context: gRPC上下文
            
        Returns:
            UnregisterAgentResponse: 注销响应对象
        """
        try:
            success = self.registry.unregister_agent(request.agent_id)
            return novamind_pb2.UnregisterAgentResponse(
                success=success,
                message="智能体已注销" if success else "智能体未找到"
            )
        except Exception as e:
            logger.error(f"注销错误: {e}")
            return novamind_pb2.UnregisterAgentResponse(
                success=False,
                message=str(e)
            )
    
    def Heartbeat(self, request, context):
        """
        处理智能体心跳 - gRPC接口实现
        
        Args:
            request: 心跳请求对象
            context: gRPC上下文
            
        Returns:
            HeartbeatResponse: 心跳响应对象
        """
        try:
            success = self.registry.update_heartbeat(request.agent_id)
            return novamind_pb2.HeartbeatResponse(
                success=success,
                timestamp=int(datetime.now().timestamp())
            )
        except Exception as e:
            logger.error(f"心跳错误: {e}")
            return novamind_pb2.HeartbeatResponse(
                success=False,
                timestamp=int(datetime.now().timestamp())
            )
    
    def SubmitTask(self, request, context):
        """
        提交任务执行 - gRPC接口实现
        
        Args:
            request: 任务提交请求对象
            context: gRPC上下文
            
        Returns:
            SubmitTaskResponse: 任务提交响应对象
        """
        try:
            task_id = str(uuid.uuid4())
            task = {
                'id': task_id,
                'type': request.task_type,
                'data': request.task_data,
                'capabilities': list(request.required_capabilities),
                'priority': request.priority,
                'submitted_at': datetime.now(),
                'status': 'pending'
            }
            
            # 添加到任务队列
            asyncio.create_task(self.task_queue.put(task))
            
            return novamind_pb2.SubmitTaskResponse(
                task_id=task_id,
                success=True,
                message="任务提交成功"
            )
        except Exception as e:
            logger.error(f"任务提交错误: {e}")
            return novamind_pb2.SubmitTaskResponse(
                task_id="",
                success=False,
                message=str(e)
            )
    
    def GetTask(self, request, context):
        """
        为智能体获取下一个可用任务 - gRPC接口实现
        
        Args:
            request: 获取任务请求对象
            context: gRPC上下文
            
        Returns:
            GetTaskResponse: 获取任务响应对象
        """
        try:
            # 获取具有所需能力的可用智能体
            available_agents = self.registry.get_available_agents(request.capabilities)
            
            if not available_agents:
                return novamind_pb2.GetTaskResponse(
                    task_id="",
                    task_type="",
                    task_data="",
                    success=False,
                    message="没有可用的智能体"
                )
            
            # 尝试从队列获取任务
            try:
                task = asyncio.run(self.task_queue.get_nowait())
                
                # 检查智能体是否具备所需能力
                agent_capabilities = set()
                for agent in available_agents:
                    if agent.agent_id == request.agent_id:
                        agent_capabilities = set(agent.capabilities)
                        break
                
                if not all(cap in agent_capabilities for cap in task['capabilities']):
                    # 将任务放回队列
                    asyncio.create_task(self.task_queue.put(task))
                    return novamind_pb2.GetTaskResponse(
                        task_id="",
                        task_type="",
                        task_data="",
                        success=False,
                        message="智能体缺乏所需能力"
                    )
                
                return novamind_pb2.GetTaskResponse(
                    task_id=task['id'],
                    task_type=task['type'],
                    task_data=task['data'],
                    success=True,
                    message="任务已分配"
                )
            except asyncio.QueueEmpty:
                return novamind_pb2.GetTaskResponse(
                    task_id="",
                    task_type="",
                    task_data="",
                    success=False,
                    message="没有可用的任务"
                )
        except Exception as e:
            logger.error(f"获取任务错误: {e}")
            return novamind_pb2.GetTaskResponse(
                task_id="",
                task_type="",
                task_data="",
                success=False,
                message=str(e)
            )
    
    def SubmitResult(self, request, context):
        """
        提交任务执行结果 - gRPC接口实现
        
        Args:
            request: 结果提交请求对象
            context: gRPC上下文
            
        Returns:
            SubmitResultResponse: 结果提交响应对象
        """
        try:
            result = {
                'task_id': request.task_id,
                'agent_id': request.agent_id,
                'result_data': request.result_data,
                'status': request.status,
                'error_message': request.error_message,
                'execution_time': request.execution_time,
                'submitted_at': datetime.now()
            }
            
            self.results[request.task_id] = result
            
            return novamind_pb2.SubmitResultResponse(
                success=True,
                message="结果提交成功"
            )
        except Exception as e:
            logger.error(f"结果提交错误: {e}")
            return novamind_pb2.SubmitResultResponse(
                success=False,
                message=str(e)
            )
    
    def GetAgentStatus(self, request, context):
        """
        获取智能体状态 - gRPC接口实现
        
        Args:
            request: 状态查询请求对象
            context: gRPC上下文
            
        Returns:
            GetAgentStatusResponse: 状态查询响应对象
        """
        try:
            if request.agent_id in self.registry.agents:
                agent = self.registry.agents[request.agent_id]
                return novamind_pb2.GetAgentStatusResponse(
                    success=True,
                    agent_id=agent.agent_id,
                    name=agent.name,
                    status=agent.status,
                    capabilities=agent.capabilities,
                    load=agent.load,
                    last_heartbeat=int(agent.last_heartbeat.timestamp()),
                    metadata=json.dumps(agent.metadata)
                )
            else:
                return novamind_pb2.GetAgentStatusResponse(
                    success=False,
                    message="智能体未找到"
                )
        except Exception as e:
            logger.error(f"状态查询错误: {e}")
            return novamind_pb2.GetAgentStatusResponse(
                success=False,
                message=str(e)
            )


class gRPCServer:
    """
    gRPC服务器 - 启动和管理gRPC服务
    
    提供服务器的启动、停止和清理功能
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 50051):
        """
        初始化gRPC服务器
        
        Args:
            host: 服务器监听地址
            port: 服务器监听端口
        """
        self.host = host
        self.port = port
        self.server = None
        self.cleanup_task = None
        
    async def start(self):
        """
        启动gRPC服务器
        
        创建gRPC服务器实例，启动监听，并开始清理任务
        """
        try:
            # 创建gRPC服务器
            self.server = grpc.aio.server(
                futures.ThreadPoolExecutor(max_workers=10)
            )
            
            # 添加服务实现
            servicer = NovaMindServiceServicer()
            novamind_pb2_grpc.add_NovaMindServiceServicer_to_server(
                servicer, self.server
            )
            
            # 启动服务器
            listen_addr = f'{self.host}:{self.port}'
            self.server.add_insecure_port(listen_addr)
            await self.server.start()
            
            logger.info(f"gRPC服务器已启动，监听地址: {listen_addr}")
            
            # 启动清理任务
            self.cleanup_task = asyncio.create_task(self._cleanup_task())
            
        except Exception as e:
            logger.error(f"启动gRPC服务器失败: {e}")
            raise
            
    async def stop(self):
        """
        停止gRPC服务器
        
        停止服务器和清理任务
        """
        if self.server:
            await self.server.stop(grace=5)
            logger.info("gRPC服务器已停止")
            
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
                
    async def _cleanup_task(self):
        """
        清理任务 - 定期清理失效的智能体
        
        每60秒执行一次清理操作
        """
        while True:
            try:
                await asyncio.sleep(60)
                # 这里可以添加清理逻辑
                logger.debug("执行定期清理任务")
            except asyncio.CancelledError:
                break


class gRPCClient:
    """
    gRPC客户端 - 与gRPC服务器通信
    
    提供客户端连接、断开和所有gRPC方法调用
    """
    
    def __init__(self, server_address: str = 'localhost:50051'):
        """
        初始化gRPC客户端
        
        Args:
            server_address: 服务器地址，格式为"host:port"
        """
        self.server_address = server_address
        self.channel = None
        self.stub = None
        
    async def connect(self):
        """
        连接到gRPC服务器
        
        建立与服务器的连接并创建存根对象
        """
        try:
            self.channel = grpc.aio.insecure_channel(self.server_address)
            self.stub = novamind_pb2_grpc.NovaMindServiceStub(self.channel)
            logger.info(f"已连接到gRPC服务器: {self.server_address}")
        except Exception as e:
            logger.error(f"连接gRPC服务器失败: {e}")
            raise
            
    async def disconnect(self):
        """
        断开与gRPC服务器的连接
        
        关闭通道连接
        """
        if self.channel:
            await self.channel.close()
            logger.info("已断开gRPC服务器连接")
            
    async def register_agent(self, agent_id: str, name: str, capabilities: List[str], 
                           endpoint: str, load: float = 0.0, metadata: Dict = None):
        """
        注册智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            capabilities: 能力列表
            endpoint: 服务端点
            load: 当前负载
            metadata: 元数据
            
        Returns:
            bool: 注册是否成功
        """
        try:
            request = novamind_pb2.RegisterAgentRequest(
                agent_id=agent_id,
                name=name,
                capabilities=capabilities,
                endpoint=endpoint,
                load=load,
                metadata=json.dumps(metadata) if metadata else ""
            )
            
            response = await self.stub.RegisterAgent(request)
            return response.success
        except Exception as e:
            logger.error(f"注册智能体失败: {e}")
            return False
            
    async def heartbeat(self, agent_id: str):
        """
        发送心跳
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            bool: 心跳是否成功
        """
        try:
            request = novamind_pb2.HeartbeatRequest(agent_id=agent_id)
            response = await self.stub.Heartbeat(request)
            return response.success
        except Exception as e:
            logger.error(f"心跳失败: {e}")
            return False
            
    async def get_task(self, agent_id: str, capabilities: List[str]):
        """
        获取任务
        
        Args:
            agent_id: 智能体ID
            capabilities: 智能体能力列表
            
        Returns:
            Dict: 任务信息，如果失败则返回None
        """
        try:
            request = novamind_pb2.GetTaskRequest(
                agent_id=agent_id,
                capabilities=capabilities
            )
            
            response = await self.stub.GetTask(request)
            
            if response.success:
                return {
                    'task_id': response.task_id,
                    'task_type': response.task_type,
                    'task_data': response.task_data
                }
            else:
                logger.warning(f"获取任务失败: {response.message}")
                return None
        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return None
            
    async def submit_result(self, task_id: str, agent_id: str, result_data: str,
                          status: str, error_message: str = "", execution_time: float = 0.0):
        """
        提交任务结果
        
        Args:
            task_id: 任务ID
            agent_id: 智能体ID
            result_data: 结果数据
            status: 执行状态
            error_message: 错误信息
            execution_time: 执行时间
            
        Returns:
            bool: 提交是否成功
        """
        try:
            request = novamind_pb2.SubmitResultRequest(
                task_id=task_id,
                agent_id=agent_id,
                result_data=result_data,
                status=status,
                error_message=error_message,
                execution_time=execution_time
            )
            
            response = await self.stub.SubmitResult(request)
            return response.success
        except Exception as e:
            logger.error(f"提交结果失败: {e}")
            return False


# 示例使用代码
async def main():
    """
    主函数 - 演示gRPC服务器的启动和客户端使用
    
    包含服务器启动、客户端连接、智能体注册等完整流程
    """
    # 启动服务器
    server = gRPCServer()
    await server.start()
    
    try:
        # 创建客户端
        client = gRPCClient()
        await client.connect()
        
        # 注册智能体
        success = await client.register_agent(
            agent_id="agent_001",
            name="测试智能体",
            capabilities=["text_generation", "data_analysis"],
            endpoint="192.168.1.100:8080",
            load=0.5,
            metadata={"version": "1.0", "model": "gpt-4"}
        )
        
        if success:
            print("智能体注册成功")
            
            # 发送心跳
            await client.heartbeat("agent_001")
            
            # 获取任务
            task = await client.get_task("agent_001", ["text_generation"])
            if task:
                print(f"获取到任务: {task}")
                
                # 提交结果
                await client.submit_result(
                    task_id=task['task_id'],
                    agent_id="agent_001",
                    result_data="任务执行完成",
                    status="completed",
                    execution_time=1.5
                )
        
        await client.disconnect()
        
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main()) 