"""
NovaMind API/WEB服务接口

基于FastAPI，支持多租户、项目隔离、Agent/工具/知识/多模态推理等API。
提供企业级的RESTful API服务，支持权限控制和监控。

主要功能：
- 多租户支持：租户级别的资源隔离和管理
- 项目隔离：项目级别的权限和数据隔离
- 智能体管理：智能体的注册、配置和状态管理
- 任务调度：异步任务提交、执行和状态跟踪
- 知识检索：统一的知识查询接口
- 多模态推理：支持文本、图像、音频、视频处理
- 系统监控：实时系统状态和性能监控
- 权限控制：基于角色的访问控制(RBAC)
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import asyncio
from novamind.core import (
    Agent, LLMAgent, MultimodalAgent, AgentConfig, AgentRole, ModelConfig,
    MultiAgentSystem, TaskScheduler, SystemMonitor, SecurityManager
)
from novamind.core.knowledge import KnowledgeManager, KnowledgeQuery, KnowledgeSourceType
from novamind.core.multimodal import MultimodalInput, Modality
from novamind.core.security import Permission

# 创建FastAPI应用实例
app = FastAPI(
    title="NovaMind API", 
    version="2.0.0",
    description="NovaMind企业级多智能体系统API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 多租户与项目隔离
class TenantContext(BaseModel):
    """
    租户上下文 - 用于多租户和多项目隔离
    
    定义API请求的租户和项目上下文信息，
    用于实现资源隔离和权限控制
    """
    tenant_id: str      # 租户ID - 标识租户
    project_id: str     # 项目ID - 标识项目
    user_id: str        # 用户ID - 标识用户

# 简单token认证
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")  # OAuth2密码承载者认证方案
security_manager = SecurityManager()                     # 安全管理器

async def get_current_tenant(token: str = Depends(oauth2_scheme)) -> TenantContext:
    """
    获取当前租户上下文
    
    Args:
        token: JWT认证令牌
        
    Returns:
        TenantContext: 租户上下文对象
        
    Raises:
        HTTPException: 当令牌无效时抛出401错误
    """
    user_id = security_manager.verify_jwt_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="无效的认证令牌")
    # 这里可扩展为多租户/多项目
    return TenantContext(tenant_id="default", project_id="default", user_id=user_id)

# 系统核心对象
system = MultiAgentSystem(name="NovaMindMAS")    # 多智能体系统
scheduler = TaskScheduler()                      # 任务调度器
monitor = SystemMonitor()                        # 系统监控器
knowledge_manager = KnowledgeManager()           # 知识管理器

# API权限声明 - 定义各API端点所需的权限
API_PERMISSIONS = {
    "/agents/register": [Permission.ADMIN, Permission.CREATE],    # 智能体注册需要管理员权限
    "/tasks/submit": [Permission.WRITE, Permission.CREATE],       # 任务提交需要写权限
    "/tasks/{task_id}": [Permission.READ],                        # 任务查询需要读权限
    "/knowledge/query": [Permission.READ],                        # 知识查询需要读权限
    "/multimodal/infer": [Permission.EXECUTE],                    # 多模态推理需要执行权限
    "/monitor/summary": [Permission.READ],                        # 监控摘要需要读权限
}

def require_permission(api_path: str):
    """
    权限检查装饰器
    
    Args:
        api_path: API路径，用于查找所需权限
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        async def wrapper(*args, tenant: TenantContext = Depends(get_current_tenant), **kwargs):
            user_id = tenant.user_id
            # 这里假设user_id就是系统user_id
            perms = API_PERMISSIONS.get(api_path, [])
            for perm in perms:
                if not security_manager.access_control.check_permission(user_id, api_path, perm):
                    raise HTTPException(status_code=403, detail=f"需要权限: {perm}")
            return await func(*args, tenant=tenant, **kwargs)
        return wrapper
    return decorator

@app.on_event("startup")
async def startup_event():
    """
    应用启动事件
    
    Args:
        None
        
    Returns:
        None
    """
    await monitor.start()      # 启动系统监控
    await scheduler.start()    # 启动任务调度器
    # 可自动注册Agent/工具/知识源

@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭事件
    
    Args:
        None
        
    Returns:
        None
    """
    await monitor.stop()       # 停止系统监控
    await scheduler.stop()     # 停止任务调度器

# ==================== Agent注册与管理API ====================

@app.post("/agents/register")
@require_permission("/agents/register")
def register_agent(agent: Dict[str, Any], tenant: TenantContext = Depends(get_current_tenant)):
    """
    注册新的智能体
    
    Args:
        agent: 智能体配置信息
        tenant: 租户上下文
        
    Returns:
        Dict: 注册结果信息
        
    Raises:
        HTTPException: 注册失败时抛出400错误
    """
    try:
        config = AgentConfig(**agent)
        # 根据角色创建不同类型的智能体
        if config.role == AgentRole.RESEARCHER:
            agent_obj = LLMAgent(config)
        else:
            agent_obj = MultimodalAgent(config)
        
        # 添加到系统
        system.add_agent(agent_obj)
        scheduler.register_agent(config.name, agent_obj)
        
        return {
            "msg": "智能体注册成功", 
            "agent_id": config.name,
            "role": config.role.value,
            "capabilities": [cap.name for cap in config.capabilities]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"智能体注册失败: {str(e)}")

# ==================== 任务管理API ====================

class TaskRequest(BaseModel):
    """
    任务请求模型
    
    Args:
        None
        
    Returns:
        None
    """
    agent: str                              # 目标智能体 - 执行任务的智能体名称
    payload: Any                            # 任务载荷 - 任务的具体内容
    priority: Optional[int] = 2             # 任务优先级 - 数字越小优先级越高
    project_id: Optional[str] = "default"   # 项目ID - 任务所属的项目

@app.post("/tasks/submit")
@require_permission("/tasks/submit")
async def submit_task(req: TaskRequest, tenant: TenantContext = Depends(get_current_tenant)):
    """
    提交任务
    
    Args:
        req: 任务请求对象
        tenant: 租户上下文
        
    Returns:
        Dict: 任务提交结果
        
    Raises:
        HTTPException: 提交失败时抛出400错误
    """
    try:
        from novamind.core.scheduler import Task, TaskPriority
        
        task = Task(
            name=f"task_{uuid.uuid4().hex[:8]}",
            description="API提交的任务",
            payload=req.payload,
            priority=TaskPriority(req.priority or 2)
        )
        
        await scheduler.submit_task(task)
        
        return {
            "msg": "任务提交成功", 
            "task_id": task.id,
            "priority": task.priority.value,
            "status": "pending"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"任务提交失败: {str(e)}")

@app.get("/tasks/{task_id}")
@require_permission("/tasks/{task_id}")
def get_task_status(task_id: str, tenant: TenantContext = Depends(get_current_tenant)):
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        tenant: 租户上下文
        
    Returns:
        Dict: 任务状态信息
        
    Raises:
        HTTPException: 任务未找到时抛出404错误
    """
    status = scheduler.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务未找到")
    return status

# ==================== 知识检索API ====================

class KnowledgeRequest(BaseModel):
    """
    知识查询请求模型
    
    Args:
        None
        
    Returns:
        None
    """
    source_type: KnowledgeSourceType        # 知识源类型 - 查询的知识源
    query: str                              # 查询内容 - 具体的查询语句或内容
    params: Optional[Dict[str, Any]] = None # 查询参数 - 额外的查询参数

@app.post("/knowledge/query")
@require_permission("/knowledge/query")
def knowledge_query(req: KnowledgeRequest, tenant: TenantContext = Depends(get_current_tenant)):
    """
    执行知识查询
    
    Args:
        req: 知识查询请求对象
        tenant: 租户上下文
        
    Returns:
        Dict: 查询结果
        
    Raises:
        HTTPException: 查询失败时抛出400错误
    """
    try:
        q = KnowledgeQuery(
            source_type=req.source_type, 
            query=req.query, 
            params=req.params or {}
        )
        result = knowledge_manager.query(q)
        
        return {
            "result": result.result,
            "source_type": result.source_type.value,
            "metadata": result.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"知识查询失败: {str(e)}")

# ==================== 多模态推理API ====================

class MultimodalRequest(BaseModel):
    """
    多模态推理请求模型
    
    Args:
        None
        
    Returns:
        None
    """
    agent: str                              # 目标智能体 - 执行推理的智能体
    modality: Modality                      # 模态类型 - 输入数据的模态
    data: Any                               # 输入数据 - 实际的模态数据
    metadata: Optional[Dict[str, Any]] = None # 元数据 - 额外的数据信息

@app.post("/multimodal/infer")
@require_permission("/multimodal/infer")
async def multimodal_infer(req: MultimodalRequest, tenant: TenantContext = Depends(get_current_tenant)):
    """
    执行多模态推理
    
    Args:
        req: 多模态推理请求对象
        tenant: 租户上下文
        
    Returns:
        Dict: 推理结果
        
    Raises:
        HTTPException: 推理失败时抛出400错误
    """
    try:
        agent = system.agents.get(req.agent)
        if not agent or not isinstance(agent, MultimodalAgent):
            raise HTTPException(status_code=404, detail="智能体未找到或不是多模态智能体")
        
        input_data = MultimodalInput(
            modality=req.modality, 
            data=req.data, 
            metadata=req.metadata or {}
        )
        
        result = await agent._execute_task(input_data, task_id=str(uuid.uuid4()))
        
        return {
            "result": result,
            "modality": req.modality.value,
            "agent": req.agent
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"多模态推理失败: {str(e)}")

# ==================== 系统监控API ====================

@app.get("/monitor/summary")
@require_permission("/monitor/summary")
def monitor_summary(tenant: TenantContext = Depends(get_current_tenant)):
    """
    获取系统监控摘要
    
    Args:
        tenant: 租户上下文
        
    Returns:
        Dict: 系统监控摘要信息
        
    Raises:
        HTTPException: 获取失败时抛出500错误
    """
    try:
        summary = monitor.get_system_summary()
        return {
            "system_status": "healthy",
            "summary": summary,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取监控摘要失败: {str(e)}")

@app.get("/monitor/agents")
@require_permission("/monitor/summary")
def get_agent_status(tenant: TenantContext = Depends(get_current_tenant)):
    """
    获取所有智能体状态
    
    Args:
        tenant: 租户上下文
        
    Returns:
        Dict: 智能体状态信息
        
    Raises:
        HTTPException: 获取失败时抛出500错误
    """
    try:
        agent_statuses = {}
        for agent_id, agent in system.agents.items():
            agent_statuses[agent_id] = {
                "name": agent.config.name,
                "role": agent.config.role.value,
                "state": agent.state.value,
                "capabilities": [cap.name for cap in agent.config.capabilities],
                "metrics": agent.get_metrics()
            }
        
        return {
            "agents": agent_statuses,
            "total_agents": len(agent_statuses),
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取智能体状态失败: {str(e)}")

# ==================== 系统信息API ====================

@app.get("/system/info")
def get_system_info():
    """
    获取系统基本信息
    
    Returns:
        Dict: 系统信息
    """
    return {
        "name": "NovaMind",
        "version": "2.0.0",
        "description": "企业级多智能体系统",
        "features": [
            "多智能体协作",
            "多模态处理",
            "知识集成",
            "任务调度",
            "系统监控",
            "权限控制"
        ],
        "api_version": "v2",
        "docs_url": "/docs"
    }

# ==================== 健康检查API ====================

@app.get("/healthz")
def healthz():
    """
    健康检查端点
    
    Returns:
        Dict: 健康状态信息
    """
    try:
        # 检查核心组件状态
        system_healthy = True
        scheduler_healthy = True
        monitor_healthy = True
        
        # 这里可以添加更详细的健康检查逻辑
        
        return {
            "status": "healthy" if all([system_healthy, scheduler_healthy, monitor_healthy]) else "unhealthy",
            "components": {
                "system": "healthy" if system_healthy else "unhealthy",
                "scheduler": "healthy" if scheduler_healthy else "unhealthy",
                "monitor": "healthy" if monitor_healthy else "unhealthy"
            },
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }

# ==================== 全局异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器
    
    Args:
        request: 请求对象
        exc: 异常对象
        
    Returns:
        JSONResponse: 错误响应
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "detail": str(exc),
            "path": request.url.path
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP异常处理器
    
    Args:
        request: 请求对象
        exc: HTTP异常对象
        
    Returns:
        JSONResponse: 错误响应
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path
        }
    ) 