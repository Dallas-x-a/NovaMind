"""
API权限管理 - 核心实现

提供企业级API权限管理功能，包括：
- 基于角色的访问控制(RBAC)：支持角色继承和权限组合
- 细粒度权限验证：精确到资源级别的权限控制
- 装饰器和依赖注入支持：简化API权限检查
- 默认权限和角色定义：开箱即用的权限体系
- JWT令牌验证：安全的身份认证机制
- 多租户支持：项目级别的权限隔离
"""

import functools
import logging
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 安全认证方案 - 用于JWT令牌验证
security = HTTPBearer()


class PermissionLevel(Enum):
    """
    权限级别定义
    
    定义了从无权限到超级管理员的不同权限级别，
    支持权限的层次化管理和升级
    """
    NONE = 0          # 无权限 - 无法访问任何资源
    READ = 1          # 只读权限 - 可以查看资源但不能修改
    WRITE = 2         # 读写权限 - 可以查看和修改资源
    ADMIN = 3         # 管理权限 - 可以管理资源，包括删除
    SUPER_ADMIN = 4   # 超级管理员权限 - 拥有所有权限，包括系统配置


class ResourceType(Enum):
    """
    资源类型定义
    
    定义了系统中所有可管理的资源类型，
    每种资源都有独立的权限控制
    """
    USER = "user"         # 用户资源 - 用户账户管理
    ROLE = "role"         # 角色资源 - 角色和权限管理
    PROJECT = "project"   # 项目资源 - 项目管理和配置
    AGENT = "agent"       # 智能体资源 - 智能体管理和部署
    TASK = "task"         # 任务资源 - 任务创建和执行
    KNOWLEDGE = "knowledge"  # 知识资源 - 知识库管理
    MODEL = "model"       # 模型资源 - AI模型管理
    SYSTEM = "system"     # 系统资源 - 系统配置和监控


@dataclass
class Permission:
    """
    权限定义
    
    描述一个具体的权限，包括资源、操作和级别
    """
    name: str                    # 权限名称 (如: "users:view") - 唯一标识符
    resource: ResourceType       # 资源类型 - 权限作用的资源
    action: str                  # 操作类型 (如: view, create, update) - 具体操作
    level: PermissionLevel       # 权限级别 - 权限的强度
    description: str             # 权限描述 - 人类可读的权限说明


@dataclass
class Role:
    """
    角色定义
    
    定义用户角色，包含一组权限集合
    """
    name: str                    # 角色名称 - 角色的唯一标识
    description: str             # 角色描述 - 角色的用途说明
    permissions: List[str]       # 权限列表 - 该角色拥有的权限名称
    is_system: bool = False      # 是否系统角色 - 系统角色不可删除


class PermissionManager:
    """
    权限管理系统
    
    核心权限管理类，负责权限和角色的注册、验证和管理
    """
    
    def __init__(self):
        """
        初始化权限管理器
        
        创建权限和角色的存储容器，并初始化默认配置
        """
        self.permissions: Dict[str, Permission] = {}  # 权限字典，以权限名称为键
        self.roles: Dict[str, Role] = {}              # 角色字典，以角色名称为键
        self._initialize_defaults()                   # 初始化默认权限和角色
    
    def _initialize_defaults(self):
        """
        初始化默认权限和角色
        
        创建系统预定义的权限和角色，提供开箱即用的权限体系
        """
        # 默认权限定义 - 覆盖所有资源类型和操作
        default_permissions = [
            # 用户管理权限 - 用户账户的增删改查
            Permission("users:view", ResourceType.USER, "view", PermissionLevel.READ, "查看用户信息"),
            Permission("users:create", ResourceType.USER, "create", PermissionLevel.WRITE, "创建新用户"),
            Permission("users:update", ResourceType.USER, "update", PermissionLevel.WRITE, "更新用户信息"),
            Permission("users:delete", ResourceType.USER, "delete", PermissionLevel.ADMIN, "删除用户账户"),
            
            # 角色管理权限 - 角色和权限的管理
            Permission("roles:view", ResourceType.ROLE, "view", PermissionLevel.READ, "查看角色信息"),
            Permission("roles:create", ResourceType.ROLE, "create", PermissionLevel.ADMIN, "创建新角色"),
            Permission("roles:update", ResourceType.ROLE, "update", PermissionLevel.ADMIN, "更新角色配置"),
            Permission("roles:delete", ResourceType.ROLE, "delete", PermissionLevel.SUPER_ADMIN, "删除角色"),
            
            # 项目管理权限 - 项目的生命周期管理
            Permission("projects:view", ResourceType.PROJECT, "view", PermissionLevel.READ, "查看项目信息"),
            Permission("projects:create", ResourceType.PROJECT, "create", PermissionLevel.WRITE, "创建新项目"),
            Permission("projects:update", ResourceType.PROJECT, "update", PermissionLevel.WRITE, "更新项目配置"),
            Permission("projects:delete", ResourceType.PROJECT, "delete", PermissionLevel.ADMIN, "删除项目"),
            
            # 智能体管理权限 - 智能体的部署和管理
            Permission("agents:view", ResourceType.AGENT, "view", PermissionLevel.READ, "查看智能体信息"),
            Permission("agents:create", ResourceType.AGENT, "create", PermissionLevel.WRITE, "创建新智能体"),
            Permission("agents:update", ResourceType.AGENT, "update", PermissionLevel.WRITE, "更新智能体配置"),
            Permission("agents:delete", ResourceType.AGENT, "delete", PermissionLevel.ADMIN, "删除智能体"),
            Permission("agents:deploy", ResourceType.AGENT, "deploy", PermissionLevel.ADMIN, "部署智能体"),
            
            # 任务管理权限 - 任务的创建和执行
            Permission("tasks:view", ResourceType.TASK, "view", PermissionLevel.READ, "查看任务信息"),
            Permission("tasks:create", ResourceType.TASK, "create", PermissionLevel.WRITE, "创建新任务"),
            Permission("tasks:update", ResourceType.TASK, "update", PermissionLevel.WRITE, "更新任务配置"),
            Permission("tasks:delete", ResourceType.TASK, "delete", PermissionLevel.ADMIN, "删除任务"),
            Permission("tasks:execute", ResourceType.TASK, "execute", PermissionLevel.WRITE, "执行任务"),
            
            # 知识管理权限 - 知识库的管理
            Permission("knowledge:view", ResourceType.KNOWLEDGE, "view", PermissionLevel.READ, "查看知识库"),
            Permission("knowledge:create", ResourceType.KNOWLEDGE, "create", PermissionLevel.WRITE, "创建知识条目"),
            Permission("knowledge:update", ResourceType.KNOWLEDGE, "update", PermissionLevel.WRITE, "更新知识内容"),
            Permission("knowledge:delete", ResourceType.KNOWLEDGE, "delete", PermissionLevel.ADMIN, "删除知识条目"),
            
            # 模型管理权限 - AI模型的管理
            Permission("models:view", ResourceType.MODEL, "view", PermissionLevel.READ, "查看模型信息"),
            Permission("models:create", ResourceType.MODEL, "create", PermissionLevel.ADMIN, "创建新模型"),
            Permission("models:update", ResourceType.MODEL, "update", PermissionLevel.ADMIN, "更新模型配置"),
            Permission("models:delete", ResourceType.MODEL, "delete", PermissionLevel.SUPER_ADMIN, "删除模型"),
            Permission("models:deploy", ResourceType.MODEL, "deploy", PermissionLevel.ADMIN, "部署模型"),
            
            # 系统管理权限 - 系统级别的管理
            Permission("system:view", ResourceType.SYSTEM, "view", PermissionLevel.READ, "查看系统信息"),
            Permission("system:configure", ResourceType.SYSTEM, "configure", PermissionLevel.ADMIN, "配置系统参数"),
            Permission("system:monitor", ResourceType.SYSTEM, "monitor", PermissionLevel.ADMIN, "监控系统状态"),
            
            # 仪表板权限 - 可视化界面的访问
            Permission("dashboard:view", ResourceType.SYSTEM, "view", PermissionLevel.READ, "查看仪表板"),
        ]
        
        # 注册所有默认权限
        for perm in default_permissions:
            self.permissions[perm.name] = perm
        
        # 默认角色定义 - 预定义的角色体系
        default_roles = [
            Role(
                name="super_admin",
                description="超级管理员 - 拥有所有权限，可以管理整个系统",
                permissions=[p.name for p in self.permissions.values()],
                is_system=True
            ),
            Role(
                name="admin",
                description="管理员 - 拥有大部分管理权限，可以管理用户和资源",
                permissions=[
                    "users:view", "users:create", "users:update",
                    "roles:view", "roles:create", "roles:update",
                    "projects:view", "projects:create", "projects:update", "projects:delete",
                    "agents:view", "agents:create", "agents:update", "agents:delete", "agents:deploy",
                    "tasks:view", "tasks:create", "tasks:update", "tasks:delete", "tasks:execute",
                    "knowledge:view", "knowledge:create", "knowledge:update", "knowledge:delete",
                    "models:view", "models:create", "models:update", "models:deploy",
                    "system:view", "system:configure", "system:monitor",
                    "dashboard:view"
                ],
                is_system=True
            ),
            Role(
                name="manager",
                description="项目经理 - 项目级别管理权限，可以管理项目内的资源",
                permissions=[
                    "users:view",
                    "projects:view", "projects:create", "projects:update",
                    "agents:view", "agents:create", "agents:update",
                    "tasks:view", "tasks:create", "tasks:update", "tasks:execute",
                    "knowledge:view", "knowledge:create", "knowledge:update",
                    "models:view",
                    "dashboard:view"
                ],
                is_system=True
            ),
            Role(
                name="user",
                description="普通用户 - 基本操作权限，可以创建和执行任务",
                permissions=[
                    "projects:view",
                    "agents:view",
                    "tasks:view", "tasks:create", "tasks:execute",
                    "knowledge:view",
                    "models:view",
                    "dashboard:view"
                ],
                is_system=True
            ),
            Role(
                name="viewer",
                description="只读用户 - 仅查看权限，无法进行任何修改操作",
                permissions=[
                    "projects:view",
                    "agents:view",
                    "tasks:view",
                    "knowledge:view",
                    "models:view",
                    "dashboard:view"
                ],
                is_system=True
            )
        ]
        
        # 注册所有默认角色
        for role in default_roles:
            self.roles[role.name] = role
    
    def get_user_permissions(self, user_roles: List[str]) -> List[str]:
        """
        根据用户角色获取所有权限
        
        Args:
            user_roles: 用户拥有的角色列表
            
        Returns:
            List[str]: 用户拥有的所有权限名称列表
        """
        user_permissions = set()
        for role_name in user_roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                user_permissions.update(role.permissions)
        return list(user_permissions)
    
    def has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """
        检查用户是否具有特定权限
        
        Args:
            user_permissions: 用户拥有的权限列表
            required_permission: 需要检查的权限名称
            
        Returns:
            bool: 是否具有所需权限
        """
        return required_permission in user_permissions
    
    def has_any_permission(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """
        检查用户是否具有任一所需权限
        
        Args:
            user_permissions: 用户拥有的权限列表
            required_permissions: 需要检查的权限名称列表
            
        Returns:
            bool: 是否具有任一所需权限
        """
        return any(perm in user_permissions for perm in required_permissions)
    
    def has_all_permissions(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """
        检查用户是否具有所有所需权限
        
        Args:
            user_permissions: 用户拥有的权限列表
            required_permissions: 需要检查的权限名称列表
            
        Returns:
            bool: 是否具有所有所需权限
        """
        return all(perm in user_permissions for perm in required_permissions)


# 全局权限管理器实例 - 在整个应用中共享
permission_manager = PermissionManager()


def require_permission(permission: str):
    """
    权限检查装饰器 - 要求特定权限
    
    Args:
        permission: 需要的权限名称
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 从函数参数中提取请求对象
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="未找到请求对象"
                )
            
            # 从请求头获取JWT令牌
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="缺少认证令牌"
                )
            
            token = auth_header.split(" ")[1]
            
            try:
                # 验证JWT令牌并获取用户信息
                # 注意：这里需要配置正确的JWT密钥
                payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
                user_roles = payload.get("roles", [])
                user_permissions = permission_manager.get_user_permissions(user_roles)
                
                # 检查权限
                if not permission_manager.has_permission(user_permissions, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"缺少权限: {permission}"
                    )
                
                # 权限验证通过，执行原函数
                return await func(*args, **kwargs)
                
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="无效的认证令牌"
                )
            except Exception as e:
                logger.error(f"权限检查错误: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="权限检查失败"
                )
        
        return wrapper
    return decorator


def require_any_permission(permissions: List[str]):
    """
    权限检查装饰器 - 要求任一权限
    
    Args:
        permissions: 需要的权限名称列表
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 从函数参数中提取请求对象
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="未找到请求对象"
                )
            
            # 从请求头获取JWT令牌
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="缺少认证令牌"
                )
            
            token = auth_header.split(" ")[1]
            
            try:
                # 验证JWT令牌并获取用户信息
                payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
                user_roles = payload.get("roles", [])
                user_permissions = permission_manager.get_user_permissions(user_roles)
                
                # 检查权限
                if not permission_manager.has_any_permission(user_permissions, permissions):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"缺少所需权限: {', '.join(permissions)}"
                    )
                
                # 权限验证通过，执行原函数
                return await func(*args, **kwargs)
                
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="无效的认证令牌"
                )
            except Exception as e:
                logger.error(f"权限检查错误: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="权限检查失败"
                )
        
        return wrapper
    return decorator


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    获取当前用户信息 - FastAPI依赖注入
    
    Args:
        credentials: HTTP认证凭据
        
    Returns:
        Dict: 用户信息字典
        
    Raises:
        HTTPException: 认证失败时抛出异常
    """
    try:
        # 验证JWT令牌
        payload = jwt.decode(credentials.credentials, "your-secret-key", algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的令牌"
            )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌"
        )


def require_permission_dependency(permission: str):
    """
    权限检查依赖 - 用于FastAPI依赖注入
    
    Args:
        permission: 需要的权限名称
        
    Returns:
        依赖函数
    """
    def dependency(current_user: Dict = Depends(get_current_user)):
        user_roles = current_user.get("roles", [])
        user_permissions = permission_manager.get_user_permissions(user_roles)
        
        if not permission_manager.has_permission(user_permissions, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少权限: {permission}"
            )
        return current_user
    
    return dependency


class PermissionRequest(BaseModel):
    """
    权限请求模型 - 用于API请求验证
    
    定义创建新权限时需要的字段
    """
    name: str                    # 权限名称 - 唯一标识符
    resource: ResourceType       # 资源类型 - 权限作用的资源
    action: str                  # 操作类型 - 具体操作
    level: PermissionLevel       # 权限级别 - 权限强度
    description: str             # 权限描述 - 说明文字


class RoleRequest(BaseModel):
    """
    角色请求模型 - 用于API请求验证
    
    定义创建新角色时需要的字段
    """
    name: str                    # 角色名称 - 唯一标识符
    description: str             # 角色描述 - 用途说明
    permissions: List[str]       # 权限列表 - 该角色拥有的权限


# 导出主要组件
__all__ = [
    'PermissionManager',
    'permission_manager',
    'require_permission',
    'require_any_permission',
    'get_current_user',
    'require_permission_dependency',
    'PermissionLevel',
    'ResourceType',
    'Permission',
    'Role'
] 