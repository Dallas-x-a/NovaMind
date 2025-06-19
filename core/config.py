"""
NovaMind配置管理系统

提供统一的配置管理、验证和环境变量支持。
支持分层配置和动态更新。

主要功能：
- 分层配置：模型、内存、工具、智能体、系统等分层配置
- 配置验证：基于Pydantic的自动配置验证
- 环境变量：支持环境变量覆盖配置
- 动态更新：运行时配置更新和合并
- 配置导出：支持多种格式的配置导出
- 配置管理：统一的配置加载和保存
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """
    模型配置设置
    
    定义大语言模型的各种参数和配置选项
    """
    
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)      # 温度参数 - 控制生成随机性
    max_tokens: Optional[int] = Field(default=None, gt=0)        # 最大token数 - 输出长度限制
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)            # top_p参数 - 核采样参数
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)  # 频率惩罚 - 减少重复内容
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)   # 存在惩罚 - 鼓励新话题
    stop: Optional[list[str]] = None                             # 停止词列表 - 生成停止条件


class MemoryConfig(BaseModel):
    """
    内存配置设置
    
    定义智能体内存系统的配置参数
    """
    
    max_tokens: int = Field(default=2000, gt=0)     # 最大token数 - 内存容量限制
    max_messages: int = Field(default=50, gt=0)     # 最大消息数 - 对话历史限制
    memory_type: str = Field(default="conversation") # 内存类型 - 记忆类型选择
    persist: bool = Field(default=False)            # 是否持久化 - 内存持久化选项


class ToolConfig(BaseModel):
    """
    工具配置设置
    
    定义工具执行和管理的配置参数
    """
    
    max_iterations: int = Field(default=5, gt=0)     # 最大迭代次数 - 工具链执行限制
    timeout: int = Field(default=30, gt=0)           # 超时时间(秒) - 工具执行超时
    retry_attempts: int = Field(default=3, gt=0)     # 重试次数 - 失败重试机制
    parallel_execution: bool = Field(default=False)  # 是否并行执行 - 并行处理选项


class AgentConfig(BaseModel):
    """
    智能体配置设置
    
    定义智能体的行为和控制参数
    """
    
    name: str                                        # 智能体名称 - 唯一标识符
    role: str = Field(default="general")            # 智能体角色 - 功能角色定义
    max_concurrent_tasks: int = Field(default=5, gt=0)  # 最大并发任务数 - 并发控制
    timeout: int = Field(default=300, gt=0)         # 任务超时时间(秒) - 任务执行超时
    retry_attempts: int = Field(default=3, gt=0)    # 重试次数 - 任务失败重试
    memory_size: int = Field(default=1000, gt=0)    # 内存大小 - 智能体内存容量
    enable_monitoring: bool = Field(default=True)   # 是否启用监控 - 性能监控开关
    security_level: str = Field(default="standard") # 安全级别 - 安全控制级别


class SystemConfig(BaseModel):
    """
    系统配置设置
    
    定义整个系统的运行参数和限制
    """
    
    max_agents: int = Field(default=100, gt=0)      # 最大智能体数量 - 系统容量限制
    max_tasks: int = Field(default=1000, gt=0)      # 最大任务数量 - 任务队列限制
    task_timeout: int = Field(default=3600, gt=0)   # 任务超时时间(秒) - 系统级超时
    enable_logging: bool = Field(default=True)      # 是否启用日志 - 日志系统开关
    log_level: str = Field(default="INFO")          # 日志级别 - 日志详细程度
    enable_metrics: bool = Field(default=True)      # 是否启用指标收集 - 监控指标开关


class SecurityConfig(BaseModel):
    """
    安全配置设置
    
    定义系统的安全相关配置
    """
    
    enable_auth: bool = Field(default=True)         # 是否启用认证 - 身份认证开关
    jwt_secret: Optional[str] = None                # JWT密钥 - 令牌签名密钥
    jwt_expiry: int = Field(default=3600, gt=0)     # JWT过期时间(秒) - 令牌有效期
    enable_rbac: bool = Field(default=True)         # 是否启用RBAC - 权限控制开关
    cors_origins: list[str] = Field(default_factory=list)  # CORS来源 - 跨域访问控制


class DatabaseConfig(BaseModel):
    """
    数据库配置设置
    
    定义数据库连接和管理的配置参数
    """
    
    type: str = Field(default="sqlite")             # 数据库类型 - 数据库选择
    url: Optional[str] = None                       # 数据库URL - 连接字符串
    host: Optional[str] = None                      # 数据库主机 - 服务器地址
    port: Optional[int] = None                      # 数据库端口 - 服务端口
    username: Optional[str] = None                  # 用户名 - 数据库用户
    password: Optional[str] = None                  # 密码 - 数据库密码
    database: Optional[str] = None                  # 数据库名 - 数据库名称
    pool_size: int = Field(default=10, gt=0)        # 连接池大小 - 连接池容量
    max_overflow: int = Field(default=20, gt=0)     # 最大溢出连接数 - 额外连接数


class Config(BaseModel):
    """
    NovaMind主配置类
    
    整合所有子配置，提供统一的配置管理接口
    """
    
    # 子配置
    model: ModelConfig = Field(default_factory=ModelConfig)      # 模型配置 - 大语言模型参数
    memory: MemoryConfig = Field(default_factory=MemoryConfig)   # 内存配置 - 记忆系统参数
    tools: ToolConfig = Field(default_factory=ToolConfig)        # 工具配置 - 工具执行参数
    agent: AgentConfig = Field(default_factory=AgentConfig)      # 智能体配置 - 智能体行为参数
    system: SystemConfig = Field(default_factory=SystemConfig)   # 系统配置 - 系统运行参数
    security: SecurityConfig = Field(default_factory=SecurityConfig)  # 安全配置 - 安全控制参数
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)  # 数据库配置 - 数据存储参数
    
    # 通用设置
    debug: bool = Field(default=False)              # 调试模式 - 开发调试开关
    log_level: str = Field(default="INFO")          # 日志级别 - 系统日志级别
    api_key: Optional[str] = None                   # API密钥 - 外部API认证
    
    # 自定义设置
    custom: Dict[str, Any] = Field(default_factory=dict)  # 自定义配置 - 扩展配置项
    
    class Config:
        env_prefix = "NOVAMIND_"                    # 环境变量前缀 - 环境变量标识
        case_sensitive = False                      # 大小写不敏感 - 配置键大小写处理
        
    def update(self, **kwargs: Any) -> None:
        """
        使用新值更新配置
        
        Args:
            **kwargs: 要更新的配置项
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), BaseModel):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                self.custom[key] = value
                
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典
        
        Returns:
            Dict[str, Any]: 配置的字典表示
        """
        return self.model_dump()
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        从字典创建配置
        
        Args:
            data: 配置数据字典
            
        Returns:
            Config: 配置实例
        """
        return cls(**data)
        
    def validate(self) -> bool:
        """
        验证配置设置
        
        Returns:
            bool: 配置是否有效
            
        Raises:
            ValueError: 当配置无效时
        """
        try:
            self.model_validate(self.model_dump())
            return True
        except Exception as e:
            raise ValueError(f"无效的配置: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        支持嵌套键访问，如 "model.temperature"
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        # 支持嵌套键，如 "model.temperature"
        keys = key.split(".")
        value = self
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        支持嵌套键设置，如 "model.temperature"
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
            
        Raises:
            KeyError: 当配置键不存在时
        """
        # 支持嵌套键，如 "model.temperature"
        keys = key.split(".")
        target = self
        
        for k in keys[:-1]:
            if hasattr(target, k):
                target = getattr(target, k)
            else:
                raise KeyError(f"配置键不存在: {key}")
                
        setattr(target, keys[-1], value)
    
    def merge(self, other_config: "Config") -> None:
        """
        合并另一个配置
        
        将另一个配置的内容合并到当前配置中
        
        Args:
            other_config: 要合并的配置
        """
        other_dict = other_config.to_dict()
        self.update(**other_dict)
    
    def export_env_vars(self) -> Dict[str, str]:
        """
        导出环境变量格式的配置
        
        Returns:
            Dict[str, str]: 环境变量字典
        """
        env_vars = {}
        
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> None:
            """
            递归展平字典为环境变量格式
            
            Args:
                d: 要展平的字典
                prefix: 键前缀
            """
            for key, value in d.items():
                env_key = f"NOVAMIND_{prefix}{key.upper()}" if prefix else f"NOVAMIND_{key.upper()}"
                if isinstance(value, dict):
                    flatten_dict(value, f"{key.upper()}_")
                else:
                    env_vars[env_key] = str(value)
        
        flatten_dict(self.to_dict())
        return env_vars


# 默认配置实例
default_config = Config()

# 配置管理器
class ConfigManager:
    """
    配置管理器 - 统一管理配置的加载和保存
    
    提供配置文件的读取、写入和环境变量加载功能
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self._config: Optional[Config] = None
        
    def load_from_file(self, file_path: str) -> Config:
        """
        从文件加载配置
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            Config: 加载的配置实例
            
        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件格式无效时
        """
        import json
        import yaml
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                data = json.load(f)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {path.suffix}")
                
        self._config = Config.from_dict(data)
        return self._config
        
    def save_to_file(self, file_path: str, format: str = "json") -> None:
        """
        保存配置到文件
        
        Args:
            file_path: 保存文件路径
            format: 文件格式 (json/yaml)
            
        Raises:
            ValueError: 当格式不支持时
        """
        import json
        import yaml
        
        if self._config is None:
            raise ValueError("没有配置可保存")
            
        data = self._config.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() in ['yml', 'yaml']:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"不支持的文件格式: {format}")
        
    def load_from_env(self) -> Config:
        """
        从环境变量加载配置
        
        Returns:
            Config: 从环境变量创建的配置实例
        """
        import os
        
        env_vars = {}
        for key, value in os.environ.items():
            if key.startswith("NOVAMIND_"):
                # 移除前缀并转换为小写
                config_key = key[9:].lower()
                env_vars[config_key] = value
                
        self._config = Config.from_dict(env_vars)
        return self._config
        
    def get_config(self) -> Config:
        """
        获取当前配置
        
        Returns:
            Config: 当前配置实例
        """
        if self._config is None:
            self._config = Config()
        return self._config
        
    def update_config(self, **kwargs: Any) -> None:
        """
        更新当前配置
        
        Args:
            **kwargs: 要更新的配置项
        """
        if self._config is None:
            self._config = Config()
        self._config.update(**kwargs) 