# NovaMind: 下一代多智能体与大模型应用框架

> **NovaMind** 是企业级多智能体与大模型应用开发框架，专注于高效、可扩展、模块化的智能体系统与AI工具链，提供完整的权限管理、分布式架构、企业级安全和专业训练框架，助力学术研究与产业落地。

---

## 🌟 框架设计理念

- **模块化与可扩展性**：核心功能高度解耦，支持灵活组合与自定义扩展
- **多大模型原生支持**：内置OpenAI、Claude、Qwen、ERNIE、Llama、DeepSeek、MiniMax、GLM、Yi等主流大模型接口
- **丰富的智能体与工具生态**：涵盖RAG、信息抽取、代码生成、联网搜索、知识图谱等多场景
- **🚀 企业级训练框架**：专业训练系统，支持实时监控、智能调优、LoRA训练
- **企业级权限管理**：完整的RBAC权限系统，支持细粒度权限控制
- **分布式Agent架构**：基于gRPC的分布式Agent系统，支持大规模部署
- **工程级最佳实践**：官方示例丰富，文档详尽，易于集成与二次开发

---

## 🚀 企业级训练框架

NovaMind提供了一套完整的**训练框架**，专门针对大语言模型进行实时监控、参数调整和LoRA训练，具有以下独特优势：

### 🎯 核心优势

- **🔄 实时训练监控**：Web界面实时查看训练状态、损失曲线、系统资源
- **🧠 智能参数调优**：基于性能自动调整超参数，支持早停和学习率调度
- **🎨 多模态训练支持**：支持文本、图像、音频等多种模态的模型训练
- **⚡ 分布式训练**：支持多GPU/多节点训练，自动负载均衡
- **📊 模型版本管理**：完整的模型生命周期管理，支持版本回滚
- **🔬 实验管理**：A/B测试和实验对比，支持实验追踪
- **🏭 生产就绪**：直接部署到生产环境，支持模型服务化

### 🛠️ 训练功能特性

#### 1. 多种训练模式
```python
# 基础语言模型训练
from novamind.training import get_training_config, LLMTrainer

config = get_training_config(
    model_preset="llama-7b",
    training_preset="standard",
    dataset_path="./data/text_data.json",
    training_name="basic_lm"
)

# LoRA微调训练
from novamind.training import get_lora_config

config = get_lora_config(
    model_preset="llama-7b",
    dataset_path="./data/instruction_data.json",
    training_name="lora_finetune",
    lora_preset="efficient"
)

# 指令微调训练
from novamind.training import get_instruction_config

config = get_instruction_config(
    model_preset="qwen-7b",
    dataset_path="./data/instruction_data.json",
    training_name="instruction_tune"
)

# 对话模型训练
from novamind.training import get_conversation_config

config = get_conversation_config(
    model_preset="chatglm3-6b",
    dataset_path="./data/conversation_data.json",
    training_name="conversation_model",
    chat_template="chatml"
)
```

#### 2. 实时监控系统
```python
# 启动监控服务器
from novamind.training import training_monitor

training_monitor.start_background()

# 添加监控回调
from novamind.training import MetricsCallback

trainer.add_callback(MetricsCallback("training_id"))

# 访问监控界面: http://localhost:8080
```

#### 3. 智能参数管理
```python
# 预设配置系统
from novamind.training import list_available_presets

presets = list_available_presets()
print(presets)
# {
#   'model_presets': ['gpt2-small', 'llama-7b', 'qwen-7b', ...],
#   'training_presets': ['quick', 'standard', 'high_quality', ...],
#   'lora_presets': ['standard', 'efficient', 'aggressive']
# }

# 自定义配置
from novamind.training import TrainingConfig, LoRAConfig

config = TrainingConfig(
    model_name="custom_model",
    model_type="llm",
    model_path="microsoft/DialoGPT-small",
    dataset_path="./data/custom_data.json",
    batch_size=8,
    learning_rate=3e-5,
    num_epochs=10,
    use_lora=True,
    lora_config=LoRAConfig(r=16, alpha=32)
)
```

#### 4. 高级训练功能
```python
# 分布式训练
config.distributed = True
config.num_gpus = 4

# 混合精度训练
config.mixed_precision = True

# 梯度检查点
config.gradient_checkpointing = True

# Flash Attention
config.flash_attention = True

# DeepSpeed支持
config.deepspeed = True
```

### 📊 监控与可视化

#### 实时监控仪表板
- **训练状态**：实时显示训练进度、当前轮数、步数
- **损失曲线**：动态更新的损失和准确率曲线
- **系统资源**：CPU、内存、GPU使用率监控
- **训练指标**：学习率、梯度范数、困惑度等关键指标

#### 训练报告生成
```python
# 生成训练报告
from novamind.training import training_monitor

report = training_monitor.generate_metrics_report("training_id")
print(report)
# {
#   'training_id': 'training_001',
#   'total_steps': 1000,
#   'final_loss': 0.1234,
#   'best_loss': 0.0987,
#   'avg_accuracy': 0.8567,
#   'total_training_time': '2:30:15'
# }

# 创建可视化图表
charts = training_monitor.create_visualization("training_id")
```

### 🎨 支持的模型架构

| 模型类型 | 支持模型 | 特点 |
|---------|---------|------|
| **GPT系列** | GPT-2, DialoGPT | 因果语言模型，适合文本生成 |
| **LLaMA系列** | LLaMA-7B, LLaMA-13B | 高性能开源模型，支持多语言 |
| **Qwen系列** | Qwen-7B, Qwen-14B | 阿里云开源模型，中文表现优秀 |
| **ChatGLM系列** | ChatGLM3-6B | 清华开源对话模型，中文优化 |
| **Baichuan系列** | Baichuan2-7B | 百川开源模型，通用性强 |
| **Yi系列** | Yi-6B, Yi-34B | 01.AI开源模型，性能优秀 |

### 🔬 实验管理

#### 实验对比
```python
# 创建多个实验配置
experiments = {
    "baseline": get_training_config("llama-7b", "standard", "data.json", "baseline"),
    "lora_8": get_lora_config("llama-7b", "data.json", "lora_8", "efficient"),
    "lora_16": get_lora_config("llama-7b", "data.json", "lora_16", "standard"),
    "instruction": get_instruction_config("llama-7b", "data.json", "instruction")
}

# 并行运行实验
from novamind.training import training_manager

for name, config in experiments.items():
    trainer = LLMTrainer(config)
    training_manager.start_training(name, trainer)
```

#### 实验结果分析
```python
# 获取所有实验结果
results = {}
for exp_name in experiments.keys():
    report = training_monitor.generate_metrics_report(exp_name)
    results[exp_name] = report

# 对比分析
for name, result in results.items():
    print(f"{name}: 最终损失={result['final_loss']:.4f}, 准确率={result['best_accuracy']:.4f}")
```

### 🔧 快速开始

#### 1. 安装依赖
```bash
# 安装训练框架
pip install torch transformers datasets wandb plotly fastapi uvicorn

# 或使用安装脚本
python install_training.py
```

#### 2. 创建训练配置
```python
from novamind.training import get_training_config, LLMTrainer

# 使用预设配置
config = get_training_config(
    model_preset="dialo-gpt-small",
    training_preset="quick",
    dataset_path="./data/sample_data.json",
    training_name="my_first_training"
)

# 创建训练器
trainer = LLMTrainer(config)
```

#### 3. 启动训练
```python
from novamind.training import training_manager, training_monitor

# 启动监控
training_monitor.start_background()

# 启动训练
training_id = training_manager.start_training("my_training", trainer)

# 查看训练状态
status = training_manager.get_training_status(training_id)
print(f"训练状态: {status['status']}")
```

#### 4. 模型推理
```python
# 文本生成
generated_text = trainer.generate_text(
    "人工智能的未来发展",
    max_length=100,
    temperature=0.7
)
print(generated_text)

# 评估生成质量
test_prompts = ["什么是机器学习？", "深度学习有什么优势？"]
metrics = trainer.evaluate_generation_quality(test_prompts)
print(metrics)
```

---

## 🏗️ 企业级架构特性

### 分布式Agent系统
- **gRPC通信协议**：高性能的Agent间通信
- **自动注册与发现**：Agent自动注册和心跳检测
- **负载均衡**：基于能力的智能任务分配
- **故障恢复**：自动检测和处理失效Agent

### 权限管理系统
- **RBAC模型**：基于角色的访问控制
- **细粒度权限**：支持资源级别的权限控制
- **多级权限**：READ、WRITE、ADMIN、SUPER_ADMIN五个级别
- **前端权限控制**：React组件级和路由级权限守卫

### 安全特性
- **JWT认证**：安全的用户认证机制
- **API权限验证**：装饰器和中间件双重保护
- **HTTPS支持**：生产环境安全传输
- **审计日志**：完整的操作日志记录

---

## 🏁 快速上手示例

### 基础Agent使用
```python
from novamind.core.agent import Agent
from novamind.core.environment import Environment
from novamind.tools.web_search import WebSearchTool

agent = Agent(
    name="researcher",
    model="gpt-4o",
    tools=[WebSearchTool()]
)
response = agent.run("简述量子计算的最新进展")
print(response)
```

### 分布式Agent部署
```python
# 启动gRPC服务器
from novamind.core.grpc_service import gRPCServer

server = gRPCServer(host='0.0.0.0', port=50051)
await server.start()

# Agent客户端连接
from novamind.core.grpc_service import gRPCClient

client = gRPCClient('localhost:50051')
await client.connect()

# 注册Agent
success, message = await client.register_agent(
    agent_id="agent-001",
    name="Text Processing Agent",
    capabilities=["text", "nlp"],
    endpoint="http://localhost:8001"
)
```

### 权限管理使用
```python
# API权限装饰器
from novamind.core.api_permissions import require_permission

@require_permission("users:view")
async def get_users(request: Request):
    return {"users": user_list}

# 依赖注入权限检查
from novamind.core.api_permissions import require_permission_dependency

@router.post("/users")
async def create_user(
    user_data: UserCreate,
    current_user = Depends(require_permission_dependency("users:create"))
):
    return create_user_in_db(user_data)
```

### 🚀 训练框架快速示例
```python
# 一行代码启动训练
from novamind.training import get_training_config, LLMTrainer, training_manager

config = get_training_config(
    model_preset="llama-7b",
    training_preset="standard", 
    dataset_path="./data/training_data.json",
    training_name="my_training"
)

trainer = LLMTrainer(config)
training_id = training_manager.start_training("my_training", trainer)

# 访问监控界面: http://localhost:8080
```

---

## 🧩 主要模块与目录结构

```
novamind/
├── core/                    # 核心模块
│   ├── agent.py            # Agent核心实现
│   ├── environment.py      # 环境管理
│   ├── models.py           # 模型管理
│   ├── tools.py            # 工具管理
│   ├── memory.py           # 记忆系统
│   ├── scheduler.py        # 任务调度器
│   ├── monitor.py          # 系统监控
│   ├── security.py         # 安全模块
│   ├── distributed.py      # 分布式支持
│   ├── grpc_service.py     # gRPC服务实现
│   ├── api_permissions.py  # API权限管理
│   ├── multimodal.py       # 多模态支持
│   └── knowledge.py        # 知识管理
├── training/               # 🆕 企业级训练框架
│   ├── trainer.py          # 基础训练器
│   ├── llm_trainer.py      # LLM专门训练器
│   ├── monitor.py          # 实时监控系统
│   ├── config.py           # 配置管理
│   ├── quick_start.py      # 快速开始指南
│   ├── example.py          # 使用示例
│   ├── advantages_demo.py  # 优势演示
│   └── __init__.py         # 模块导出
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── components/     # React组件
│   │   ├── pages/          # 页面组件
│   │   ├── contexts/       # React上下文
│   │   └── App.js          # 主应用
│   └── package.json
├── api.py                  # FastAPI主服务器
├── mcp/                    # 通用大模型能力包
├── downloads/              # 模型下载脚本
├── examples/               # 官方示例
├── models/                 # 大模型API/SDK集成
├── tools/                  # 智能体可用工具
├── datagen/                # 数据生成与增强
├── storages/               # 存储与知识库
├── train/                  # 训练与微调脚本
├── tests/                  # 单元测试
├── requirements.txt        # 基础依赖
├── requirements_training.txt # 训练框架依赖
├── install_training.py     # 训练框架安装脚本
└── README.md               # 项目文档
```

---

## 🔐 权限管理系统详解

### 权限级别
- **NONE (0)**: 无权限
- **READ (1)**: 只读权限
- **WRITE (2)**: 读写权限
- **ADMIN (3)**: 管理权限
- **SUPER_ADMIN (4)**: 超级管理员权限

### 资源类型
- **USER**: 用户管理
- **ROLE**: 角色管理
- **PROJECT**: 项目管理
- **AGENT**: Agent管理
- **TASK**: 任务管理
- **KNOWLEDGE**: 知识管理
- **MODEL**: 模型管理
- **SYSTEM**: 系统管理

### 系统角色
- **Super Admin**: 拥有所有权限，用于系统初始化
- **Admin**: 拥有大部分管理权限，用于日常管理
- **Manager**: 项目级别管理权限，用于项目管理
- **User**: 基本操作权限，用于日常使用
- **Viewer**: 只读权限，用于监控和查看

### 权限使用示例

#### 前端权限控制
```jsx
// 单权限检查
<PermissionGuard permission="users:view">
  <UserList />
</PermissionGuard>

// 多权限检查 (任一)
<PermissionGuard permissions={["users:view", "users:create"]} any={true}>
  <UserManagement />
</PermissionGuard>

// 多权限检查 (全部)
<PermissionGuard permissions={["users:view", "users:delete"]} all={true}>
  <UserAdmin />
</PermissionGuard>
```

#### API权限控制
```python
# 装饰器方式
@require_permission("users:view")
async def get_users(request: Request):
    return {"users": user_list}

# 依赖注入方式
@router.post("/users")
async def create_user(
    user_data: UserCreate,
    current_user = Depends(require_permission_dependency("users:create"))
):
    return create_user_in_db(user_data)
```

---

## 🚀 部署与运维

### 环境要求
- **Python**: 3.8+
- **CUDA**: 11.8+ (GPU训练)
- **内存**: 16GB+ (推荐32GB+)
- **存储**: 100GB+ SSD

### 安装指南
```bash
# 克隆仓库
git clone https://github.com/your-org/novamind.git
cd novamind

# 安装基础依赖
pip install -r requirements.txt

# 安装训练框架
python install_training.py

# 或手动安装训练依赖
pip install -r requirements_training.txt

# 初始化配置
python -m novamind.core.config init
```

### 快速启动
```bash
# 启动API服务器
python -m novamind.api

# 启动训练监控
python -c "from novamind.training.monitor import training_monitor; training_monitor.start()"

# 启动前端应用
cd frontend && npm start
```


### 训练框架部署
```bash
# 开发环境
python -m novamind.training.quick_start

# 生产环境
python -m novamind.training.example

# 优势演示
python -m novamind.training.advantages_demo
```

---

## 📚 文档与资源

### 官方文档
- [快速开始指南](./docs/quickstart.md)
- [训练框架文档](./training/quick_start.py)
- [API参考文档](./docs/api.md)
- [部署指南](./docs/deployment.md)

### 示例项目
- [基础Agent示例](./examples/basic_agent.py)
- [分布式训练示例](./training/example.py)
- [权限管理示例](./examples/permissions.py)
- [训练框架示例](./training/quick_start.py)
- [优势演示](./training/advantages_demo.py)

### 社区资源
- [GitHub Issues](https://github.com/your-org/novamind/issues)
- [Discord社区](https://discord.gg/novamind)

---

## 🆚 技术对比

### 与LangChain对比

| 特性 | NovaMind | LangChain |
|------|----------|-----------|
| **实时监控** | ✅ Web界面实时监控，WebSocket推送 | ❌ 基础日志输出 |
| **参数调优** | ✅ 智能预设系统，自动调优 | ❌ 手动配置，无自动调优 |
| **多模态支持** | ✅ 统一接口，多模态原生支持 | ❌ 主要针对文本，多模态支持有限 |
| **分布式训练** | ✅ 原生分布式支持，自动负载均衡 | ❌ 需要额外配置，支持有限 |
| **版本管理** | ✅ 完整版本管理，自动检查点 | ❌ 基础保存功能 |
| **实验管理** | ✅ A/B测试，实验对比分析 | ❌ 无内置实验管理 |
| **生产就绪** | ✅ 企业级特性，容器化部署 | ❌ 主要面向研究，生产支持有限 |
| **权限管理** | ✅ 完整RBAC权限系统 | ❌ 无内置权限管理 |
| **分布式Agent** | ✅ gRPC分布式架构 | ❌ 无分布式Agent支持 |

### 与Hugging Face对比

| 特性 | NovaMind | Hugging Face |
|------|----------|--------------|
| **训练框架** | ✅ 企业级训练框架，实时监控 | ❌ 基础训练脚本 |
| **权限管理** | ✅ 完整RBAC权限系统 | ❌ 无权限管理 |
| **分布式Agent** | ✅ gRPC分布式架构 | ❌ 无Agent系统 |
| **多模态支持** | ✅ 统一多模态接口 | ✅ 支持多模态 |
| **模型支持** | ✅ 主流模型全覆盖 | ✅ 模型生态丰富 |
| **生产部署** | ✅ 企业级部署方案 | ❌ 主要面向研究 |

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看我们的[贡献指南](./CONTRIBUTING.md)了解详情。

### 贡献类型
- 🐛 Bug报告和修复
- ✨ 新功能开发
- 📖 文档改进
- 🧪 测试用例
- 💡 想法和建议

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/your-org/novamind.git
cd novamind

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements_training.txt
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black .
isort .
```

---

## 📄 许可证

本项目采用 [MIT License](./LICENSE) 许可证。

---

## 🙏 致谢

感谢所有为NovaMind项目做出贡献的开发者和研究人员！

特别感谢以下开源项目：
- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [React](https://github.com/facebook/react)

---

**NovaMind** - 让AI开发更简单、更高效、更强大！ 🚀



---

*最后更新时间: 2024年12月*
