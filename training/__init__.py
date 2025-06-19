"""
NovaMind训练模块

提供企业级AI模型训练功能，包括：
- 基础训练框架
- LLM专门训练器
- 实时监控系统
- 参数调优
- LoRA训练支持
"""

from .trainer import (
    BaseTrainer,
    TrainingConfig,
    TrainingStatus,
    ModelType,
    TrainingMetrics,
    LoRAConfig,
    TrainingManager,
    training_manager
)

from .llm_trainer import (
    LLMTrainingConfig,
    LLMTrainer,
    RLHFTrainer,
    TextDataset,
    ConversationDataset
)

from .monitor import (
    TrainingMonitor,
    MetricsCallback,
    training_monitor
)

__all__ = [
    # 基础训练框架
    'BaseTrainer',
    'TrainingConfig', 
    'TrainingStatus',
    'ModelType',
    'TrainingMetrics',
    'LoRAConfig',
    'TrainingManager',
    'training_manager',
    
    # LLM训练器
    'LLMTrainingConfig',
    'LLMTrainer',
    'RLHFTrainer',
    'TextDataset',
    'ConversationDataset',
    
    # 监控系统
    'TrainingMonitor',
    'MetricsCallback',
    'training_monitor'
] 