"""
NovaMind训练配置系统

提供完整的训练配置管理，包括：
- 不同模型架构的预设配置
- 训练场景配置
- 环境配置
- 优化配置
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from .trainer import TrainingConfig, LoRAConfig, ModelType


@dataclass
class ModelPreset:
    """模型预设配置"""
    name: str
    model_path: str
    tokenizer_path: Optional[str] = None
    model_type: ModelType = ModelType.LLM
    task_type: str = "causal_lm"
    max_length: int = 2048
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True


@dataclass
class TrainingPreset:
    """训练预设配置"""
    name: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    gradient_accumulation_steps: int
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 5


class TrainingScenarios:
    """训练场景配置"""
    
    # 模型预设
    MODEL_PRESETS = {
        # 小型模型
        "gpt2-small": ModelPreset(
            name="GPT-2 Small",
            model_path="gpt2",
            max_length=1024
        ),
        "dialo-gpt-small": ModelPreset(
            name="DialoGPT Small",
            model_path="microsoft/DialoGPT-small",
            max_length=1024
        ),
        "llama-7b": ModelPreset(
            name="LLaMA 7B",
            model_path="meta-llama/Llama-2-7b-hf",
            max_length=4096,
            use_flash_attention=True
        ),
        "llama-13b": ModelPreset(
            name="LLaMA 13B",
            model_path="meta-llama/Llama-2-13b-hf",
            max_length=4096,
            use_flash_attention=True
        ),
        "qwen-7b": ModelPreset(
            name="Qwen 7B",
            model_path="Qwen/Qwen-7B",
            max_length=32768,
            use_flash_attention=True
        ),
        "chatglm3-6b": ModelPreset(
            name="ChatGLM3 6B",
            model_path="THUDM/chatglm3-6b",
            max_length=8192
        ),
        "baichuan2-7b": ModelPreset(
            name="Baichuan2 7B",
            model_path="baichuan-inc/Baichuan2-7B-Base",
            max_length=4096
        ),
        "yi-6b": ModelPreset(
            name="Yi 6B",
            model_path="01-ai/Yi-6B",
            max_length=4096
        )
    }
    
    # 训练预设
    TRAINING_PRESETS = {
        # 快速实验
        "quick": TrainingPreset(
            name="快速实验",
            batch_size=4,
            learning_rate=5e-5,
            num_epochs=2,
            warmup_steps=50,
            gradient_accumulation_steps=1
        ),
        # 标准训练
        "standard": TrainingPreset(
            name="标准训练",
            batch_size=8,
            learning_rate=3e-5,
            num_epochs=5,
            warmup_steps=100,
            gradient_accumulation_steps=2
        ),
        # 高质量训练
        "high_quality": TrainingPreset(
            name="高质量训练",
            batch_size=16,
            learning_rate=1e-5,
            num_epochs=10,
            warmup_steps=200,
            gradient_accumulation_steps=4
        ),
        # LoRA微调
        "lora": TrainingPreset(
            name="LoRA微调",
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=3,
            warmup_steps=50,
            gradient_accumulation_steps=2
        ),
        # 指令微调
        "instruction": TrainingPreset(
            name="指令微调",
            batch_size=4,
            learning_rate=2e-5,
            num_epochs=5,
            warmup_steps=100,
            gradient_accumulation_steps=2
        ),
        # 对话训练
        "conversation": TrainingPreset(
            name="对话训练",
            batch_size=2,
            learning_rate=1e-5,
            num_epochs=8,
            warmup_steps=150,
            gradient_accumulation_steps=4
        )
    }
    
    # LoRA配置预设
    LORA_PRESETS = {
        "standard": LoRAConfig(
            r=16,
            alpha=32,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        ),
        "efficient": LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        ),
        "aggressive": LoRAConfig(
            r=32,
            alpha=64,
            dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
    }


class EnvironmentConfig(BaseModel):
    """环境配置"""
    
    # 基础路径
    base_dir: str = Field(default="./novamind", description="基础目录")
    data_dir: str = Field(default="./data", description="数据目录")
    output_dir: str = Field(default="./outputs", description="输出目录")
    cache_dir: str = Field(default="./cache", description="缓存目录")
    
    # 模型路径
    model_cache_dir: str = Field(default="./models", description="模型缓存目录")
    checkpoint_dir: str = Field(default="./checkpoints", description="检查点目录")
    
    # 日志配置
    log_dir: str = Field(default="./logs", description="日志目录")
    log_level: str = Field(default="INFO", description="日志级别")
    
    # 监控配置
    monitor_port: int = Field(default=8080, description="监控端口")
    enable_wandb: bool = Field(default=True, description="启用WandB")
    wandb_project: str = Field(default="novamind-training", description="WandB项目名")
    
    # 分布式配置
    distributed_backend: str = Field(default="nccl", description="分布式后端")
    master_addr: str = Field(default="localhost", description="主节点地址")
    master_port: str = Field(default="29500", description="主节点端口")
    
    # 性能配置
    num_workers: int = Field(default=4, description="数据加载器工作进程数")
    pin_memory: bool = Field(default=True, description="固定内存")
    prefetch_factor: int = Field(default=2, description="预取因子")
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.base_dir,
            self.data_dir,
            self.output_dir,
            self.cache_dir,
            self.model_cache_dir,
            self.checkpoint_dir,
            self.log_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """获取模型路径"""
        return str(Path(self.model_cache_dir) / model_name)
    
    def get_output_path(self, training_name: str) -> str:
        """获取输出路径"""
        return str(Path(self.output_dir) / training_name)
    
    def get_checkpoint_path(self, training_name: str) -> str:
        """获取检查点路径"""
        return str(Path(self.checkpoint_dir) / training_name)


class TrainingConfigBuilder:
    """训练配置构建器"""
    
    def __init__(self, env_config: EnvironmentConfig):
        """
        初始化配置构建器
        
        Args:
            env_config: 环境配置
        """
        self.env_config = env_config
        self.scenarios = TrainingScenarios()
    
    def build_config(
        self,
        model_preset: str,
        training_preset: str,
        dataset_path: str,
        training_name: str,
        **kwargs
    ) -> TrainingConfig:
        """
        构建训练配置
        
        Args:
            model_preset: 模型预设名称
            training_preset: 训练预设名称
            dataset_path: 数据集路径
            training_name: 训练名称
            **kwargs: 额外配置参数
            
        Returns:
            TrainingConfig: 训练配置
        """
        # 获取预设配置
        model_preset_config = self.scenarios.MODEL_PRESETS[model_preset]
        training_preset_config = self.scenarios.TRAINING_PRESETS[training_preset]
        
        # 构建基础配置
        config_dict = {
            "model_name": training_name,
            "model_type": model_preset_config.model_type,
            "model_path": model_preset_config.model_path,
            "tokenizer_path": model_preset_config.tokenizer_path,
            "dataset_path": dataset_path,
            "output_dir": self.env_config.get_output_path(training_name),
            "max_length": model_preset_config.max_length,
            "flash_attention": model_preset_config.use_flash_attention,
            "gradient_checkpointing": model_preset_config.gradient_checkpointing,
            "task_type": model_preset_config.task_type,
            
            # 训练参数
            "batch_size": training_preset_config.batch_size,
            "learning_rate": training_preset_config.learning_rate,
            "num_epochs": training_preset_config.num_epochs,
            "warmup_steps": training_preset_config.warmup_steps,
            "gradient_accumulation_steps": training_preset_config.gradient_accumulation_steps,
            "optimizer": training_preset_config.optimizer,
            "scheduler": training_preset_config.scheduler,
            "weight_decay": training_preset_config.weight_decay,
            "gradient_clipping": training_preset_config.gradient_clipping,
            "mixed_precision": training_preset_config.mixed_precision,
            "early_stopping_patience": training_preset_config.early_stopping_patience,
            
            # 监控配置
            "enable_wandb": self.env_config.enable_wandb,
            "log_interval": 10,
            "eval_interval": 50,
            "save_interval": 100,
            
            # 高级配置
            "distributed": False,
            "num_gpus": 1
        }
        
        # 应用额外参数
        config_dict.update(kwargs)
        
        return TrainingConfig(**config_dict)
    
    def build_lora_config(
        self,
        model_preset: str,
        training_preset: str,
        dataset_path: str,
        training_name: str,
        lora_preset: str = "standard",
        **kwargs
    ) -> TrainingConfig:
        """
        构建LoRA训练配置
        
        Args:
            model_preset: 模型预设名称
            training_preset: 训练预设名称
            dataset_path: 数据集路径
            training_name: 训练名称
            lora_preset: LoRA预设名称
            **kwargs: 额外配置参数
            
        Returns:
            TrainingConfig: 训练配置
        """
        # 获取基础配置
        config = self.build_config(model_preset, training_preset, dataset_path, training_name, **kwargs)
        
        # 添加LoRA配置
        lora_config = self.scenarios.LORA_PRESETS[lora_preset]
        config.use_lora = True
        config.lora_config = lora_config
        
        return config
    
    def build_instruction_config(
        self,
        model_preset: str,
        dataset_path: str,
        training_name: str,
        **kwargs
    ) -> TrainingConfig:
        """
        构建指令微调配置
        
        Args:
            model_preset: 模型预设名称
            dataset_path: 数据集路径
            training_name: 训练名称
            **kwargs: 额外配置参数
            
        Returns:
            TrainingConfig: 训练配置
        """
        config = self.build_config(
            model_preset,
            "instruction",
            dataset_path,
            training_name,
            instruction_tuning=True,
            **kwargs
        )
        
        return config
    
    def build_conversation_config(
        self,
        model_preset: str,
        dataset_path: str,
        training_name: str,
        chat_template: str = "chatml",
        **kwargs
    ) -> TrainingConfig:
        """
        构建对话训练配置
        
        Args:
            model_preset: 模型预设名称
            dataset_path: 数据集路径
            training_name: 训练名称
            chat_template: 对话模板
            **kwargs: 额外配置参数
            
        Returns:
            TrainingConfig: 训练配置
        """
        config = self.build_config(
            model_preset,
            "conversation",
            dataset_path,
            training_name,
            chat_template=chat_template,
            **kwargs
        )
        
        return config


# 默认环境配置
DEFAULT_ENV_CONFIG = EnvironmentConfig()

# 全局配置构建器
config_builder = TrainingConfigBuilder(DEFAULT_ENV_CONFIG)


def get_training_config(
    model_preset: str,
    training_preset: str,
    dataset_path: str,
    training_name: str,
    **kwargs
) -> TrainingConfig:
    """
    获取训练配置的便捷函数
    
    Args:
        model_preset: 模型预设名称
        training_preset: 训练预设名称
        dataset_path: 数据集路径
        training_name: 训练名称
        **kwargs: 额外配置参数
        
    Returns:
        TrainingConfig: 训练配置
    """
    return config_builder.build_config(model_preset, training_preset, dataset_path, training_name, **kwargs)


def get_lora_config(
    model_preset: str,
    dataset_path: str,
    training_name: str,
    lora_preset: str = "standard",
    **kwargs
) -> TrainingConfig:
    """
    获取LoRA训练配置的便捷函数
    
    Args:
        model_preset: 模型预设名称
        dataset_path: 数据集路径
        training_name: 训练名称
        lora_preset: LoRA预设名称
        **kwargs: 额外配置参数
        
    Returns:
        TrainingConfig: 训练配置
    """
    return config_builder.build_lora_config(
        model_preset, "lora", dataset_path, training_name, lora_preset, **kwargs
    )


def get_instruction_config(
    model_preset: str,
    dataset_path: str,
    training_name: str,
    **kwargs
) -> TrainingConfig:
    """
    获取指令微调配置的便捷函数
    
    Args:
        model_preset: 模型预设名称
        dataset_path: 数据集路径
        training_name: 训练名称
        **kwargs: 额外配置参数
        
    Returns:
        TrainingConfig: 训练配置
    """
    return config_builder.build_instruction_config(model_preset, dataset_path, training_name, **kwargs)


def get_conversation_config(
    model_preset: str,
    dataset_path: str,
    training_name: str,
    chat_template: str = "chatml",
    **kwargs
) -> TrainingConfig:
    """
    获取对话训练配置的便捷函数
    
    Args:
        model_preset: 模型预设名称
        dataset_path: 数据集路径
        training_name: 训练名称
        chat_template: 对话模板
        **kwargs: 额外配置参数
        
    Returns:
        TrainingConfig: 训练配置
    """
    return config_builder.build_conversation_config(
        model_preset, dataset_path, training_name, chat_template, **kwargs
    )


def list_available_presets() -> Dict[str, List[str]]:
    """
    列出可用的预设配置
    
    Returns:
        Dict[str, List[str]]: 预设配置列表
    """
    scenarios = TrainingScenarios()
    return {
        "model_presets": list(scenarios.MODEL_PRESETS.keys()),
        "training_presets": list(scenarios.TRAINING_PRESETS.keys()),
        "lora_presets": list(scenarios.LORA_PRESETS.keys())
    }


def get_preset_info(preset_type: str, preset_name: str) -> Dict[str, Any]:
    """
    获取预设配置信息
    
    Args:
        preset_type: 预设类型 ("model", "training", "lora")
        preset_name: 预设名称
        
    Returns:
        Dict[str, Any]: 预设信息
    """
    scenarios = TrainingScenarios()
    
    if preset_type == "model":
        preset = scenarios.MODEL_PRESETS.get(preset_name)
    elif preset_type == "training":
        preset = scenarios.TRAINING_PRESETS.get(preset_name)
    elif preset_type == "lora":
        preset = scenarios.LORA_PRESETS.get(preset_name)
    else:
        raise ValueError(f"不支持的预设类型: {preset_type}")
    
    if preset is None:
        raise ValueError(f"预设不存在: {preset_name}")
    
    return preset.__dict__ 