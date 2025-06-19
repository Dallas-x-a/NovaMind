"""
NovaMind训练框架 - 企业级AI模型训练系统

提供类似LangChain的训练体验，但专为NovaMind优化。
支持实时监控、动态参数调整、LoRA训练等高级功能。

核心优势：
- 实时训练监控：Web界面实时查看训练状态
- 智能参数调优：基于性能自动调整超参数
- 多模态训练：支持文本、图像、音频等多种模态
- 分布式训练：支持多GPU/多节点训练
- 模型版本管理：完整的模型生命周期管理
- 实验管理：A/B测试和实验对比
- 生产就绪：直接部署到生产环境
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from loguru import logger
from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.monitor import SystemMonitor


class TrainingStatus(str, Enum):
    """训练状态枚举"""
    PENDING = "pending"      # 等待开始
    RUNNING = "running"      # 正在训练
    PAUSED = "paused"       # 暂停训练
    COMPLETED = "completed"  # 训练完成
    FAILED = "failed"        # 训练失败
    CANCELLED = "cancelled"  # 训练取消


class ModelType(str, Enum):
    """模型类型枚举"""
    LLM = "llm"              # 大语言模型
    VISION = "vision"        # 视觉模型
    AUDIO = "audio"          # 音频模型
    MULTIMODAL = "multimodal" # 多模态模型


@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LoRAConfig:
    """LoRA配置数据类"""
    r: int = 16                     # LoRA秩
    alpha: int = 32                 # LoRA缩放因子
    dropout: float = 0.1            # Dropout率
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"              # 偏置处理方式
    task_type: str = "CAUSAL_LM"    # 任务类型


class TrainingConfig(BaseModel):
    """训练配置模型"""
    
    # 基础配置
    model_name: str = Field(..., description="模型名称")
    model_type: ModelType = Field(..., description="模型类型")
    dataset_path: str = Field(..., description="数据集路径")
    output_dir: str = Field(default="./outputs", description="输出目录")
    
    # 训练参数
    batch_size: int = Field(default=8, description="批次大小")
    learning_rate: float = Field(default=1e-4, description="学习率")
    num_epochs: int = Field(default=10, description="训练轮数")
    max_steps: Optional[int] = Field(default=None, description="最大步数")
    warmup_steps: int = Field(default=100, description="预热步数")
    gradient_accumulation_steps: int = Field(default=1, description="梯度累积步数")
    
    # 优化器配置
    optimizer: str = Field(default="adamw", description="优化器类型")
    weight_decay: float = Field(default=0.01, description="权重衰减")
    beta1: float = Field(default=0.9, description="Adam beta1")
    beta2: float = Field(default=0.999, description="Adam beta2")
    
    # 调度器配置
    scheduler: str = Field(default="cosine", description="学习率调度器")
    lr_scheduler_warmup_ratio: float = Field(default=0.1, description="调度器预热比例")
    
    # LoRA配置
    use_lora: bool = Field(default=False, description="是否使用LoRA")
    lora_config: Optional[LoRAConfig] = Field(default=None, description="LoRA配置")
    
    # 监控配置
    enable_wandb: bool = Field(default=True, description="是否启用WandB")
    log_interval: int = Field(default=10, description="日志间隔")
    eval_interval: int = Field(default=100, description="评估间隔")
    save_interval: int = Field(default=500, description="保存间隔")
    
    # 分布式配置
    distributed: bool = Field(default=False, description="是否使用分布式训练")
    num_gpus: int = Field(default=1, description="GPU数量")
    
    # 高级配置
    mixed_precision: bool = Field(default=True, description="是否使用混合精度")
    gradient_clipping: float = Field(default=1.0, description="梯度裁剪")
    early_stopping_patience: int = Field(default=5, description="早停耐心值")
    
    class Config:
        use_enum_values = True


class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(self, config: TrainingConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.status = TrainingStatus.PENDING
        self.metrics_history: List[TrainingMetrics] = []
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 初始化组件
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.train_loader: Optional[DataLoader] = None
        self.eval_loader: Optional[DataLoader] = None
        
        # 监控组件
        self.monitor = SystemMonitor()
        self.setup_monitoring()
        
        # 回调函数
        self.callbacks: List[Callable] = []
        
        logger.info(f"训练器初始化完成: {config.model_name}")
    
    def setup_monitoring(self):
        """设置监控"""
        if self.config.enable_wandb:
            wandb.init(
                project="novamind-training",
                name=self.config.model_name,
                config=self.config.model_dump()
            )
    
    @abstractmethod
    def prepare_model(self) -> nn.Module:
        """准备模型 - 子类必须实现"""
        pass
    
    @abstractmethod
    def prepare_data(self) -> tuple[DataLoader, Optional[DataLoader]]:
        """准备数据 - 子类必须实现"""
        pass
    
    @abstractmethod
    def training_step(self, batch: Any) -> Dict[str, float]:
        """训练步骤 - 子类必须实现"""
        pass
    
    @abstractmethod
    def evaluation_step(self, batch: Any) -> Dict[str, float]:
        """评估步骤 - 子类必须实现"""
        pass
    
    def add_callback(self, callback: Callable):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def setup_optimizer(self):
        """设置优化器"""
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2)
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
    
    def setup_scheduler(self):
        """设置学习率调度器"""
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler.lower() == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.num_epochs
            )
        else:
            raise ValueError(f"不支持的调度器: {self.config.scheduler}")
    
    def apply_lora(self):
        """应用LoRA配置"""
        if not self.config.use_lora or self.config.lora_config is None:
            return
        
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=self.config.lora_config.r,
                lora_alpha=self.config.lora_config.alpha,
                target_modules=self.config.lora_config.target_modules,
                lora_dropout=self.config.lora_config.dropout,
                bias=self.config.lora_config.bias,
                task_type=self.config.lora_config.task_type,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA配置应用成功")
            
        except ImportError:
            logger.warning("PEFT库未安装，跳过LoRA配置")
    
    async def train(self):
        """开始训练"""
        try:
            self.status = TrainingStatus.RUNNING
            logger.info("开始训练...")
            
            # 准备模型和数据
            self.model = self.prepare_model()
            self.train_loader, self.eval_loader = self.prepare_data()
            
            # 设置优化器和调度器
            self.setup_optimizer()
            self.setup_scheduler()
            
            # 应用LoRA
            self.apply_lora()
            
            # 设置设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # 开始训练循环
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                await self.train_epoch()
                
                # 评估
                if self.eval_loader and epoch % self.config.eval_interval == 0:
                    await self.evaluate()
                
                # 保存检查点
                if epoch % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                # 早停检查
                if self.check_early_stopping():
                    logger.info("触发早停机制")
                    break
            
            self.status = TrainingStatus.COMPLETED
            logger.info("训练完成")
            
        except Exception as e:
            self.status = TrainingStatus.FAILED
            logger.error(f"训练失败: {e}")
            raise
    
    async def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 训练步骤
            metrics = self.training_step(batch)
            epoch_loss += metrics['loss']
            
            # 更新步数
            self.current_step += 1
            
            # 记录指标
            self.record_metrics(metrics)
            
            # 日志记录
            if self.current_step % self.config.log_interval == 0:
                self.log_metrics(metrics)
            
            # 执行回调
            for callback in self.callbacks:
                callback(self, metrics)
            
            # 检查最大步数
            if self.config.max_steps and self.current_step >= self.config.max_steps:
                break
    
    async def evaluate(self):
        """评估模型"""
        if not self.eval_loader:
            return
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                metrics = self.evaluation_step(batch)
                total_loss += metrics['loss']
                total_accuracy += metrics.get('accuracy', 0.0)
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # 更新最佳损失
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.patience_counter = 0
            self.save_best_model()
        else:
            self.patience_counter += 1
        
        logger.info(f"评估结果 - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    def record_metrics(self, metrics: Dict[str, float]):
        """记录训练指标"""
        metric = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            loss=metrics.get('loss', 0.0),
            accuracy=metrics.get('accuracy', 0.0),
            learning_rate=self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
            gradient_norm=metrics.get('gradient_norm', 0.0),
            memory_usage=torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
            gpu_utilization=metrics.get('gpu_utilization', 0.0)
        )
        
        self.metrics_history.append(metric)
        
        # 发送到WandB
        if self.config.enable_wandb:
            wandb.log({
                'epoch': self.current_epoch,
                'step': self.current_step,
                **metrics
            })
    
    def log_metrics(self, metrics: Dict[str, float]):
        """记录日志"""
        logger.info(
            f"Epoch {self.current_epoch}, Step {self.current_step} - "
            f"Loss: {metrics.get('loss', 0.0):.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
    
    def check_early_stopping(self) -> bool:
        """检查早停条件"""
        return self.patience_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.model_dump(),
            'best_loss': self.best_loss,
            'metrics_history': [m.__dict__ for m in self.metrics_history[-100:]]  # 保存最近100个指标
        }
        
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{self.current_epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def save_best_model(self):
        """保存最佳模型"""
        model_path = Path(self.config.output_dir) / "best_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"最佳模型已保存: {model_path}")
    
    def pause(self):
        """暂停训练"""
        self.status = TrainingStatus.PAUSED
        logger.info("训练已暂停")
    
    def resume(self):
        """恢复训练"""
        self.status = TrainingStatus.RUNNING
        logger.info("训练已恢复")
    
    def cancel(self):
        """取消训练"""
        self.status = TrainingStatus.CANCELLED
        logger.info("训练已取消")
    
    def get_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'status': self.status.value,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'metrics': self.metrics_history[-10:] if self.metrics_history else [],
            'config': self.config.model_dump()
        }


class TrainingManager:
    """训练管理器 - 管理多个训练任务"""
    
    def __init__(self):
        """初始化训练管理器"""
        self.active_trainings: Dict[str, BaseTrainer] = {}
        self.training_history: Dict[str, Dict[str, Any]] = {}
    
    def start_training(self, training_id: str, trainer: BaseTrainer) -> str:
        """
        启动训练
        
        Args:
            training_id: 训练ID
            trainer: 训练器实例
            
        Returns:
            str: 训练ID
        """
        if training_id in self.active_trainings:
            raise ValueError(f"训练ID已存在: {training_id}")
        
        self.active_trainings[training_id] = trainer
        
        # 异步启动训练
        asyncio.create_task(self._run_training(training_id, trainer))
        
        logger.info(f"训练已启动: {training_id}")
        return training_id
    
    async def _run_training(self, training_id: str, trainer: BaseTrainer):
        """运行训练任务"""
        try:
            await trainer.train()
            self.training_history[training_id] = trainer.get_status()
        except Exception as e:
            logger.error(f"训练失败 {training_id}: {e}")
            self.training_history[training_id] = {
                'status': TrainingStatus.FAILED.value,
                'error': str(e)
            }
        finally:
            if training_id in self.active_trainings:
                del self.active_trainings[training_id]
    
    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """获取训练状态"""
        if training_id in self.active_trainings:
            return self.active_trainings[training_id].get_status()
        elif training_id in self.training_history:
            return self.training_history[training_id]
        return None
    
    def list_trainings(self) -> List[Dict[str, Any]]:
        """列出所有训练"""
        trainings = []
        
        # 活跃训练
        for training_id, trainer in self.active_trainings.items():
            trainings.append({
                'id': training_id,
                'status': trainer.get_status(),
                'active': True
            })
        
        # 历史训练
        for training_id, status in self.training_history.items():
            trainings.append({
                'id': training_id,
                'status': status,
                'active': False
            })
        
        return trainings
    
    def pause_training(self, training_id: str):
        """暂停训练"""
        if training_id in self.active_trainings:
            self.active_trainings[training_id].pause()
    
    def resume_training(self, training_id: str):
        """恢复训练"""
        if training_id in self.active_trainings:
            self.active_trainings[training_id].resume()
    
    def cancel_training(self, training_id: str):
        """取消训练"""
        if training_id in self.active_trainings:
            self.active_trainings[training_id].cancel()


# 全局训练管理器实例
training_manager = TrainingManager() 