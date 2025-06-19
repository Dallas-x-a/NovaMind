"""
NovaMind LLM训练器 - 专门针对大语言模型的训练系统

提供针对大语言模型的专门训练功能：
- 支持多种LLM架构（GPT、BERT、T5等）
- 智能分词和数据处理
- 文本生成质量评估
- 对话训练支持
- 指令微调
- 强化学习训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    DataCollatorForLanguageModeling, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, HfArgumentParser
)
from datasets import Dataset as HFDataset
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

from .trainer import BaseTrainer, TrainingConfig, TrainingMetrics, LoRAConfig
from ..core.config import Config


class LLMTrainingConfig(TrainingConfig):
    """LLM训练配置"""
    
    # 模型配置
    model_path: str = Field(..., description="预训练模型路径")
    tokenizer_path: Optional[str] = Field(default=None, description="分词器路径")
    
    # 数据处理配置
    max_length: int = Field(default=512, description="最大序列长度")
    truncation: bool = Field(default=True, description="是否截断")
    padding: str = Field(default="max_length", description="填充策略")
    
    # 训练任务配置
    task_type: str = Field(default="causal_lm", description="任务类型")
    instruction_tuning: bool = Field(default=False, description="是否进行指令微调")
    chat_template: Optional[str] = Field(default=None, description="对话模板")
    
    # 评估配置
    eval_metrics: List[str] = Field(default_factory=lambda: ["perplexity"], description="评估指标")
    generation_config: Dict[str, Any] = Field(default_factory=dict, description="生成配置")
    
    # 高级配置
    gradient_checkpointing: bool = Field(default=True, description="梯度检查点")
    flash_attention: bool = Field(default=False, description="使用Flash Attention")
    deepspeed: bool = Field(default=False, description="使用DeepSpeed")
    fsdp: bool = Field(default=False, description="使用FSDP")


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        初始化文本数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif data_path.endswith('.jsonl'):
            self.data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        else:
            # 纯文本文件
            with open(data_path, 'r', encoding='utf-8') as f:
                texts = f.read().split('\n')
                self.data = [{'text': text.strip()} for text in texts if text.strip()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if 'text' in item:
            # 纯文本数据
            text = item['text']
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze()
            }
        elif 'instruction' in item and 'response' in item:
            # 指令数据
            instruction = item['instruction']
            response = item['response']
            
            # 构建输入文本
            if item.get('input'):
                input_text = f"Instruction: {instruction}\nInput: {item['input']}\nResponse: {response}"
            else:
                input_text = f"Instruction: {instruction}\nResponse: {response}"
            
            encoding = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze()
            }
        else:
            raise ValueError(f"不支持的数据格式: {item}")


class ConversationDataset(Dataset):
    """对话数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, chat_template: str = "chatml"):
        """
        初始化对话数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            chat_template: 对话模板
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = chat_template
        
        # 加载对话数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.conversations = json.load(f)
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # 应用对话模板
        if self.chat_template == "chatml":
            formatted_text = self._format_chatml(conversation)
        elif self.chat_template == "alpaca":
            formatted_text = self._format_alpaca(conversation)
        else:
            formatted_text = self._format_default(conversation)
        
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }
    
    def _format_chatml(self, conversation):
        """ChatML格式"""
        formatted = ""
        for turn in conversation:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted
    
    def _format_alpaca(self, conversation):
        """Alpaca格式"""
        if len(conversation) >= 2:
            instruction = conversation[0].get('content', '')
            response = conversation[1].get('content', '')
            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return ""
    
    def _format_default(self, conversation):
        """默认格式"""
        return "\n".join([turn.get('content', '') for turn in conversation])


class LLMTrainer(BaseTrainer):
    """LLM训练器"""
    
    def __init__(self, config: LLMTrainingConfig):
        """
        初始化LLM训练器
        
        Args:
            config: LLM训练配置
        """
        super().__init__(config)
        self.config = config
        self.tokenizer = None
        self.model = None
        
        # 初始化分词器
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """设置分词器"""
        tokenizer_path = self.config.tokenizer_path or self.config.model_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # 设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print(f"分词器加载成功: {tokenizer_path}")
            
        except Exception as e:
            print(f"分词器加载失败: {e}")
            raise
    
    def prepare_model(self) -> nn.Module:
        """准备模型"""
        try:
            if self.config.task_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
                    gradient_checkpointing=self.config.gradient_checkpointing,
                    use_flash_attention_2=self.config.flash_attention
                )
            elif self.config.task_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
                    gradient_checkpointing=self.config.gradient_checkpointing
                )
            else:
                raise ValueError(f"不支持的任务类型: {self.config.task_type}")
            
            print(f"模型加载成功: {self.config.model_path}")
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def prepare_data(self) -> tuple[DataLoader, Optional[DataLoader]]:
        """准备数据"""
        # 训练数据
        if self.config.instruction_tuning or self.config.chat_template:
            train_dataset = ConversationDataset(
                self.config.dataset_path,
                self.tokenizer,
                self.config.max_length,
                self.config.chat_template
            )
        else:
            train_dataset = TextDataset(
                self.config.dataset_path,
                self.tokenizer,
                self.config.max_length
            )
        
        # 数据收集器
        if self.config.task_type == "causal_lm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=4
        )
        
        # 评估数据（如果有的话）
        eval_loader = None
        if hasattr(self.config, 'eval_dataset_path') and self.config.eval_dataset_path:
            if self.config.instruction_tuning or self.config.chat_template:
                eval_dataset = ConversationDataset(
                    self.config.eval_dataset_path,
                    self.tokenizer,
                    self.config.max_length,
                    self.config.chat_template
                )
            else:
                eval_dataset = TextDataset(
                    self.config.eval_dataset_path,
                    self.tokenizer,
                    self.config.max_length
                )
            
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=4
            )
        
        return train_loader, eval_loader
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        # 移动数据到设备
        device = next(self.model.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
        
        # 计算梯度范数
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # 优化器步进
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # 调度器步进
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'gradient_norm': grad_norm,
            'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        }
    
    def evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """评估步骤"""
        device = next(self.model.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 计算困惑度
            perplexity = torch.exp(loss).item()
            
            # 计算准确率（如果适用）
            accuracy = 0.0
            if self.config.task_type == "causal_lm":
                # 对于因果语言模型，计算下一个token的准确率
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # 计算准确率
                pred_tokens = torch.argmax(shift_logits, dim=-1)
                correct = (pred_tokens == shift_labels).sum().item()
                total = shift_labels.numel()
                accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity,
            'accuracy': accuracy
        }
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            
        Returns:
            str: 生成的文本
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(device)
        
        # 生成配置
        generation_config = {
            'max_length': max_length,
            'temperature': temperature,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            **self.config.generation_config
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入提示
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def evaluate_generation_quality(self, test_prompts: List[str], references: List[str] = None) -> Dict[str, float]:
        """
        评估生成质量
        
        Args:
            test_prompts: 测试提示列表
            references: 参考文本列表（可选）
            
        Returns:
            Dict[str, float]: 评估指标
        """
        generated_texts = []
        
        for prompt in test_prompts:
            generated = self.generate_text(prompt)
            generated_texts.append(generated)
        
        metrics = {}
        
        # 计算平均长度
        avg_length = np.mean([len(text) for text in generated_texts])
        metrics['avg_length'] = avg_length
        
        # 计算困惑度
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for prompt in test_prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                ).to(device)
                
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item() * inputs['input_ids'].numel()
                total_tokens += inputs['input_ids'].numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_loss)
        metrics['perplexity'] = perplexity
        
        # 如果有参考文本，计算BLEU分数
        if references:
            try:
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                
                bleu_scores = []
                for ref, hyp in zip(references, generated_texts):
                    ref_tokens = ref.split()
                    hyp_tokens = hyp.split()
                    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)
                    bleu_scores.append(bleu)
                
                metrics['bleu_score'] = np.mean(bleu_scores)
                
            except ImportError:
                print("NLTK未安装，跳过BLEU计算")
        
        return metrics
    
    def save_model(self, output_path: str):
        """保存模型"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(output_path)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_path)
        
        # 保存配置
        config_path = output_path / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"模型已保存到: {output_path}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        model_path = Path(model_path)
        
        # 加载模型
        if self.config.task_type == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载配置
        config_path = model_path / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.config = LLMTrainingConfig(**config_data)
        
        print(f"模型已从 {model_path} 加载")


class RLHFTrainer(LLMTrainer):
    """强化学习训练器（RLHF）"""
    
    def __init__(self, config: LLMTrainingConfig):
        super().__init__(config)
        self.reward_model = None
        self.critic_model = None
    
    def setup_reward_model(self, reward_model_path: str):
        """设置奖励模型"""
        self.reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
        self.reward_model.eval()
        print(f"奖励模型已加载: {reward_model_path}")
    
    def compute_reward(self, generated_text: str) -> float:
        """计算奖励分数"""
        if self.reward_model is None:
            return 0.0
        
        # 这里实现具体的奖励计算逻辑
        # 可以使用预训练的奖励模型或其他评估方法
        return 0.0
    
    def rlhf_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """RLHF训练步骤"""
        # 实现PPO或其他RLHF算法
        # 这里是一个简化的实现
        return self.training_step(batch) 