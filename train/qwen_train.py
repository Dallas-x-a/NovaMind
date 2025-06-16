from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoConfig,
    EvalPrediction
)
import json
import torch
from datasets import Dataset
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# 设置环境变量，禁用tokenizer并行处理以避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 基础配置
MODEL_PATH = "Qwen/Qwen-7B"  # Qwen模型路径
DATASET_PATH = "example.json"  # 训练数据路径
DEVICE = torch.device("cuda:0")  # 使用GPU训练

# 训练参数配置
TRAINING_CONFIG = {
    # 批次大小设置
    "per_device_train_batch_size": 2,  # 每个设备的训练批次大小
    "per_device_eval_batch_size": 1,   # 每个设备的评估批次大小
    "gradient_accumulation_steps": 32,  # 梯度累积步数，用于模拟更大的批次大小
    
    # 学习率和优化器设置
    "learning_rate": 5e-6,             # 学习率
    "num_train_epochs": 5,             # 训练轮数
    "warmup_ratio": 0.1,               # 预热比例
    "weight_decay": 0.01,              # 权重衰减
    "optim": "adamw_torch",            # 优化器类型
    
    # 序列长度和训练策略
    "max_seq_length": 2048,            # 最大序列长度，Qwen支持更长的上下文
    "logging_steps": 50,               # 日志记录步数
    "eval_steps": 100,                 # 评估步数
    "save_steps": 500,                 # 保存步数
    
    # 输出和保存设置
    "output_dir": "./qwen_fine_tuned",  # 输出目录
    "qwen_model": "./qwen_final_model", # 最终模型保存路径
    
    # 训练优化设置
    "fp16": True,                      # 使用混合精度训练
    "lr_scheduler_type": "cosine",     # 学习率调度器类型
    "evaluation_strategy": "steps",     # 评估策略
    "save_strategy": "steps",          # 保存策略
    "save_total_limit": 3,             # 保存的检查点数量限制
    "load_best_model_at_end": True,    # 训练结束时加载最佳模型
    "metric_for_best_model": "eval_loss", # 用于选择最佳模型的指标
    "greater_is_better": False,        # 指标是否越大越好
    "remove_unused_columns": False,    # 是否移除未使用的列
    "ddp_find_unused_parameters": False, # DDP是否查找未使用的参数
    "torch_compile": True,             # 是否使用torch.compile加速
    "gradient_checkpointing": True,    # 是否使用梯度检查点以节省显存
    
    # 数据集划分比例
    "train_ratio": 0.8,               # 训练集比例
    "val_ratio": 0.1,                 # 验证集比例
    "test_ratio": 0.1,                # 测试集比例
}

def compute_metrics(eval_preds):
    """
    计算模型评估指标
    
    参数:
        eval_preds: 包含预测值和标签的元组
        
    返回:
        包含各项评估指标的字典，包括：
        - eval_loss: 评估损失
        - token_accuracy: token级别的准确率
        - sequence_accuracy: 序列级别的准确率
        - perplexity: 困惑度
    """
    predictions, labels = eval_preds
    predictions = predictions.argmax(axis=-1)
    
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(predictions).float(),
        torch.tensor(labels).long()
    ).item()
    
    # 创建有效token的掩码（排除填充和特殊token）
    valid_mask = (labels != -100) & (labels != tokenizer.pad_token_id)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    # 计算token级别的准确率
    token_accuracy = (valid_predictions == valid_labels).mean()
    
    # 计算序列级别的准确率
    sequence_correct = 0
    total_sequences = 0
    
    for pred_seq, label_seq in zip(predictions, labels):
        valid_positions = (label_seq != -100) & (label_seq != tokenizer.pad_token_id)
        if valid_positions.any():
            pred_tokens = pred_seq[valid_positions]
            label_tokens = label_seq[valid_positions]
            if (pred_tokens == label_tokens).all():
                sequence_correct += 1
            total_sequences += 1
    
    sequence_accuracy = sequence_correct / total_sequences if total_sequences > 0 else 0
    
    # 计算困惑度
    perplexity = torch.exp(torch.tensor(loss)).item()
    
    return {
        "eval_loss": loss,
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "perplexity": perplexity
    }

def format_instruction(example):
    """
    格式化训练样本，构建标准化的提示模板
    
    参数:
        example: 包含问题和答案的字典
        
    返回:
        格式化后的文本字典
    """
    # 系统提示词，定义模型角色和行为准则
    system_prompt = (
        "你是一个专业的暖通领域智能助手。请严格按照以下要求回答问题：\n"
        "1. 回答必须准确、专业、完整\n"
        "2. 使用清晰的语言和结构化的格式\n"
        "3. 如果涉及具体数值，必须准确引用\n"
        "4. 如果涉及专业术语，需要适当解释\n"
        "5. 回答要简洁明了，避免冗余内容\n"
    )
    
    # 构建完整的提示文本
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{example['question'].strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['answer'].strip()}<|im_end|>"
    )
    
    return {"text": full_prompt}

def prepare_dataset(examples, tokenizer, max_length):
    """
    准备训练数据集，包括数据清洗和预处理
    
    参数:
        examples: 原始样本列表
        tokenizer: 分词器
        max_length: 最大序列长度
        
    返回:
        处理后的数据集字典
    """
    print(f"\n开始处理 {len(examples)} 个样本...")
    valid_count = 0
    too_short = 0
    too_long = 0
    invalid_format = 0
    
    texts = []
    lengths = []
    
    # 预处理所有文本
    for example in examples:
        # 数据清洗和规范化
        text = format_instruction(example)["text"]
        text = " ".join(text.split())  # 规范化空白字符
        text = text.replace("\n\n\n", "\n\n")  # 规范化换行
        text = text.replace("  ", " ")  # 规范化空格
        
        # 验证文本格式
        if not text.startswith("<|im_start|>system") or not text.endswith("<|im_end|>"):
            invalid_format += 1
            continue
            
        # 检查序列长度
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
        token_length = len(tokens)
        
        if token_length <= 20:  # 最小长度限制
            too_short += 1
            continue
        if token_length >= max_length * 0.8:  # 最大长度限制
            too_long += 1
            continue
            
        texts.append(text)
        lengths.append(token_length)
        valid_count += 1
    
    # 打印数据清洗统计信息
    print(f"\n数据清洗结果:")
    print(f"有效样本数: {valid_count}")
    print(f"过短样本数: {too_short}")
    print(f"过长样本数: {too_long}")
    print(f"格式无效样本数: {invalid_format}")
    print(f"样本保留率: {valid_count/len(examples)*100:.2f}%")
    
    if not texts:
        raise ValueError("没有有效的训练样本")
    
    # 批量处理tokenization
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_special_tokens_mask=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        add_special_tokens=True,
        verbose=False
    )
    
    # 处理标签
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[tokenized["special_tokens_mask"].bool()] = -100
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "special_tokens_mask": tokenized["special_tokens_mask"]
    }

def load_and_split_dataset(tokenizer, file_path):
    """
    加载并划分数据集
    
    参数:
        tokenizer: 分词器
        file_path: 数据文件路径
        
    返回:
        训练集、验证集和测试集
    """
    # 加载数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("数据必须是列表格式")
    
    # 处理数据
    examples = []
    for item in data:
        if item is None:
            continue
        if isinstance(item, list):
            for sub_item in item:
                if sub_item is not None and isinstance(sub_item, dict):
                    if 'question' in sub_item and 'answer' in sub_item:
                        examples.append(sub_item)
        elif isinstance(item, dict):
            if 'question' in item and 'answer' in item:
                examples.append(item)
    
    if not examples:
        raise ValueError("没有找到有效的问答对")
    
    print(f"成功加载 {len(examples)} 个训练样本")
    
    # 划分数据集
    train_examples, temp_examples = train_test_split(
        examples,
        train_size=TRAINING_CONFIG["train_ratio"],
        random_state=42
    )
    
    val_examples, test_examples = train_test_split(
        temp_examples,
        train_size=TRAINING_CONFIG["val_ratio"]/(TRAINING_CONFIG["val_ratio"] + TRAINING_CONFIG["test_ratio"]),
        random_state=42
    )
    
    print(f"训练集大小: {len(train_examples)}")
    print(f"验证集大小: {len(val_examples)}")
    print(f"测试集大小: {len(test_examples)}")
    
    # 创建数据集
    train_dataset = Dataset.from_dict(prepare_dataset(train_examples, tokenizer, TRAINING_CONFIG["max_seq_length"]))
    val_dataset = Dataset.from_dict(prepare_dataset(val_examples, tokenizer, TRAINING_CONFIG["max_seq_length"]))
    test_dataset = Dataset.from_dict(prepare_dataset(test_examples, tokenizer, TRAINING_CONFIG["max_seq_length"]))
    
    return train_dataset, val_dataset, test_dataset

class CustomTrainer(Trainer):
    """
    自定义训练器类，继承自Trainer
    用于实现自定义的训练逻辑和回调
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs.get('tokenizer')
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        自定义损失计算逻辑
        
        参数:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
            
        返回:
            损失值或(损失值, 输出)元组
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss

if __name__ == "__main__":
    # 检查数据文件是否存在
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"训练数据文件 {DATASET_PATH} 不存在")

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 确保tokenizer有正确的特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载模型配置...")
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    # 配置模型参数
    config.use_cache = False
    config.gradient_checkpointing = True
    if hasattr(config, 'sliding_window'):
        config.sliding_window = None
    if hasattr(config, 'use_sliding_window'):
        config.use_sliding_window = False
    if hasattr(config, 'use_sdpa'):
        config.use_sdpa = False

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,  # Qwen使用float16
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()

    print("准备数据集...")
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(tokenizer, DATASET_PATH)
    
    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 配置训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        output_dir=TRAINING_CONFIG["output_dir"],
        optim=TRAINING_CONFIG["optim"],
        fp16=TRAINING_CONFIG["fp16"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        warmup_steps=int(TRAINING_CONFIG["num_train_epochs"] * TRAINING_CONFIG["warmup_ratio"]),
        weight_decay=TRAINING_CONFIG["weight_decay"],
        logging_dir="./logs",
        remove_unused_columns=TRAINING_CONFIG["remove_unused_columns"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        dataloader_num_workers=2,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=TRAINING_CONFIG["ddp_find_unused_parameters"],
        dataloader_pin_memory=False,
        evaluation_strategy=TRAINING_CONFIG["evaluation_strategy"],
        load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
        greater_is_better=TRAINING_CONFIG["greater_is_better"],
        torch_compile=TRAINING_CONFIG["torch_compile"],
    )

    print("创建训练器...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    print("启动训练...")
    for epoch in range(TRAINING_CONFIG["num_train_epochs"]):
        print(f"\n开始第 {epoch + 1} 轮训练...")
        trainer.train()
        
        # 每轮训练后保存模型
        epoch_model_path = os.path.join(TRAINING_CONFIG["output_dir"], f"epoch_{epoch + 1}")
        print(f"保存第 {epoch + 1} 轮模型到 {epoch_model_path}")
        os.makedirs(epoch_model_path, exist_ok=True)
        model.save_pretrained(epoch_model_path)
        tokenizer.save_pretrained(epoch_model_path)

    print("评估模型...")
    # 使用较小的batch size进行评估
    eval_batch_size = 1
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in range(0, len(test_dataset), eval_batch_size):
            batch = test_dataset[i:i + eval_batch_size]
            inputs = {
                "input_ids": torch.stack([torch.tensor(x) for x in batch["input_ids"]]).to(DEVICE),
                "attention_mask": torch.stack([torch.tensor(x) for x in batch["attention_mask"]]).to(DEVICE),
                "labels": torch.stack([torch.tensor(x) for x in batch["labels"]]).to(DEVICE)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item() * len(batch["input_ids"])
            total_samples += len(batch["input_ids"])
            
            # 清理GPU内存
            del outputs
            del inputs
            torch.cuda.empty_cache()
            
            if i % 10 == 0:
                print(f"已评估 {i}/{len(test_dataset)} 个样本")
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"测试集结果: 平均损失={avg_loss:.4f}, 困惑度={perplexity:.4f}")
    
    print("保存最终模型...")
    os.makedirs(TRAINING_CONFIG["qwen_model"], exist_ok=True)
    model.save_pretrained(TRAINING_CONFIG["qwen_model"])
    tokenizer.save_pretrained(TRAINING_CONFIG["qwen_model"])
    print(f"训练完成，最终模型已保存至 {TRAINING_CONFIG['qwen_model']}") 