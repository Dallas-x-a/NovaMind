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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


# 基础配置
MODEL_PATH = "deepseek_7B/"
DATASET_PATH = "example.json"
DEVICE = torch.device("cuda:0")


# 训练参数配置
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 5e-6,
    "num_train_epochs": 5,
    "max_seq_length": 1024,
    "logging_steps": 50,
    "eval_steps": 100,
    "save_steps": 500,
    "output_dir": "./fine_tuned_model",
    "deepseek_model": "./deepseek_fine_tuned",
    "optim": "adamw_torch",
    "fp16": True,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "remove_unused_columns": False,
    "ddp_find_unused_parameters": False,
    "torch_compile": True,
    "gradient_checkpointing": True,
}

def compute_metrics(eval_preds):
    """计算评估指标，包括损失、准确率和困惑度"""
    predictions, labels = eval_preds
    predictions = predictions.argmax(axis=-1)
    
    # 计算损失
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(predictions).float(),
        torch.tensor(labels).long()
    ).item()
    
    # 计算准确率（只考虑非填充和非特殊token的位置）
    valid_mask = (labels != -100) & (labels != tokenizer.pad_token_id)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    # 计算token级别的准确率
    token_accuracy = (valid_predictions == valid_labels).mean()
    
    # 计算序列级别的准确率（整个回答完全正确的比例）
    sequence_correct = 0
    total_sequences = 0
    
    # 按序列长度分组计算
    for pred_seq, label_seq in zip(predictions, labels):
        # 找到有效token的位置
        valid_positions = (label_seq != -100) & (label_seq != tokenizer.pad_token_id)
        if valid_positions.any():
            pred_tokens = pred_seq[valid_positions]
            label_tokens = label_seq[valid_positions]
            # 如果预测序列完全匹配标签序列，则认为序列正确
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
    """优化后的训练样本格式构建，添加更严格的提示模板"""
    system_prompt = (
        "你是一个专业的暖通领域智能助手。请严格按照以下要求回答问题：\n"
        "1. 回答必须准确、专业、完整\n"
        "2. 使用清晰的语言和结构化的格式\n"
        "3. 如果涉及具体数值，必须准确引用\n"
        "4. 如果涉及专业术语，需要适当解释\n"
        "5. 回答要简洁明了，避免冗余内容\n"
    )
    
    # 构建完整的提示文本，添加更清晰的结构
    full_prompt = (
        f"### 系统指令:\n{system_prompt}\n\n"
        f"### 问题:\n{example['question'].strip()}\n\n"
        f"### 回答:\n{example['answer'].strip()}\n\n"
        f"### 结束\n</s>"
    )
    
    return {"text": full_prompt}

def prepare_dataset(examples, tokenizer, max_length):
    """优化数据集处理，添加更严格的数据清洗和预处理"""
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
        if not text.startswith("### 系统指令:") or not text.endswith("</s>"):
            invalid_format += 1
            continue
            
        # 使用tokenizer的快速模式检查长度
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
        token_length = len(tokens)
        
        if token_length <= 20:  # 增加最小长度限制
            too_short += 1
            continue
        if token_length >= max_length * 0.8:  # 调整最大长度限制
            too_long += 1
            continue
            
        texts.append(text)
        lengths.append(token_length)
        valid_count += 1
    
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
    
    # 优化labels处理
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
    """加载并划分数据集"""
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

if __name__ == "__main__":
    # 检查数据文件是否存在
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"训练数据文件 {DATASET_PATH} 不存在")

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        pad_token='</s>',
        eos_token='</s>',
        trust_remote_code=True
    )

    print("加载模型配置...")
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
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
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.gradient_checkpointing_enable()

    print("准备数据集...")
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(tokenizer, DATASET_PATH)
    
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

    # 清理缓存
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
    os.makedirs(TRAINING_CONFIG["deepseek_model"], exist_ok=True)
    model.save_pretrained(TRAINING_CONFIG["deepseek_model"])
    tokenizer.save_pretrained(TRAINING_CONFIG["deepseek_model"])
    print(f"训练完成，最终模型已保存至 {TRAINING_CONFIG['deepseek_model']}")