"""
NovaMind训练框架快速开始指南

这个文件展示了如何使用NovaMind训练框架进行快速训练。
包含多个简单易用的示例，帮助用户快速上手。
"""

import asyncio
from pathlib import Path
from typing import Dict, Any

from .trainer import TrainingManager, training_manager
from .llm_trainer import LLMTrainer
from .config import (
    get_training_config,
    get_lora_config,
    get_instruction_config,
    get_conversation_config,
    list_available_presets,
    DEFAULT_ENV_CONFIG
)
from .monitor import training_monitor, MetricsCallback


def quick_start_example():
    """快速开始示例"""
    print("🚀 NovaMind训练框架快速开始")
    print("=" * 50)
    
    # 1. 查看可用预设
    presets = list_available_presets()
    print("📋 可用预设配置:")
    print(f"  模型预设: {presets['model_presets']}")
    print(f"  训练预设: {presets['training_presets']}")
    print(f"  LoRA预设: {presets['lora_presets']}")
    print()
    
    # 2. 创建环境配置
    DEFAULT_ENV_CONFIG.create_directories()
    print("📁 目录结构已创建")
    print()
    
    # 3. 创建示例数据
    create_sample_data()
    print("📊 示例数据已创建")
    print()


def create_sample_data():
    """创建示例数据"""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # 创建简单的文本数据
    sample_texts = [
        "人工智能是计算机科学的一个重要分支。",
        "机器学习使计算机能够从数据中学习。",
        "深度学习使用神经网络进行模式识别。",
        "自然语言处理让计算机理解人类语言。",
        "计算机视觉使计算机能够理解图像。"
    ]
    
    # 保存为JSON格式
    import json
    text_data = [{"text": text} for text in sample_texts]
    
    with open(data_dir / "sample_texts.json", "w", encoding="utf-8") as f:
        json.dump(text_data, f, ensure_ascii=False, indent=2)
    
    # 创建指令数据
    instruction_data = [
        {
            "instruction": "解释什么是人工智能",
            "response": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        },
        {
            "instruction": "什么是机器学习？",
            "response": "机器学习是人工智能的一个子集，使计算机能够在没有明确编程的情况下学习和改进。"
        }
    ]
    
    with open(data_dir / "sample_instructions.json", "w", encoding="utf-8") as f:
        json.dump(instruction_data, f, ensure_ascii=False, indent=2)


async def example_1_basic_training():
    """示例1：基础训练"""
    print("📚 示例1：基础语言模型训练")
    print("-" * 30)
    
    # 使用预设配置创建训练配置
    config = get_training_config(
        model_preset="dialo-gpt-small",  # 使用小型模型
        training_preset="quick",         # 快速训练
        dataset_path="./data/sample_texts.json",
        training_name="basic_example"
    )
    
    # 创建训练器
    trainer = LLMTrainer(config)
    
    # 添加监控回调
    trainer.add_callback(MetricsCallback("basic_example"))
    
    # 启动训练
    training_id = training_manager.start_training("basic_example", trainer)
    print(f"✅ 训练已启动，ID: {training_id}")
    
    # 等待训练完成
    await wait_for_training_completion(training_id)
    
    # 测试文本生成
    test_text_generation(trainer)
    
    print("✅ 基础训练示例完成\n")


async def example_2_lora_training():
    """示例2：LoRA微调"""
    print("🎯 示例2：LoRA微调训练")
    print("-" * 30)
    
    # 使用LoRA配置
    config = get_lora_config(
        model_preset="dialo-gpt-small",
        dataset_path="./data/sample_instructions.json",
        training_name="lora_example",
        lora_preset="efficient"  # 使用高效LoRA配置
    )
    
    # 创建训练器
    trainer = LLMTrainer(config)
    
    # 添加监控回调
    trainer.add_callback(MetricsCallback("lora_example"))
    
    # 启动训练
    training_id = training_manager.start_training("lora_example", trainer)
    print(f"✅ LoRA训练已启动，ID: {training_id}")
    
    # 等待训练完成
    await wait_for_training_completion(training_id)
    
    print("✅ LoRA训练示例完成\n")


async def example_3_instruction_tuning():
    """示例3：指令微调"""
    print("📝 示例3：指令微调训练")
    print("-" * 30)
    
    # 使用指令微调配置
    config = get_instruction_config(
        model_preset="dialo-gpt-small",
        dataset_path="./data/sample_instructions.json",
        training_name="instruction_example"
    )
    
    # 创建训练器
    trainer = LLMTrainer(config)
    
    # 添加监控回调
    trainer.add_callback(MetricsCallback("instruction_example"))
    
    # 启动训练
    training_id = training_manager.start_training("instruction_example", trainer)
    print(f"✅ 指令微调已启动，ID: {training_id}")
    
    # 等待训练完成
    await wait_for_training_completion(training_id)
    
    # 测试指令响应
    test_instruction_response(trainer)
    
    print("✅ 指令微调示例完成\n")


async def example_4_monitoring():
    """示例4：监控系统"""
    print("📊 示例4：训练监控系统")
    print("-" * 30)
    
    # 启动监控服务器
    training_monitor.start_background()
    print("🌐 监控服务器已启动")
    print("   访问 http://localhost:8080 查看实时监控")
    
    # 显示训练状态
    trainings = training_manager.list_trainings()
    print(f"📈 当前训练任务数量: {len(trainings)}")
    
    for training in trainings:
        status = training['status']
        print(f"   - {training['id']}: {status['status']}")
    
    print("✅ 监控系统示例完成\n")


async def wait_for_training_completion(training_id: str, timeout: int = 300):
    """
    等待训练完成
    
    Args:
        training_id: 训练ID
        timeout: 超时时间（秒）
    """
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = training_manager.get_training_status(training_id)
        if status:
            current_status = status['status']
            print(f"   训练状态: {current_status}")
            
            if current_status in ['completed', 'failed', 'cancelled']:
                return status
        
        await asyncio.sleep(5)
    
    print(f"   训练超时，ID: {training_id}")


def test_text_generation(trainer: LLMTrainer):
    """测试文本生成"""
    print("🧪 测试文本生成:")
    
    test_prompts = [
        "人工智能的应用",
        "机器学习的优势"
    ]
    
    for prompt in test_prompts:
        try:
            generated = trainer.generate_text(prompt, max_length=50, temperature=0.7)
            print(f"   提示: {prompt}")
            print(f"   生成: {generated}")
            print()
        except Exception as e:
            print(f"   生成失败: {e}")


def test_instruction_response(trainer: LLMTrainer):
    """测试指令响应"""
    print("🧪 测试指令响应:")
    
    test_instructions = [
        "解释什么是深度学习",
        "机器学习有哪些类型？"
    ]
    
    for instruction in test_instructions:
        try:
            # 构建指令格式
            prompt = f"Instruction: {instruction}\nResponse:"
            generated = trainer.generate_text(prompt, max_length=100, temperature=0.7)
            print(f"   指令: {instruction}")
            print(f"   响应: {generated}")
            print()
        except Exception as e:
            print(f"   响应失败: {e}")


def show_usage_tips():
    """显示使用提示"""
    print("💡 使用提示:")
    print("=" * 50)
    print("1. 选择合适的模型预设:")
    print("   - 小型模型 (gpt2-small, dialo-gpt-small): 快速实验")
    print("   - 中型模型 (llama-7b, qwen-7b): 平衡性能")
    print("   - 大型模型 (llama-13b): 最佳效果")
    print()
    print("2. 选择合适的训练预设:")
    print("   - quick: 快速实验，验证想法")
    print("   - standard: 标准训练，平衡效果")
    print("   - high_quality: 高质量训练，最佳效果")
    print()
    print("3. 监控训练过程:")
    print("   - 访问 http://localhost:8080 查看实时监控")
    print("   - 查看训练日志了解详细信息")
    print("   - 使用WandB进行实验管理")
    print()
    print("4. 保存和加载模型:")
    print("   - 训练完成后模型自动保存")
    print("   - 使用 trainer.load_model() 加载模型")
    print("   - 使用 trainer.generate_text() 进行推理")
    print()


async def run_all_examples():
    """运行所有示例"""
    print("🚀 开始运行NovaMind训练框架示例")
    print("=" * 60)
    
    try:
        # 快速开始
        quick_start_example()
        
        # 运行示例
        await example_1_basic_training()
        await example_2_lora_training()
        await example_3_instruction_tuning()
        await example_4_monitoring()
        
        # 显示使用提示
        show_usage_tips()
        
        print("🎉 所有示例运行完成！")
        print("📖 更多信息请查看文档和示例代码")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


def simple_training_example():
    """简单训练示例 - 一行代码启动训练"""
    print("⚡ 简单训练示例")
    print("=" * 30)
    
    # 一行代码启动训练
    config = get_training_config(
        model_preset="dialo-gpt-small",
        training_preset="quick",
        dataset_path="./data/sample_texts.json",
        training_name="simple_example"
    )
    
    trainer = LLMTrainer(config)
    trainer.add_callback(MetricsCallback("simple_example"))
    
    # 启动训练（异步）
    training_id = training_manager.start_training("simple_example", trainer)
    print(f"✅ 简单训练已启动: {training_id}")
    
    return training_id


if __name__ == "__main__":
    # 运行所有示例
    asyncio.run(run_all_examples())
    
    # 或者运行简单示例
    # simple_training_example() 