"""
NovaMind训练框架使用示例

展示如何使用NovaMind训练框架进行不同类型的模型训练：
1. 基础语言模型训练
2. 指令微调训练
3. 对话模型训练
4. LoRA微调
5. 实时监控
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from .trainer import TrainingManager, TrainingStatus
from .llm_trainer import LLMTrainingConfig, LLMTrainer
from .monitor import training_monitor, MetricsCallback


def create_sample_data():
    """创建示例数据"""
    
    # 创建输出目录
    data_dir = Path("./sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. 纯文本数据
    text_data = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
        "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。",
        "计算机视觉是人工智能的一个分支，使计算机能够从图像和视频中理解和提取信息。"
    ]
    
    with open(data_dir / "text_data.json", "w", encoding="utf-8") as f:
        json.dump([{"text": text} for text in text_data], f, ensure_ascii=False, indent=2)
    
    # 2. 指令数据
    instruction_data = [
        {
            "instruction": "解释什么是人工智能",
            "response": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。"
        },
        {
            "instruction": "什么是机器学习？",
            "response": "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。通过分析数据，机器学习算法可以识别模式并做出预测或决策。"
        },
        {
            "instruction": "深度学习与传统机器学习有什么区别？",
            "response": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。与传统机器学习相比，深度学习能够自动学习特征，而传统方法通常需要手动特征工程。"
        }
    ]
    
    with open(data_dir / "instruction_data.json", "w", encoding="utf-8") as f:
        json.dump(instruction_data, f, ensure_ascii=False, indent=2)
    
    # 3. 对话数据
    conversation_data = [
        [
            {"role": "user", "content": "你好，请介绍一下你自己"},
            {"role": "assistant", "content": "你好！我是NovaMind AI助手，一个基于先进人工智能技术开发的智能对话系统。我可以帮助您回答问题、提供信息、协助写作等多种任务。"}
        ],
        [
            {"role": "user", "content": "你能做什么？"},
            {"role": "assistant", "content": "我可以进行自然语言对话、回答问题、协助写作、代码生成、数据分析等多种任务。我会尽力为您提供准确、有用的帮助。"}
        ],
        [
            {"role": "user", "content": "请解释一下机器学习的基本概念"},
            {"role": "assistant", "content": "机器学习是人工智能的一个核心分支，它使计算机能够从数据中学习并改进性能，而无需明确编程。主要包括监督学习、无监督学习和强化学习三种类型。"}
        ]
    ]
    
    with open(data_dir / "conversation_data.json", "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
    
    print("示例数据已创建在 ./sample_data 目录中")
    return data_dir


async def example_basic_training():
    """示例1：基础语言模型训练"""
    print("\n=== 示例1：基础语言模型训练 ===")
    
    # 创建训练配置
    config = LLMTrainingConfig(
        model_name="basic_lm_training",
        model_type="llm",
        model_path="microsoft/DialoGPT-small",  # 使用小型模型作为示例
        dataset_path="./sample_data/text_data.json",
        output_dir="./outputs/basic_training",
        batch_size=4,
        learning_rate=5e-5,
        num_epochs=3,
        max_length=128,
        log_interval=5,
        eval_interval=10,
        save_interval=20
    )
    
    # 创建训练器
    trainer = LLMTrainer(config)
    
    # 添加监控回调
    trainer.add_callback(MetricsCallback("basic_training"))
    
    # 启动训练
    training_id = training_manager.start_training("basic_training", trainer)
    print(f"基础训练已启动，ID: {training_id}")
    
    # 等待训练完成
    while True:
        status = training_manager.get_training_status(training_id)
        if status and status['status'] in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
            print(f"训练完成，状态: {status['status']}")
            break
        await asyncio.sleep(5)


async def example_instruction_tuning():
    """示例2：指令微调训练"""
    print("\n=== 示例2：指令微调训练 ===")
    
    # 创建训练配置
    config = LLMTrainingConfig(
        model_name="instruction_tuning",
        model_type="llm",
        model_path="microsoft/DialoGPT-small",
        dataset_path="./sample_data/instruction_data.json",
        output_dir="./outputs/instruction_tuning",
        batch_size=2,
        learning_rate=3e-5,
        num_epochs=2,
        max_length=256,
        instruction_tuning=True,
        log_interval=3,
        eval_interval=5,
        save_interval=10
    )
    
    # 创建训练器
    trainer = LLMTrainer(config)
    
    # 添加监控回调
    trainer.add_callback(MetricsCallback("instruction_tuning"))
    
    # 启动训练
    training_id = training_manager.start_training("instruction_tuning", trainer)
    print(f"指令微调训练已启动，ID: {training_id}")
    
    # 等待训练完成
    while True:
        status = training_manager.get_training_status(training_id)
        if status and status['status'] in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
            print(f"训练完成，状态: {status['status']}")
            break
        await asyncio.sleep(5)


async def example_conversation_training():
    """示例3：对话模型训练"""
    print("\n=== 示例3：对话模型训练 ===")
    
    # 创建训练配置
    config = LLMTrainingConfig(
        model_name="conversation_training",
        model_type="llm",
        model_path="microsoft/DialoGPT-small",
        dataset_path="./sample_data/conversation_data.json",
        output_dir="./outputs/conversation_training",
        batch_size=2,
        learning_rate=2e-5,
        num_epochs=2,
        max_length=512,
        chat_template="chatml",
        log_interval=3,
        eval_interval=5,
        save_interval=10
    )
    
    # 创建训练器
    trainer = LLMTrainer(config)
    
    # 添加监控回调
    trainer.add_callback(MetricsCallback("conversation_training"))
    
    # 启动训练
    training_id = training_manager.start_training("conversation_training", trainer)
    print(f"对话训练已启动，ID: {training_id}")
    
    # 等待训练完成
    while True:
        status = training_manager.get_training_status(training_id)
        if status and status['status'] in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
            print(f"训练完成，状态: {status['status']}")
            break
        await asyncio.sleep(5)


async def example_lora_training():
    """示例4：LoRA微调训练"""
    print("\n=== 示例4：LoRA微调训练 ===")
    
    # 创建LoRA配置
    lora_config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 创建训练配置
    config = LLMTrainingConfig(
        model_name="lora_training",
        model_type="llm",
        model_path="microsoft/DialoGPT-small",
        dataset_path="./sample_data/instruction_data.json",
        output_dir="./outputs/lora_training",
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=2,
        max_length=256,
        instruction_tuning=True,
        use_lora=True,
        lora_config=lora_config,
        log_interval=3,
        eval_interval=5,
        save_interval=10
    )
    
    # 创建训练器
    trainer = LLMTrainer(config)
    
    # 添加监控回调
    trainer.add_callback(MetricsCallback("lora_training"))
    
    # 启动训练
    training_id = training_manager.start_training("lora_training", trainer)
    print(f"LoRA训练已启动，ID: {training_id}")
    
    # 等待训练完成
    while True:
        status = training_manager.get_training_status(training_id)
        if status and status['status'] in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
            print(f"训练完成，状态: {status['status']}")
            break
        await asyncio.sleep(5)


def example_text_generation():
    """示例5：文本生成测试"""
    print("\n=== 示例5：文本生成测试 ===")
    
    # 加载训练好的模型（这里使用示例）
    config = LLMTrainingConfig(
        model_name="test_generation",
        model_type="llm",
        model_path="microsoft/DialoGPT-small",
        dataset_path="./sample_data/text_data.json",
        output_dir="./outputs/test_generation"
    )
    
    trainer = LLMTrainer(config)
    
    # 测试提示
    test_prompts = [
        "人工智能的未来发展",
        "机器学习的应用领域",
        "深度学习的技术优势"
    ]
    
    print("生成文本示例：")
    for prompt in test_prompts:
        generated = trainer.generate_text(prompt, max_length=100, temperature=0.7)
        print(f"提示: {prompt}")
        print(f"生成: {generated}")
        print("-" * 50)


def example_monitoring():
    """示例6：监控系统演示"""
    print("\n=== 示例6：监控系统演示 ===")
    
    # 启动监控服务器
    training_monitor.start_background()
    print("监控服务器已启动，访问 http://localhost:8080 查看仪表板")
    
    # 显示训练列表
    trainings = training_manager.list_trainings()
    print(f"当前训练任务数量: {len(trainings)}")
    
    for training in trainings:
        print(f"训练ID: {training['id']}")
        print(f"状态: {training['status']['status']}")
        print(f"活跃: {training['active']}")
        print("-" * 30)


async def main():
    """主函数"""
    print("🚀 NovaMind训练框架示例")
    print("=" * 50)
    
    # 创建示例数据
    create_sample_data()
    
    # 启动监控系统
    training_monitor.start_background()
    print("监控系统已启动: http://localhost:8080")
    
    try:
        # 运行各种训练示例
        await example_basic_training()
        await example_instruction_tuning()
        await example_conversation_training()
        await example_lora_training()
        
        # 文本生成测试
        example_text_generation()
        
        # 监控演示
        example_monitoring()
        
        print("\n✅ 所有示例运行完成！")
        print("访问 http://localhost:8080 查看训练监控仪表板")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
    
    # 保持监控服务器运行
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 程序已退出")


if __name__ == "__main__":
    asyncio.run(main()) 