"""
NovaMind训练框架优势演示

展示NovaMind训练框架相比其他框架（如LangChain）的独特优势：
1. 实时监控和可视化
2. 智能参数调优
3. 多模态训练支持
4. 分布式训练
5. 模型版本管理
6. 实验管理
7. 生产就绪
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List

from .trainer import TrainingManager, training_manager
from .llm_trainer import LLMTrainer
from .config import (
    get_training_config,
    get_lora_config,
    get_instruction_config,
    list_available_presets
)
from .monitor import training_monitor, MetricsCallback


class NovaMindAdvantagesDemo:
    """NovaMind训练框架优势演示类"""
    
    def __init__(self):
        """初始化演示"""
        self.training_manager = training_manager
        self.monitor = training_monitor
        self.demo_results = {}
    
    def demonstrate_advantage_1_realtime_monitoring(self):
        """优势1：实时监控和可视化"""
        print("🎯 优势1：实时监控和可视化")
        print("=" * 50)
        
        # 启动监控服务器
        self.monitor.start_background()
        print("✅ 监控服务器已启动: http://localhost:8080")
        print("   实时查看训练状态、损失曲线、系统资源")
        print("   支持WebSocket实时数据推送")
        print("   提供交互式图表和仪表板")
        print()
        
        # 创建示例训练
        config = get_training_config(
            model_preset="dialo-gpt-small",
            training_preset="quick",
            dataset_path="./data/sample_data.json",
            training_name="demo_monitoring"
        )
        
        trainer = LLMTrainer(config)
        trainer.add_callback(MetricsCallback("demo_monitoring"))
        
        # 启动训练
        training_id = self.training_manager.start_training("demo_monitoring", trainer)
        print(f"✅ 演示训练已启动: {training_id}")
        print("   访问 http://localhost:8080 查看实时监控")
        print()
        
        return training_id
    
    def demonstrate_advantage_2_smart_parameter_tuning(self):
        """优势2：智能参数调优"""
        print("🧠 优势2：智能参数调优")
        print("=" * 50)
        
        # 展示预设配置系统
        presets = list_available_presets()
        print("📋 智能预设配置系统:")
        print(f"   模型预设: {len(presets['model_presets'])} 种")
        print(f"   训练预设: {len(presets['training_presets'])} 种")
        print(f"   LoRA预设: {len(presets['lora_presets'])} 种")
        print()
        
        # 展示自动参数调优
        print("⚡ 自动参数调优功能:")
        print("   - 基于性能自动调整学习率")
        print("   - 智能早停机制")
        print("   - 动态批次大小调整")
        print("   - 自适应梯度裁剪")
        print()
        
        # 创建多个配置进行对比
        configs = {
            "quick": get_training_config("dialo-gpt-small", "quick", "./data/sample_data.json", "demo_quick"),
            "standard": get_training_config("dialo-gpt-small", "standard", "./data/sample_data.json", "demo_standard"),
            "high_quality": get_training_config("dialo-gpt-small", "high_quality", "./data/sample_data.json", "demo_quality")
        }
        
        print("🔬 配置对比实验:")
        for name, config in configs.items():
            print(f"   {name}: LR={config.learning_rate}, BS={config.batch_size}, Epochs={config.num_epochs}")
        print()
        
        return configs
    
    def demonstrate_advantage_3_multimodal_support(self):
        """优势3：多模态训练支持"""
        print("🎨 优势3：多模态训练支持")
        print("=" * 50)
        
        print("📊 支持的多模态类型:")
        print("   - 文本模型 (LLM)")
        print("   - 视觉模型 (Vision)")
        print("   - 音频模型 (Audio)")
        print("   - 多模态模型 (Multimodal)")
        print()
        
        print("🔄 统一训练接口:")
        print("   - 相同的配置系统")
        print("   - 统一的监控界面")
        print("   - 一致的API接口")
        print("   - 跨模态实验管理")
        print()
        
        # 展示不同模态的配置
        multimodal_configs = {
            "text": get_training_config("dialo-gpt-small", "standard", "./data/text_data.json", "demo_text"),
            "vision": get_training_config("vision-model", "standard", "./data/vision_data.json", "demo_vision"),
            "audio": get_training_config("audio-model", "standard", "./data/audio_data.json", "demo_audio")
        }
        
        print("🎯 多模态配置示例:")
        for modality, config in multimodal_configs.items():
            print(f"   {modality}: {config.model_type} - {config.model_path}")
        print()
        
        return multimodal_configs
    
    def demonstrate_advantage_4_distributed_training(self):
        """优势4：分布式训练"""
        print("⚡ 优势4：分布式训练")
        print("=" * 50)
        
        print("🌐 分布式训练特性:")
        print("   - 多GPU自动并行")
        print("   - 多节点集群支持")
        print("   - 自动负载均衡")
        print("   - 故障恢复机制")
        print("   - 通信优化")
        print()
        
        # 展示分布式配置
        distributed_config = get_training_config(
            model_preset="llama-7b",
            training_preset="high_quality",
            dataset_path="./data/large_dataset.json",
            training_name="demo_distributed"
        )
        distributed_config.distributed = True
        distributed_config.num_gpus = 4
        
        print("🔧 分布式配置示例:")
        print(f"   分布式模式: {distributed_config.distributed}")
        print(f"   GPU数量: {distributed_config.num_gpus}")
        print(f"   批次大小: {distributed_config.batch_size}")
        print(f"   梯度累积: {distributed_config.gradient_accumulation_steps}")
        print()
        
        return distributed_config
    
    def demonstrate_advantage_5_model_versioning(self):
        """优势5：模型版本管理"""
        print("📦 优势5：模型版本管理")
        print("=" * 50)
        
        print("🔄 版本管理功能:")
        print("   - 自动版本号生成")
        print("   - 模型检查点保存")
        print("   - 版本回滚支持")
        print("   - 模型对比分析")
        print("   - 生产部署管理")
        print()
        
        # 创建多个版本的训练
        versions = ["v1.0", "v1.1", "v2.0"]
        version_configs = {}
        
        for version in versions:
            config = get_training_config(
                model_preset="dialo-gpt-small",
                training_preset="standard",
                dataset_path=f"./data/version_{version}.json",
                training_name=f"demo_version_{version}"
            )
            version_configs[version] = config
        
        print("📋 版本管理示例:")
        for version, config in version_configs.items():
            print(f"   {version}: {config.model_name} -> {config.output_dir}")
        print()
        
        return version_configs
    
    def demonstrate_advantage_6_experiment_management(self):
        """优势6：实验管理"""
        print("🔬 优势6：实验管理")
        print("=" * 50)
        
        print("📊 实验管理功能:")
        print("   - A/B测试支持")
        print("   - 实验对比分析")
        print("   - 结果可视化")
        print("   - 实验追踪")
        print("   - 最佳实践推荐")
        print()
        
        # 创建实验配置
        experiments = {
            "baseline": get_training_config("dialo-gpt-small", "standard", "./data/baseline.json", "exp_baseline"),
            "lora_8": get_lora_config("dialo-gpt-small", "./data/lora_data.json", "exp_lora_8", "efficient"),
            "lora_16": get_lora_config("dialo-gpt-small", "./data/lora_data.json", "exp_lora_16", "standard"),
            "instruction": get_instruction_config("dialo-gpt-small", "./data/instruction_data.json", "exp_instruction")
        }
        
        print("🧪 实验配置示例:")
        for name, config in experiments.items():
            lora_info = f" (LoRA r={config.lora_config.r})" if config.use_lora else ""
            instruction_info = " (指令微调)" if config.instruction_tuning else ""
            print(f"   {name}: {config.model_name}{lora_info}{instruction_info}")
        print()
        
        return experiments
    
    def demonstrate_advantage_7_production_ready(self):
        """优势7：生产就绪"""
        print("🏭 优势7：生产就绪")
        print("=" * 50)
        
        print("🚀 生产级特性:")
        print("   - 企业级安全")
        print("   - 高可用性")
        print("   - 性能优化")
        print("   - 监控告警")
        print("   - 自动部署")
        print("   - 容器化支持")
        print()
        
        # 展示生产配置
        production_config = get_training_config(
            model_preset="llama-7b",
            training_preset="high_quality",
            dataset_path="./data/production_data.json",
            training_name="production_model"
        )
        
        # 添加生产级配置
        production_config.enable_wandb = True
        production_config.gradient_checkpointing = True
        production_config.mixed_precision = True
        production_config.flash_attention = True
        
        print("⚙️ 生产配置示例:")
        print(f"   混合精度: {production_config.mixed_precision}")
        print(f"   梯度检查点: {production_config.gradient_checkpointing}")
        print(f"   Flash Attention: {production_config.flash_attention}")
        print(f"   WandB集成: {production_config.enable_wandb}")
        print()
        
        return production_config
    
    def compare_with_langchain(self):
        """与LangChain对比"""
        print("🆚 与LangChain训练框架对比")
        print("=" * 50)
        
        comparison = {
            "实时监控": {
                "NovaMind": "✅ Web界面实时监控，WebSocket推送",
                "LangChain": "❌ 基础日志输出"
            },
            "参数调优": {
                "NovaMind": "✅ 智能预设系统，自动调优",
                "LangChain": "❌ 手动配置，无自动调优"
            },
            "多模态支持": {
                "NovaMind": "✅ 统一接口，多模态原生支持",
                "LangChain": "❌ 主要针对文本，多模态支持有限"
            },
            "分布式训练": {
                "NovaMind": "✅ 原生分布式支持，自动负载均衡",
                "LangChain": "❌ 需要额外配置，支持有限"
            },
            "版本管理": {
                "NovaMind": "✅ 完整版本管理，自动检查点",
                "LangChain": "❌ 基础保存功能"
            },
            "实验管理": {
                "NovaMind": "✅ A/B测试，实验对比分析",
                "LangChain": "❌ 无内置实验管理"
            },
            "生产就绪": {
                "NovaMind": "✅ 企业级特性，容器化部署",
                "LangChain": "❌ 主要面向研究，生产支持有限"
            }
        }
        
        for feature, comparison_data in comparison.items():
            print(f"{feature}:")
            print(f"   NovaMind: {comparison_data['NovaMind']}")
            print(f"   LangChain: {comparison_data['LangChain']}")
            print()
    
    def run_comprehensive_demo(self):
        """运行综合演示"""
        print("🚀 NovaMind训练框架优势综合演示")
        print("=" * 60)
        print()
        
        try:
            # 演示各个优势
            self.demonstrate_advantage_1_realtime_monitoring()
            time.sleep(2)
            
            self.demonstrate_advantage_2_smart_parameter_tuning()
            time.sleep(2)
            
            self.demonstrate_advantage_3_multimodal_support()
            time.sleep(2)
            
            self.demonstrate_advantage_4_distributed_training()
            time.sleep(2)
            
            self.demonstrate_advantage_5_model_versioning()
            time.sleep(2)
            
            self.demonstrate_advantage_6_experiment_management()
            time.sleep(2)
            
            self.demonstrate_advantage_7_production_ready()
            time.sleep(2)
            
            # 与LangChain对比
            self.compare_with_langchain()
            
            print("🎉 优势演示完成！")
            print("📖 更多详细信息请查看文档和示例代码")
            print("🌐 访问 http://localhost:8080 查看实时监控")
            
        except Exception as e:
            print(f"❌ 演示过程中出错: {e}")
            import traceback
            traceback.print_exc()


def create_demo_data():
    """创建演示数据"""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # 创建示例数据文件
    sample_data = [
        {"text": "这是一个示例文本数据。"},
        {"text": "用于演示NovaMind训练框架。"},
        {"text": "支持多种训练模式和配置。"}
    ]
    
    import json
    with open(data_dir / "sample_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 演示数据已创建")


async def main():
    """主函数"""
    print("🎯 NovaMind训练框架优势演示")
    print("=" * 60)
    
    # 创建演示数据
    create_demo_data()
    
    # 创建演示实例
    demo = NovaMindAdvantagesDemo()
    
    # 运行综合演示
    demo.run_comprehensive_demo()
    
    # 保持监控服务器运行
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 演示结束")


if __name__ == "__main__":
    asyncio.run(main()) 