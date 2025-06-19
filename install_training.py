#!/usr/bin/env python3
"""
NovaMind训练框架安装脚本

自动检查和安装训练框架所需的依赖包，
并设置必要的环境配置。
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本检查通过: {sys.version}")
    return True


def check_cuda_availability():
    """检查CUDA可用性"""
    print("🔧 检查CUDA可用性...")
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA可用: 版本 {cuda_version}, 设备数量: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {device_name} ({memory:.1f}GB)")
            
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU训练（性能可能受限）")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，无法检查CUDA")
        return False


def install_package(package, upgrade=False):
    """安装Python包"""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {package} 安装成功")
            return True
        else:
            print(f"❌ {package} 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {package} 安装出错: {e}")
        return False


def install_core_dependencies():
    """安装核心依赖"""
    print("📦 安装核心依赖...")
    
    core_packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0"
    ]
    
    success_count = 0
    for package in core_packages:
        if install_package(package):
            success_count += 1
    
    print(f"📊 核心依赖安装完成: {success_count}/{len(core_packages)}")
    return success_count == len(core_packages)


def install_monitoring_dependencies():
    """安装监控依赖"""
    print("📊 安装监控依赖...")
    
    monitoring_packages = [
        "wandb>=0.15.0",
        "plotly>=5.15.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "websockets>=11.0.0"
    ]
    
    success_count = 0
    for package in monitoring_packages:
        if install_package(package):
            success_count += 1
    
    print(f"📊 监控依赖安装完成: {success_count}/{len(monitoring_packages)}")
    return success_count == len(monitoring_packages)


def install_utility_dependencies():
    """安装工具依赖"""
    print("🛠️ 安装工具依赖...")
    
    utility_packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "loguru>=0.7.0",
        "pydantic>=2.0.0",
        "psutil>=5.9.0",
        "tqdm>=4.65.0"
    ]
    
    success_count = 0
    for package in utility_packages:
        if install_package(package):
            success_count += 1
    
    print(f"📊 工具依赖安装完成: {success_count}/{len(utility_packages)}")
    return success_count == len(utility_packages)


def create_directories():
    """创建必要的目录"""
    print("📁 创建目录结构...")
    
    directories = [
        "./data",
        "./outputs",
        "./logs",
        "./models",
        "./checkpoints",
        "./cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")


def create_sample_config():
    """创建示例配置文件"""
    print("⚙️ 创建示例配置...")
    
    config_content = """# NovaMind训练框架示例配置

# 环境配置
environment:
  base_dir: "./novamind"
  data_dir: "./data"
  output_dir: "./outputs"
  log_dir: "./logs"
  model_cache_dir: "./models"
  checkpoint_dir: "./checkpoints"
  monitor_port: 8080
  enable_wandb: true

# 默认训练配置
default_training:
  model_preset: "dialo-gpt-small"
  training_preset: "quick"
  batch_size: 4
  learning_rate: 5e-5
  num_epochs: 3
  max_length: 512

# 监控配置
monitoring:
  log_interval: 10
  eval_interval: 50
  save_interval: 100
  enable_wandb: true
  wandb_project: "novamind-training"
"""
    
    config_path = Path("./training_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print(f"✅ 配置文件已创建: {config_path}")


def test_installation():
    """测试安装"""
    print("🧪 测试安装...")
    
    try:
        # 测试核心模块导入
        import torch
        import transformers
        import datasets
        import wandb
        import plotly
        import fastapi
        
        print("✅ 核心模块导入测试通过")
        
        # 测试CUDA
        if torch.cuda.is_available():
            print("✅ CUDA测试通过")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
        
        # 测试训练框架
        try:
            from training.trainer import BaseTrainer, TrainingConfig
            from training.llm_trainer import LLMTrainer
            from training.monitor import TrainingMonitor
            print("✅ 训练框架模块导入测试通过")
        except ImportError as e:
            print(f"⚠️  训练框架模块导入失败: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入测试失败: {e}")
        return False


def show_next_steps():
    """显示后续步骤"""
    print("\n🎉 安装完成！")
    print("=" * 50)
    print("📖 后续步骤:")
    print("1. 查看快速开始指南:")
    print("   python -m training.quick_start")
    print()
    print("2. 运行训练示例:")
    print("   python -m training.example")
    print()
    print("3. 查看优势演示:")
    print("   python -m training.advantages_demo")
    print()
    print("4. 启动监控服务器:")
    print("   python -c 'from training.monitor import training_monitor; training_monitor.start()'")
    print()
    print("5. 访问监控界面:")
    print("   http://localhost:8080")
    print()
    print("📚 更多信息请查看:")
    print("   - README.md")
    print("   - training/example.py")
    print("   - training/quick_start.py")


def main():
    """主安装函数"""
    print("🚀 NovaMind训练框架安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查CUDA
    check_cuda_availability()
    
    print("\n📦 开始安装依赖...")
    
    # 安装依赖
    core_success = install_core_dependencies()
    monitoring_success = install_monitoring_dependencies()
    utility_success = install_utility_dependencies()
    
    if not (core_success and monitoring_success and utility_success):
        print("❌ 部分依赖安装失败，请检查网络连接或手动安装")
        sys.exit(1)
    
    # 创建目录结构
    create_directories()
    
    # 创建示例配置
    create_sample_config()
    
    # 测试安装
    if not test_installation():
        print("❌ 安装测试失败")
        sys.exit(1)
    
    # 显示后续步骤
    show_next_steps()


if __name__ == "__main__":
    main() 