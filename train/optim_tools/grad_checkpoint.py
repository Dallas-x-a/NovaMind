"""
梯度检查点工具
------------------
用于大模型训练时节省显存，适配pytorch和transformers。

用法示例：
    from optim_tools.grad_checkpoint import enable_gradient_checkpointing
    enable_gradient_checkpointing(model)

作者：NovaMind
"""

def enable_gradient_checkpointing(model):
    """
    启用模型的梯度检查点功能（节省显存）。
    参数：
        model: 支持gradient_checkpointing_enable()方法的模型
    返回：
        None
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("[梯度检查点] 已启用gradient checkpointing")
    else:
        print("[梯度检查点] 当前模型不支持gradient checkpointing") 