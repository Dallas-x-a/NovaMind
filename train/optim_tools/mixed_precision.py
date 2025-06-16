"""
混合精度训练工具
------------------
支持自动切换fp16/bf16混合精度训练，适配transformers和pytorch训练流程。

用法示例：
    from optim_tools.mixed_precision import set_mixed_precision
    set_mixed_precision(model, use_bf16=True)

作者：NovaMind
"""
import torch

def set_mixed_precision(model, use_bf16=False):
    """
    设置模型为混合精度训练模式。
    参数：
        model: 需要设置的pytorch模型
        use_bf16: 是否使用bfloat16（否则为fp16）
    返回：
        None
    """
    if use_bf16:
        dtype = torch.bfloat16
        print("[混合精度] 使用bfloat16训练模式")
    else:
        dtype = torch.float16
        print("[混合精度] 使用fp16训练模式")
    
    for param in model.parameters():
        param.data = param.data.to(dtype)
    if hasattr(model, 'to'):
        model.to(dtype)
    # 注意：transformers的Trainer等框架通常通过TrainingArguments自动控制混合精度
    # 这里为自定义训练流程提供手动切换支持 