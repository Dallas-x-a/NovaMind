"""
梯度累积与裁剪工具
------------------
支持自定义训练循环中的梯度累积与梯度裁剪，适配pytorch训练流程。

用法示例：
    from optim_tools.grad_accum_clip import backward_with_accumulation, clip_gradients
    backward_with_accumulation(loss, optimizer, step, accumulation_steps)
    clip_gradients(model, max_norm=1.0)

作者：NovaMind
"""
import torch

def backward_with_accumulation(loss, optimizer, step, accumulation_steps):
    """
    梯度累积反向传播
    参数：
        loss: 当前batch的损失
        optimizer: 优化器
        step: 当前全局步数
        accumulation_steps: 累积步数
    返回：
        None
    """
    loss = loss / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

def clip_gradients(model, max_norm=1.0):
    """
    对模型参数进行梯度裁剪，防止梯度爆炸。
    参数：
        model: 需要裁剪梯度的模型
        max_norm: 最大梯度范数
    返回：
        None
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) 