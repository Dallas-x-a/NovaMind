# 通义千问 Qwen 模型下载说明

## 1. HuggingFace下载（推荐）
- 适用Qwen1.5、Qwen2等全系列
- 需有HuggingFace账号

示例：
```bash
pip install huggingface_hub
python huggingface_download.py --repo Qwen/Qwen1.5-7B-Chat --out qwen1.5-7b
```

## 2. 国内镜像加速
- 使用 `--mirror` 参数自动切换到 https://hf-mirror.com
```bash
python huggingface_download.py --repo Qwen/Qwen1.5-7B-Chat --mirror --out qwen1.5-7b
```

## 3. 阿里云ModelScope
- 访问 https://modelscope.cn/models?q=qwen
- 支持Web页面下载和API下载

## 4. 常见问题
- Qwen部分模型需申请商用授权
- 下载需科学上网或使用国内镜像
- 权重文件较大，建议断点续传 