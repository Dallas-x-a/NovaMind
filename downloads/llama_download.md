# Llama2/3 模型下载说明

## 1. HuggingFace下载（推荐）
- 需在Meta官网申请并同意协议，获得下载权限
- 适用Llama2/3全系列

示例：
```bash
pip install huggingface_hub
python huggingface_download.py --repo meta-llama/Meta-Llama-3-8B-Instruct --out llama3-8b
```

## 2. 国内镜像加速
- 使用 `--mirror` 参数自动切换到 https://hf-mirror.com
```bash
python huggingface_download.py --repo meta-llama/Meta-Llama-3-8B-Instruct --mirror --out llama3-8b
```

## 3. Meta官网直链
- 访问 https://ai.meta.com/resources/models-and-libraries/llama-downloads/
- 填写表单，获得下载链接

## 4. 常见问题
- 下载需科学上网或使用国内镜像
- 权重文件较大，建议断点续传
- 需有HuggingFace账号并同意Meta协议 