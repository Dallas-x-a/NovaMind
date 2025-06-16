# Yi大模型（零一万物）下载说明

## 1. HuggingFace下载（推荐）
- 适用Yi-6B/9B/34B/200B等全系列
- 需有HuggingFace账号

示例：
```bash
pip install huggingface_hub
python huggingface_download.py --repo 01-ai/Yi-34B-Chat --out yi-34b-chat
```

## 2. 国内镜像加速
- 使用 `--mirror` 参数自动切换到 https://hf-mirror.com
```bash
python huggingface_download.py --repo 01-ai/Yi-34B-Chat --mirror --out yi-34b-chat
```

## 3. 官方直链
- 访问 https://huggingface.co/01-ai
- 按官方说明下载

## 4. 常见问题
- 下载需科学上网或使用国内镜像
- 权重文件较大，建议断点续传 