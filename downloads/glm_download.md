# 智谱GLM 模型下载说明

## 1. HuggingFace下载（推荐）
- 适用GLM-3/GLM-4等全系列
- 需有HuggingFace账号

示例：
```bash
pip install huggingface_hub
python huggingface_download.py --repo THUDM/glm-4-9b-chat --out glm-4-9b-chat
```

## 2. 国内镜像加速
- 使用 `--mirror` 参数自动切换到 https://hf-mirror.com
```bash
python huggingface_download.py --repo THUDM/glm-4-9b-chat --mirror --out glm-4-9b-chat
```

## 3. 官方直链
- 访问 https://github.com/THUDM/GLM-4
- 按官方说明下载

## 4. 常见问题
- 下载需科学上网或使用国内镜像
- 权重文件较大，建议断点续传 