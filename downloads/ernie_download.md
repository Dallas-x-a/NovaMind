# 百度文心ERNIE 模型下载说明

## 1. HuggingFace下载（推荐）
- 适用ERNIE全系列（ERNIE 3.0/4.0、ERNIE-Bot等）
- 需有HuggingFace账号

示例：
```bash
pip install huggingface_hub
python huggingface_download.py --repo baidu/ERNIE-4.0-EN-7B --out ernie-4.0-7b
```

## 2. 国内镜像加速
- 使用 `--mirror` 参数自动切换到 https://hf-mirror.com
```bash
python huggingface_download.py --repo baidu/ERNIE-4.0-EN-7B --mirror --out ernie-4.0-7b
```

## 3. 百度AI Studio
- 访问 https://aistudio.baidu.com/aistudio/modellist
- 支持Web页面下载和API下载

## 4. 常见问题
- ERNIE部分模型需申请商用授权
- 下载需科学上网或使用国内镜像
- 权重文件较大，建议断点续传 