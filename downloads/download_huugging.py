import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# 下载整个库
snapshot_download(repo_id="meta-llama/meta-Llama-3.1-8B-Instruct", local_dir='models/sentence-transformer', local_dir_use_symlinks=False, resume_download=True)