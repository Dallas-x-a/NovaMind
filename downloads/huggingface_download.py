"""
通用 HuggingFace 大模型下载脚本
支持：
- 官方与国内镜像（如hf-mirror.com）
- 断点续传
- 指定模型版本、分支、文件过滤
- 适用Llama、Mistral、Baichuan、Qwen等主流模型

依赖：
    pip install huggingface_hub

用法示例：
    python huggingface_download.py --repo meta-llama/Meta-Llama-3-8B-Instruct --out llama3-8b
    python huggingface_download.py --repo Qwen/Qwen1.5-7B-Chat --mirror --out qwen1.5-7b

常见问题：
- 国内建议加 --mirror 参数，自动切换到 https://hf-mirror.com
- 需有HuggingFace账号并同意部分模型协议
"""
import argparse
import os
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="HuggingFace模型下载工具")
    parser.add_argument('--repo', type=str, required=True, help='模型仓库名，如 meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--out', type=str, default=None, help='本地保存目录')
    parser.add_argument('--mirror', action='store_true', help='使用hf-mirror.com国内镜像')
    parser.add_argument('--revision', type=str, default=None, help='指定分支/commit/tag')
    parser.add_argument('--resume', action='store_true', help='断点续传')
    parser.add_argument('--allow_patterns', type=str, nargs='*', help='只下载指定文件/文件夹')
    args = parser.parse_args()

    if args.mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print('已切换到hf-mirror.com镜像')

    print(f"开始下载: {args.repo}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=args.out or args.repo.replace('/', '_'),
        local_dir_use_symlinks=False,
        resume_download=args.resume,
        revision=args.revision,
        allow_patterns=args.allow_patterns
    )
    print('下载完成！')

if __name__ == "__main__":
    main() 