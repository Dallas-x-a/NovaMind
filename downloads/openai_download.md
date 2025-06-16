# OpenAI GPT 模型下载说明

## 1. API方式（推荐）
OpenAI GPT-3/4/4o等模型官方仅支持API调用，不开放权重下载。
- 需注册OpenAI账号并充值
- 使用 `openai` Python SDK

示例：
```python
import openai
openai.api_key = 'YOUR_OPENAI_API_KEY'
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "你好，介绍一下人工智能。"}]
)
print(response['choices'][0]['message']['content'])
```

## 2. 本地权重下载（第三方）
OpenAI官方不开放GPT-3/4权重。部分社区有GPT-2权重可下载：
- [GPT-2 (OpenAI官方)](https://github.com/openai/gpt-2)
- [GPT-2 HuggingFace](https://huggingface.co/openai-gpt)

下载示例：
```bash
pip install transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```

## 3. 其他说明
- GPT-3/4/4o等大模型暂无法本地部署，仅能API调用。
- 若需本地部署，建议使用Llama、Qwen、Baichuan等开源大模型。 