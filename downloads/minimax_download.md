# MiniMax 大模型下载说明

## 1. API方式（推荐）
- MiniMax官方仅支持API调用，不开放权重下载
- 需注册MiniMax账号并获取API Key

示例：
```python
import requests
url = 'https://api.minimax.chat/v1/text/chatcompletion_pro'
headers = {'Authorization': 'Bearer YOUR_MINIMAX_API_KEY'}
data = {
    'model': 'abab5.5s-chat',
    'messages': [{'role': 'user', 'content': '你好，介绍一下人工智能。'}]
}
resp = requests.post(url, headers=headers, json=data)
print(resp.json())
```

## 2. HuggingFace/本地权重
- 暂无官方权重开放
- 可关注社区动态

## 3. 其他说明
- 仅API方式可用，暂无法本地部署 