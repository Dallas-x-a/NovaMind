# Novamind: 下一代多智能体与大模型应用框架



> **Novamind** 是面向未来的多智能体与大模型应用开发框架，专注于高效、可扩展、模块化的智能体系统与AI工具链，助力学术研究与产业落地。

---

## 🌟 框架设计理念

- **模块化与可扩展性**：核心功能高度解耦，支持灵活组合与自定义扩展。
- **多大模型原生支持**：内置OpenAI、Claude、Qwen、ERNIE、Llama、DeepSeek、MiniMax、GLM、Yi等主流大模型接口。
- **丰富的智能体与工具生态**：涵盖RAG、信息抽取、代码生成、联网搜索、知识图谱等多场景。
- **专业级数据生成与训练**：支持大规模数据生成、微调、评测与知识存储。
- **工程级最佳实践**：官方示例丰富，文档详尽，易于集成与二次开发。

---

## 🚀 快速安装

```bash
pip install novamind
# 或开发版
pip install -e ".[dev]"
```

---

## 🏁 快速上手示例

```python
from novamind.core.agent import Agent
from novamind.core.environment import Environment
from novamind.tools.web_search import WebSearchTool

agent = Agent(
    name="researcher",
    model="gpt-4o",
    tools=[WebSearchTool()]
)
response = agent.run("简述量子计算的最新进展")
print(response)
```

---

## 🧩 主要模块与目录结构

```
novamind/
├── core/         # 核心Agent、环境、模型等
├── mcp/          # 通用大模型能力包（摘要、代码生成、问答、信息抽取、联网搜索等）
├── downloads/    # 主流大模型一键下载脚本与说明
├── examples/     # 官方示例（多模型chat、摘要、数据生成、知识图谱等）
├── models/       # 大模型API/SDK集成
├── tools/        # 智能体可用工具
├── datagen/      # 数据生成与增强
├── storages/     # 存储与知识库
├── train/        # 训练与微调脚本
├── tests/        # 单元测试
└── ...
```

---

## 🔥 官方示例（examples/）

- **多大模型Chat**：
  - `openai_chat.py`、`anthropic_chat.py`、`qwen_chat.py`、`deepseek_chat.py`、`llama_chat.py`、`ernie_chat.py`、`glm_chat.py`、`minimax_chat.py`、`yichat_chat.py`
- **通用能力演示**：
  - `text_summarization.py`（多模型摘要）
  - `data_generation.py`（数据生成与分析）
  - `basic_agent.py`（基础智能体）
  - `pdf2kg.py`（PDF内容抽取到Neo4j知识图谱）

---

## 📦 大模型下载与集成（downloads/）

- `huggingface_download.py`：通用HuggingFace模型下载脚本，支持镜像、断点续传
- `openai_download.md`、`llama_download.md`、`qwen_download.md`、`deepseek_download.md`、`ernie_download.md`、`glm_download.md`、`minimax_download.md`、`yichat_download.md`：各大模型下载与API调用说明

---

## 🛠️ 通用大模型能力包（mcp/）

- `text_summarizer.py`：多模型文本摘要
- `text2code.py`：文本转代码
- `qa_tool.py`：通用问答
- `information_extractor.py`：信息抽取
- `web_search_tool.py`：联网搜索

---

## 📚 文档与Cookbook

- [官方文档站点](https://novamind.readthedocs.io/)
- [Cookbook与用例](docs/cookbook.md)（持续丰富中）

---

## 🧪 单元测试

所有核心模块均配有测试用例，见 `tests/` 目录。

---

## 🤝 贡献与社区

我们欢迎任何形式的贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

- [GitHub Issues](https://github.com/yourusername/novamind/issues)
- [Discussions](https://github.com/yourusername/novamind/discussions)
- [社区交流群/Discord/微信群](#)

---

## 📄 许可证

本项目采用 Apache License 2.0 许可证，详见 [LICENSE](LICENSE)。

---

## 📖 引用

```bibtex
@software{novamind2025,
  author = {Novamind Team},
  title = {Novamind: 下一代多智能体与大模型应用框架},
  year = {2025},
  url = {https://github.com/yourusername/novamind}
}
```

---

> Novamind 致力于推动智能体与大模型技术的开放创新，欢迎学术与产业合作！ 