"""通用文本摘要工具，支持多种大模型。"""
import os
from typing import Optional
from novamind.core.agent import Agent
from novamind.core.environment import Environment
from dotenv import load_dotenv

class TextSummarizer:
    def __init__(self, model, min_length: int = 50, max_length: int = 200, style: str = "concise"):
        self.model = model
        self.agent = Agent(name="summarizer", model=model, description="通用文本摘要工具")
        self.min_length = min_length
        self.max_length = max_length
        self.style = style

    async def summarize(self, text: str) -> str:
        prompt = f"请对以下文本进行摘要，要求：风格{self.style}，长度{self.min_length}-{self.max_length}字。\n原文：{text}\n摘要："
        env = Environment(variables={"temperature": 0.7}, constraints={"max_tokens": self.max_length * 2, "timeout": 30})
        response = await self.agent.run(prompt, environment=env)
        return response["response"].content.strip() 