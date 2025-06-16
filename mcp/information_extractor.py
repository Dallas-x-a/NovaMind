"""通用信息抽取工具，支持多种大模型。"""
from novamind.core.agent import Agent
from novamind.core.environment import Environment
from typing import List

class InformationExtractor:
    def __init__(self, model, fields: List[str]):
        self.model = model
        self.fields = fields
        self.agent = Agent(name="info_extractor", model=model, description="通用信息抽取工具")

    async def extract(self, text: str) -> str:
        fields_str = ', '.join(self.fields)
        prompt = f"请从以下文本中提取如下字段：{fields_str}。\n文本：{text}\n提取结果："
        env = Environment(variables={"temperature": 0.1}, constraints={"max_tokens": 512, "timeout": 30})
        response = await self.agent.run(prompt, environment=env)
        return response["response"].content.strip() 