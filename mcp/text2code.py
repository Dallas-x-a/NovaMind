"""通用文本转代码工具，支持多种大模型。"""
from novamind.core.agent import Agent
from novamind.core.environment import Environment

class Text2Code:
    def __init__(self, model, language: str = "python"):
        self.model = model
        self.language = language
        self.agent = Agent(name="text2code", model=model, description="通用文本转代码工具")

    async def generate_code(self, instruction: str) -> str:
        prompt = f"请用{self.language}实现如下需求：\n{instruction}\n代码："
        env = Environment(variables={"temperature": 0.2}, constraints={"max_tokens": 1024, "timeout": 60})
        response = await self.agent.run(prompt, environment=env)
        return response["response"].content.strip() 