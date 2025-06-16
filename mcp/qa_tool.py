"""通用问答工具，支持多种大模型。"""
from novamind.core.agent import Agent
from novamind.core.environment import Environment

class QATool:
    def __init__(self, model):
        self.model = model
        self.agent = Agent(name="qa_tool", model=model, description="通用问答工具")

    async def ask(self, question: str, context: str = "") -> str:
        prompt = f"请根据以下上下文回答问题。\n上下文：{context}\n问题：{question}\n答案："
        env = Environment(variables={"temperature": 0.3}, constraints={"max_tokens": 512, "timeout": 30})
        response = await self.agent.run(prompt, environment=env)
        return response["response"].content.strip() 