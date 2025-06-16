"""通用联网搜索工具，支持多种大模型。"""
from novamind.core.agent import Agent
from novamind.core.environment import Environment
from novamind.tools.web_search import WebSearchTool

class WebSearchCaller:
    def __init__(self, model, max_results: int = 3):
        self.model = model
        self.web_search = WebSearchTool(max_results=max_results)
        self.agent = Agent(name="web_search_caller", model=model, tools=[self.web_search], description="通用联网搜索工具")

    async def search_and_ask(self, query: str) -> str:
        prompt = f"请根据联网搜索结果回答以下问题：{query}"
        env = Environment(variables={"temperature": 0.5}, constraints={"max_tokens": 512, "timeout": 40})
        response = await self.agent.run(prompt, environment=env)
        return response["response"].content.strip() 