"""OpenAI GPT 对话示例。"""
import asyncio
import os
from dotenv import load_dotenv
from novamind.models.openai import OpenAIModel, OpenAIConfig
from novamind.core.agent import Agent
from novamind.core.environment import Environment

async def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAIModel(OpenAIConfig(model_name="gpt-3.5-turbo", api_key=api_key))
    agent = Agent(name="openai_chat", model=model, description="OpenAI对话助手")
    env = Environment(variables={"temperature": 0.7}, constraints={"max_tokens": 512, "timeout": 30})
    question = "请用简明语言介绍一下人工智能的发展历程。"
    response = await agent.run(question, environment=env)
    print("用户: ", question)
    print("助手: ", response["response"].content)

if __name__ == "__main__":
    asyncio.run(main()) 