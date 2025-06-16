"""Novamind 文本摘要示例脚本。

这个示例展示了如何使用 Novamind 进行文本摘要，包括：
1. 使用 OpenAI 模型生成文本摘要
2. 支持多种摘要类型（提取式、生成式）
3. 支持自定义摘要长度和风格
4. 批量处理多个文档
5. 保存和加载摘要结果

使用方法：
1. 确保设置了必要的环境变量（如 OPENAI_API_KEY）
2. 运行脚本：python text_summarization.py
3. 摘要结果将保存在 summaries 目录下
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os

from dotenv import load_dotenv

from ..models.openai import OpenAIModel, OpenAIConfig
from ..core.agent import Agent
from ..core.environment import Environment


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """摘要相关错误的基类。"""
    pass


class ConfigurationError(SummarizationError):
    """配置相关错误。"""
    pass


class GenerationError(SummarizationError):
    """生成过程相关错误。"""
    pass


@dataclass
class SummaryConfig:
    """摘要配置。"""
    style: str = "concise"  # concise, detailed, academic, casual
    max_length: int = 200
    min_length: int = 50
    extractive: bool = False  # True for extractive, False for abstractive


@dataclass
class SummaryResult:
    """摘要结果。"""
    original_text: str
    summary: str
    config: SummaryConfig
    metadata: Dict[str, Any]


class TextSummarizer:
    """文本摘要器。"""

    def __init__(self, model: OpenAIModel):
        """初始化摘要器。

        Args:
            model (OpenAIModel): OpenAI 模型实例
        """
        self.model = model
        self.agent = Agent(
            name="summarizer",
            model=model,
            description="一个能够生成高质量文本摘要的智能助手",
        )

    async def summarize(
        self,
        text: str,
        config: Optional[SummaryConfig] = None
    ) -> SummaryResult:
        """生成文本摘要。

        Args:
            text (str): 要摘要的文本
            config (Optional[SummaryConfig]): 摘要配置

        Returns:
            SummaryResult: 摘要结果

        Raises:
            GenerationError: 当生成过程失败时
        """
        try:
            config = config or SummaryConfig()

            # 构建提示
            style_prompt = {
                "concise": "简洁明了",
                "detailed": "详细全面",
                "academic": "学术严谨",
                "casual": "通俗易懂"
            }.get(config.style, "简洁明了")

            method = "提取关键句子" if config.extractive else "生成新的摘要"

            prompt = f"""请对以下文本进行摘要，要求：
1. 摘要风格：{style_prompt}
2. 摘要方法：{method}
3. 摘要长度：{config.min_length}-{config.max_length}字
4. 保留原文的关键信息和主要观点

原文：
{text}

请生成摘要："""

            # 创建环境
            env = Environment(
                variables={
                    "temperature": 0.7,
                    "max_iterations": 1,
                },
                constraints={
                    "max_tokens": config.max_length * 2,  # 预留足够空间
                    "timeout": 30,
                },
            )

            # 生成摘要
            response = await self.agent.run(prompt, environment=env)
            summary = response["response"].content.strip()

            # 验证摘要长度
            if len(summary) < config.min_length:
                logger.warning(f"摘要长度 ({len(summary)}) 小于最小要求 ({config.min_length})")
            elif len(summary) > config.max_length:
                logger.warning(f"摘要长度 ({len(summary)}) 超过最大限制 ({config.max_length})")
                summary = summary[:config.max_length] + "..."

            return SummaryResult(
                original_text=text,
                summary=summary,
                config=config,
                metadata={
                    "length": len(summary),
                    "style": config.style,
                    "method": "extractive" if config.extractive else "abstractive",
                }
            )

        except Exception as e:
            raise GenerationError(f"生成摘要时出错: {str(e)}")

    async def batch_summarize(
        self,
        texts: List[str],
        config: Optional[SummaryConfig] = None
    ) -> List[SummaryResult]:
        """批量生成文本摘要。

        Args:
            texts (List[str]): 要摘要的文本列表
            config (Optional[SummaryConfig]): 摘要配置

        Returns:
            List[SummaryResult]: 摘要结果列表

        Raises:
            GenerationError: 当生成过程失败时
        """
        try:
            results = []
            for i, text in enumerate(texts, 1):
                try:
                    logger.info(f"正在处理第 {i}/{len(texts)} 个文本...")
                    result = await self.summarize(text, config)
                    results.append(result)
                except GenerationError as e:
                    logger.error(f"处理第 {i} 个文本时出错: {str(e)}")
                    continue
                await asyncio.sleep(1)  # 避免速率限制
            return results
        except Exception as e:
            raise GenerationError(f"批量生成摘要时出错: {str(e)}")

    async def save_summaries(
        self,
        results: List[SummaryResult],
        output_dir: Path
    ) -> None:
        """保存摘要结果。

        Args:
            results (List[SummaryResult]): 摘要结果列表
            output_dir (Path): 输出目录

        Raises:
            GenerationError: 当保存过程失败时
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存为 JSON 文件
            data = [
                {
                    "original_text": r.original_text,
                    "summary": r.summary,
                    "config": {
                        "style": r.config.style,
                        "max_length": r.config.max_length,
                        "min_length": r.config.min_length,
                        "extractive": r.config.extractive,
                    },
                    "metadata": r.metadata,
                }
                for r in results
            ]

            output_file = output_dir / "summaries.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"摘要结果已保存到 {output_file}")

        except Exception as e:
            raise GenerationError(f"保存摘要结果时出错: {str(e)}")


async def main():
    """主函数。"""
    try:
        # 加载环境变量
        load_dotenv()

        # 验证环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("未设置 OPENAI_API_KEY 环境变量")

        # 创建模型和摘要器
        model = OpenAIModel(
            config=OpenAIConfig(
                model_name="gpt-3.5-turbo",
                api_key=api_key,
            ),
        )
        summarizer = TextSummarizer(model)

        # 示例文本
        texts = [
            """人工智能（AI）是计算机科学的一个分支，它致力于开发能够模拟人类智能的系统。
            这些系统能够学习、推理、感知、规划和解决问题。AI 技术已经在医疗、金融、交通等
            多个领域得到广泛应用。例如，在医疗领域，AI 可以辅助医生进行疾病诊断；在金融
            领域，AI 可以用于风险评估和欺诈检测；在交通领域，AI 正在推动自动驾驶技术的
            发展。然而，AI 的发展也带来了一些挑战，如隐私保护、就业影响和伦理问题等。
            未来，AI 技术将继续发展，为人类社会带来更多机遇和挑战。""",

            """量子计算是一种利用量子力学原理进行计算的新型计算范式。与传统计算机使用
            比特（0或1）不同，量子计算机使用量子比特（qubit），可以同时处于多个状态。
            这种特性使得量子计算机在某些特定问题上具有巨大的计算优势，如大数分解、
            数据库搜索和量子模拟等。目前，IBM、Google、微软等科技公司都在积极研发
            量子计算机。虽然量子计算技术还处于早期阶段，但已经取得了一些重要突破。
            例如，Google 在2019年宣布实现了"量子优越性"，证明了量子计算机在特定
            任务上超越了传统计算机。未来，量子计算有望在密码学、药物研发、材料科学
            等领域带来革命性的变革。""",

            """可持续发展是当今世界面临的重要议题。它强调在满足当代人需求的同时，
            不损害后代人满足其需求的能力。可持续发展包括三个主要方面：环境保护、
            经济发展和社会公平。在环境保护方面，需要减少污染、保护生物多样性、
            应对气候变化等；在经济发展方面，需要促进绿色经济、提高资源利用效率、
            发展可再生能源等；在社会公平方面，需要消除贫困、减少不平等、促进
            教育公平等。实现可持续发展需要政府、企业和个人的共同努力。各国政府
            需要制定相应的政策和法规，企业需要承担社会责任，个人需要改变消费
            习惯。只有通过多方合作，才能实现可持续发展的目标。"""
        ]

        # 生成摘要
        config = SummaryConfig(
            style="academic",
            max_length=150,
            min_length=50,
            extractive=False
        )

        results = await summarizer.batch_summarize(texts, config)

        # 打印摘要结果
        for i, result in enumerate(results, 1):
            print(f"\n文本 {i} 的摘要:")
            print(f"原文长度: {len(result.original_text)} 字")
            print(f"摘要长度: {len(result.summary)} 字")
            print(f"摘要风格: {result.config.style}")
            print(f"摘要方法: {'提取式' if result.config.extractive else '生成式'}")
            print(f"摘要内容:\n{result.summary}\n")

        # 保存摘要结果
        output_dir = Path("summaries")
        await summarizer.save_summaries(results, output_dir)

    except ConfigurationError as e:
        logger.error(f"配置错误: {str(e)}")
    except GenerationError as e:
        logger.error(f"生成错误: {str(e)}")
    except Exception as e:
        logger.error(f"发生未预期的错误: {str(e)}", exc_info=True)
    finally:
        # 清理资源
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main()) 