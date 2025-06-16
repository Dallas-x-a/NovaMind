"""Novamind 数据生成示例脚本。

这个示例展示了如何使用 Novamind 的数据生成功能来：
1. 生成和验证人物数据
2. 生成和验证公司数据
3. 保存生成的数据到文件
4. 加载和分析已保存的数据

使用方法：
1. 确保设置了必要的环境变量（如 OPENAI_API_KEY）
2. 运行脚本：python data_generation.py
3. 生成的数据将保存在 data 目录下
4. 可以查看日志了解生成和分析过程

数据格式：
- 人物数据 (persons.json):
  {
    "name": str,
    "age": int,
    "occupation": str,
    "skills": List[str],
    "bio": str
  }
- 公司数据 (companies.json):
  {
    "name": str,
    "industry": str,
    "employee_count": int,
    "description": str
  }
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import os

from dotenv import load_dotenv

from ..datagen import Person, Company, PersonGenerator, CompanyGenerator
from ..models.openai import OpenAIModel, OpenAIConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataGenerationError(Exception):
    """数据生成相关错误的基类。"""
    pass


class ConfigurationError(DataGenerationError):
    """配置相关错误。"""
    pass


class GenerationError(DataGenerationError):
    """生成过程相关错误。"""
    pass


class ValidationError(DataGenerationError):
    """数据验证相关错误。"""
    pass


@dataclass
class GenerationStats:
    """生成统计信息。"""
    total_generated: int
    valid_count: int
    invalid_count: int
    validation_errors: List[str]


async def setup_generators() -> Tuple[PersonGenerator, CompanyGenerator]:
    """设置数据生成器。

    Returns:
        Tuple[PersonGenerator, CompanyGenerator]: 人物和公司生成器实例

    Raises:
        ConfigurationError: 当配置无效时
    """
    try:
        # 验证环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("未设置 OPENAI_API_KEY 环境变量")

        # 创建 OpenAI 模型
        model = OpenAIModel(
            OpenAIConfig(
                model_name="gpt-3.5-turbo",
                api_key=api_key,
                temperature=0.7
            )
        )

        # 创建生成器
        return PersonGenerator(model=model), CompanyGenerator(model=model)

    except Exception as e:
        raise ConfigurationError(f"设置生成器时出错: {str(e)}")


async def generate_and_validate_data() -> Tuple[List[Person], List[Company]]:
    """生成并验证数据。

    Returns:
        Tuple[List[Person], List[Company]]: 验证通过的人物和公司数据列表

    Raises:
        GenerationError: 当生成过程失败时
        ValidationError: 当验证过程失败时
    """
    try:
        # 设置生成器
        person_gen, company_gen = await setup_generators()

        # 生成人物数据
        logger.info("开始生成人物数据...")
        try:
            persons = await person_gen.generate(
                constraints=(
                    "- 生成 5 个不同职业的人物\n"
                    "- 包含至少 2 个医生\n"
                    "- 技能列表要符合职业特点\n"
                    "- 每个人物都要有详细的个人简介"
                )
            )
            logger.info(f"生成 {len(persons)} 个人物数据")
        except Exception as e:
            raise GenerationError(f"生成人物数据时出错: {str(e)}")

        # 验证人物数据
        try:
            valid_persons = await person_gen.filter(persons)
            logger.info(f"验证通过 {len(valid_persons)} 个人物数据")
        except Exception as e:
            raise ValidationError(f"验证人物数据时出错: {str(e)}")

        # 打印人物数据示例
        if valid_persons:
            logger.info("\n人物数据示例:")
            for person in valid_persons[:2]:
                logger.info(json.dumps(person.model_dump(), ensure_ascii=False, indent=2))

        # 生成公司数据
        logger.info("\n开始生成公司数据...")
        try:
            companies = await company_gen.generate(
                constraints=(
                    "- 生成 3 个不同行业的公司\n"
                    "- 包含至少 1 个科技公司\n"
                    "- 公司规模要合理\n"
                    "- 每个公司都要有详细的简介"
                )
            )
            logger.info(f"生成 {len(companies)} 个公司数据")
        except Exception as e:
            raise GenerationError(f"生成公司数据时出错: {str(e)}")

        # 验证公司数据
        try:
            valid_companies = await company_gen.filter(companies)
            logger.info(f"验证通过 {len(valid_companies)} 个公司数据")
        except Exception as e:
            raise ValidationError(f"验证公司数据时出错: {str(e)}")

        # 打印公司数据示例
        if valid_companies:
            logger.info("\n公司数据示例:")
            for company in valid_companies[:2]:
                logger.info(json.dumps(company.model_dump(), ensure_ascii=False, indent=2))

        # 保存数据
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            await person_gen.save(valid_persons, data_dir / "persons.json")
            await company_gen.save(valid_companies, data_dir / "companies.json")
            logger.info("\n数据已保存到 data 目录")
        except Exception as e:
            raise GenerationError(f"保存数据时出错: {str(e)}")

        return valid_persons, valid_companies

    except (GenerationError, ValidationError) as e:
        raise
    except Exception as e:
        raise GenerationError(f"生成和验证数据时发生未预期的错误: {str(e)}")


async def load_and_analyze_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """加载并分析数据。

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: 人物和公司数据的分析结果

    Raises:
        DataGenerationError: 当加载或分析过程失败时
    """
    try:
        # 创建生成器
        person_gen = PersonGenerator()
        company_gen = CompanyGenerator()

        # 加载数据
        data_dir = Path("data")
        if not data_dir.exists():
            raise DataGenerationError("data 目录不存在")

        try:
            persons = await person_gen.load(data_dir / "persons.json")
            companies = await company_gen.load(data_dir / "companies.json")
        except Exception as e:
            raise DataGenerationError(f"加载数据时出错: {str(e)}")

        analysis_results = {
            "persons": {},
            "companies": {}
        }

        # 分析人物数据
        if persons:
            logger.info("\n人物数据分析:")
            try:
                occupations = {}
                age_groups = {"0-20": 0, "21-40": 0, "41-60": 0, "61+": 0}

                for person in persons:
                    # 统计职业分布
                    occupations[person.occupation] = occupations.get(person.occupation, 0) + 1

                    # 统计年龄分布
                    if person.age <= 20:
                        age_groups["0-20"] += 1
                    elif person.age <= 40:
                        age_groups["21-40"] += 1
                    elif person.age <= 60:
                        age_groups["41-60"] += 1
                    else:
                        age_groups["61+"] += 1

                analysis_results["persons"] = {
                    "occupations": occupations,
                    "age_groups": age_groups
                }

                logger.info("职业分布:")
                for occupation, count in occupations.items():
                    logger.info(f"- {occupation}: {count}人")

                logger.info("\n年龄分布:")
                for age_group, count in age_groups.items():
                    logger.info(f"- {age_group}: {count}人")

            except Exception as e:
                raise DataGenerationError(f"分析人物数据时出错: {str(e)}")

        # 分析公司数据
        if companies:
            logger.info("\n公司数据分析:")
            try:
                industries = {}
                size_groups = {"小型(1-50人)": 0, "中型(51-500人)": 0, "大型(501+人)": 0}

                for company in companies:
                    # 统计行业分布
                    industries[company.industry] = industries.get(company.industry, 0) + 1

                    # 统计规模分布
                    if company.employee_count <= 50:
                        size_groups["小型(1-50人)"] += 1
                    elif company.employee_count <= 500:
                        size_groups["中型(51-500人)"] += 1
                    else:
                        size_groups["大型(501+人)"] += 1

                analysis_results["companies"] = {
                    "industries": industries,
                    "size_groups": size_groups
                }

                logger.info("行业分布:")
                for industry, count in industries.items():
                    logger.info(f"- {industry}: {count}家")

                logger.info("\n规模分布:")
                for size_group, count in size_groups.items():
                    logger.info(f"- {size_group}: {count}家")

            except Exception as e:
                raise DataGenerationError(f"分析公司数据时出错: {str(e)}")

        return analysis_results["persons"], analysis_results["companies"]

    except DataGenerationError as e:
        raise
    except Exception as e:
        raise DataGenerationError(f"加载和分析数据时发生未预期的错误: {str(e)}")


async def main():
    """主函数。"""
    try:
        # 加载环境变量
        load_dotenv()

        # 生成并验证数据
        await generate_and_validate_data()

        # 加载并分析数据
        await load_and_analyze_data()

    except ConfigurationError as e:
        logger.error(f"配置错误: {str(e)}")
    except GenerationError as e:
        logger.error(f"生成错误: {str(e)}")
    except ValidationError as e:
        logger.error(f"验证错误: {str(e)}")
    except DataGenerationError as e:
        logger.error(f"数据处理错误: {str(e)}")
    except Exception as e:
        logger.error(f"发生未预期的错误: {str(e)}", exc_info=True)
    finally:
        # 清理资源
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main()) 