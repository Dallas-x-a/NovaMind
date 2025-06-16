"""Novamind 数据生成示例。"""

from typing import List, Optional
from pydantic import BaseModel, Field

from .llm import LLMDataGenerator, LLMDataGeneratorConfig


class Person(BaseModel):
    """人物信息模型。"""
    
    name: str = Field(..., description="姓名")
    age: int = Field(..., ge=0, lt=120, description="年龄")
    occupation: str = Field(..., description="职业")
    skills: List[str] = Field(default_factory=list, description="技能列表")
    bio: Optional[str] = Field(None, description="个人简介")


class Company(BaseModel):
    """公司信息模型。"""
    
    name: str = Field(..., description="公司名称")
    industry: str = Field(..., description="所属行业")
    founded_year: int = Field(..., ge=1800, le=2024, description="成立年份")
    employee_count: int = Field(..., ge=1, description="员工数量")
    headquarters: str = Field(..., description="总部所在地")
    description: Optional[str] = Field(None, description="公司简介")


class PersonGenerator(LLMDataGenerator[Person]):
    """人物信息生成器。"""
    
    def __init__(self, **kwargs):
        """初始化人物信息生成器。"""
        config = LLMDataGeneratorConfig(
            model_name="gpt-3.5-turbo",
            prompt_template=(
                "请生成一些人物信息数据。\n"
                "要求生成的数据要真实合理，包含不同年龄段、不同职业的人物。\n"
                "每个人物都应该有合适的技能列表和简介。\n"
                "数据格式如下:\n{format}\n\n"
                "额外要求:\n{requirements}"
            )
        )
        super().__init__(Person, config=config, **kwargs)
        
    async def _custom_validate(self, data: Person) -> bool:
        """自定义验证。
        
        参数:
            data: 要验证的数据
            
        返回:
            数据是否有效
        """
        # 验证技能列表
        if not data.skills:
            return False
            
        # 验证简介
        if data.bio and len(data.bio) < 10:
            return False
            
        # 验证职业与技能的关联性
        if data.occupation.lower() in ["医生", "医生", "physician"]:
            if not any(skill.lower() in ["医疗", "诊断", "治疗", "medicine", "diagnosis"] for skill in data.skills):
                return False
                
        return True


class CompanyGenerator(LLMDataGenerator[Company]):
    """公司信息生成器。"""
    
    def __init__(self, **kwargs):
        """初始化公司信息生成器。"""
        config = LLMDataGeneratorConfig(
            model_name="gpt-3.5-turbo",
            prompt_template=(
                "请生成一些公司信息数据。\n"
                "要求生成的数据要真实合理，包含不同行业、不同规模的公司。\n"
                "每个公司都应该有合适的成立年份和员工数量。\n"
                "数据格式如下:\n{format}\n\n"
                "额外要求:\n{requirements}"
            )
        )
        super().__init__(Company, config=config, **kwargs)
        
    async def _custom_validate(self, data: Company) -> bool:
        """自定义验证。
        
        参数:
            data: 要验证的数据
            
        返回:
            数据是否有效
        """
        # 验证公司规模与员工数量的合理性
        if data.employee_count < 10 and "大型" in data.description:
            return False
            
        # 验证成立年份与公司描述的合理性
        if data.founded_year > 2000 and "百年老店" in data.description:
            return False
            
        # 验证行业与公司描述的关联性
        if "科技" in data.industry.lower() and "传统制造业" in data.description:
            return False
            
        return True


async def generate_example_data():
    """生成示例数据。"""
    # 创建生成器
    person_gen = PersonGenerator()
    company_gen = CompanyGenerator()
    
    # 生成人物数据
    persons = await person_gen.generate(
        constraints=(
            "- 生成 5 个不同职业的人物\n"
            "- 包含至少 2 个医生\n"
            "- 技能列表要符合职业特点"
        )
    )
    
    # 生成公司数据
    companies = await company_gen.generate(
        constraints=(
            "- 生成 3 个不同行业的公司\n"
            "- 包含至少 1 个科技公司\n"
            "- 公司规模要合理"
        )
    )
    
    # 保存数据
    await person_gen.save(persons, "data/persons.json")
    await company_gen.save(companies, "data/companies.json")
    
    return persons, companies 