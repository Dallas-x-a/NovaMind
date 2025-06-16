"""Novamind LLM 数据生成器实现。"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ..models.base import BaseModel as LLMModel
from ..models.openai import OpenAIModel, OpenAIConfig
from .base import DataGenerator, DataGeneratorConfig

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMDataGeneratorConfig(DataGeneratorConfig):
    """LLM 数据生成器配置。"""
    
    model_name: str = "gpt-3.5-turbo"
    prompt_template: str = "请生成以下格式的数据:\n{format}\n\n要求:\n{requirements}"
    format_template: str = "请以 JSON 格式返回数据，格式如下:\n{json_schema}"
    requirements_template: str = "- 数据必须符合格式要求\n- 数据必须真实合理\n- 数据必须多样化\n- 数据必须符合以下约束:\n{constraints}"


class LLMDataGenerator(DataGenerator[T]):
    """基于 LLM 的数据生成器。"""
    
    def __init__(
        self,
        data_type: Type[T],
        model: Optional[LLMModel] = None,
        config: Optional[LLMDataGeneratorConfig] = None,
    ):
        """初始化 LLM 数据生成器。
        
        参数:
            data_type: 数据类型
            model: LLM 模型
            config: 生成器配置
        """
        super().__init__(config or LLMDataGeneratorConfig())
        self.data_type = data_type
        self.model = model or OpenAIModel(
            OpenAIConfig(model_name=self.config.model_name)
        )
        
    def _get_prompt(self, constraints: Optional[str] = None) -> str:
        """获取生成提示。
        
        参数:
            constraints: 数据约束
            
        返回:
            生成提示
        """
        json_schema = self.data_type.model_json_schema()
        format_str = self.config.format_template.format(json_schema=json.dumps(json_schema, indent=2))
        requirements = self.config.requirements_template.format(
            constraints=constraints or "无特殊约束"
        )
        return self.config.prompt_template.format(
            format=format_str,
            requirements=requirements
        )
        
    async def generate(self, constraints: Optional[str] = None, **kwargs: Any) -> List[T]:
        """生成数据。
        
        参数:
            constraints: 数据约束
            **kwargs: 其他生成参数
            
        返回:
            生成的数据列表
        """
        prompt = self._get_prompt(constraints)
        response = await self.model.generate([{"role": "user", "content": prompt}])
        
        try:
            # 尝试解析 JSON 数据
            data_list = json.loads(response.content)
            if not isinstance(data_list, list):
                data_list = [data_list]
                
            # 转换为目标类型
            result = []
            for data in data_list:
                try:
                    item = self.data_type.model_validate(data)
                    if await self.validate(item):
                        result.append(item)
                except ValidationError as e:
                    logger.warning(f"数据验证失败: {e}")
                    continue
                    
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            return []
            
    async def validate(self, data: T) -> bool:
        """验证生成的数据。
        
        参数:
            data: 要验证的数据
            
        返回:
            数据是否有效
        """
        try:
            # 基本验证
            if not isinstance(data, self.data_type):
                return False
                
            # 自定义验证
            return await self._custom_validate(data)
            
        except Exception as e:
            logger.error(f"数据验证异常: {e}")
            return False
            
    async def _custom_validate(self, data: T) -> bool:
        """自定义数据验证。
        
        参数:
            data: 要验证的数据
            
        返回:
            数据是否有效
        """
        # 子类可以重写此方法实现自定义验证
        return True
        
    async def filter(self, data_list: List[T]) -> List[T]:
        """过滤生成的数据。
        
        参数:
            data_list: 要过滤的数据列表
            
        返回:
            过滤后的数据列表
        """
        result = []
        for data in data_list:
            if await self.validate(data):
                result.append(data)
        return result
        
    async def save(self, data_list: List[T], path: str) -> None:
        """保存生成的数据。
        
        参数:
            data_list: 要保存的数据列表
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [item.model_dump() for item in data_list]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    async def load(self, path: str) -> List[T]:
        """加载生成的数据。
        
        参数:
            path: 数据路径
            
        返回:
            加载的数据列表
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                data = [data]
                
            result = []
            for item in data:
                try:
                    instance = self.data_type.model_validate(item)
                    if await self.validate(instance):
                        result.append(instance)
                except ValidationError as e:
                    logger.warning(f"数据验证失败: {e}")
                    continue
                    
            return result
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return [] 