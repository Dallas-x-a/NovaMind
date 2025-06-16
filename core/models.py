"""Model interfaces for Novamind."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    """Response from a model."""
    
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None


class ModelMessage(BaseModel):
    """Message for model input/output."""
    
    role: str
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Model configuration."""
    
    model_name: str
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ):
        """Initialize the model.
        
        Args:
            config: Model configuration
            **kwargs: Additional arguments
        """
        self.config = config or ModelConfig(model_name="default")
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name
        
    @abstractmethod
    async def generate(
        self,
        messages: List[ModelMessage],
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response.
        
        Args:
            messages: List of input messages
            **kwargs: Additional arguments
            
        Returns:
            Model response
        """
        pass
        
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[ModelMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[ModelResponse, None]:
        """Generate a streaming response.
        
        Args:
            messages: List of input messages
            **kwargs: Additional arguments
            
        Yields:
            Model response chunks
        """
        pass
        
    def format_messages(
        self,
        messages: List[Union[str, Dict[str, Any], ModelMessage]],
    ) -> List[ModelMessage]:
        """Format messages for model input.
        
        Args:
            messages: List of messages in various formats
            
        Returns:
            List of formatted ModelMessage objects
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, str):
                formatted.append(ModelMessage(role="user", content=msg))
            elif isinstance(msg, dict):
                formatted.append(ModelMessage(**msg))
            elif isinstance(msg, ModelMessage):
                formatted.append(msg)
            else:
                raise TypeError(f"Unsupported message type: {type(msg)}")
        return formatted
        
    def update_config(self, **kwargs: Any) -> None:
        """Update model configuration.
        
        Args:
            **kwargs: Configuration updates
        """
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "model_name": self.model_name,
            "config": self.config.model_dump(),
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Create model from dictionary."""
        config = ModelConfig(**data["config"])
        return cls(config=config)


class ModelRegistry:
    """Registry for managing models."""
    
    def __init__(self):
        """Initialize the registry."""
        self._models: Dict[str, type[BaseModel]] = {}
        
    def register(self, name: str, model_class: type[BaseModel]) -> None:
        """Register a model class.
        
        Args:
            name: Model name
            model_class: Model class to register
        """
        if name in self._models:
            raise ValueError(f"Model {name} already registered")
        self._models[name] = model_class
        
    def unregister(self, name: str) -> None:
        """Unregister a model class.
        
        Args:
            name: Name of model to unregister
        """
        if name not in self._models:
            raise ValueError(f"Model {name} not found")
        del self._models[name]
        
    def get_model_class(self, name: str) -> type[BaseModel]:
        """Get a model class by name.
        
        Args:
            name: Name of model to get
            
        Returns:
            The requested model class
        """
        if name not in self._models:
            raise ValueError(f"Model {name} not found")
        return self._models[name]
        
    def create_model(
        self,
        name: str,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Create a model instance.
        
        Args:
            name: Name of model to create
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        model_class = self.get_model_class(name)
        return model_class(config=config, **kwargs)
        
    def list_models(self) -> List[str]:
        """List all registered models.
        
        Returns:
            List of model names
        """
        return list(self._models.keys()) 