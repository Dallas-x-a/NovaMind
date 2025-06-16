"""Configuration management for Novamind."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration settings."""
    
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[list[str]] = None


class MemoryConfig(BaseModel):
    """Memory configuration settings."""
    
    max_tokens: int = Field(default=2000, gt=0)
    max_messages: int = Field(default=50, gt=0)
    memory_type: str = Field(default="conversation")
    persist: bool = Field(default=False)


class ToolConfig(BaseModel):
    """Tool configuration settings."""
    
    max_iterations: int = Field(default=5, gt=0)
    timeout: int = Field(default=30, gt=0)
    retry_attempts: int = Field(default=3, gt=0)
    parallel_execution: bool = Field(default=False)


class Config(BaseModel):
    """Main configuration class for Novamind."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    
    # General settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    api_key: Optional[str] = None
    
    # Custom settings
    custom: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        env_prefix = "NOVAMIND_"
        case_sensitive = False
        
    def update(self, **kwargs: Any) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), BaseModel):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                self.custom[key] = value
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(**data)
        
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            self.model_validate(self.model_dump())
            return True
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}") 