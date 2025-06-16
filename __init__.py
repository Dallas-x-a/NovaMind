"""Novamind - Next Generation AI Agent Framework."""

from importlib.metadata import version

__version__ = version("novamind")

from novamind.core.agent import Agent
from novamind.core.environment import Environment
from novamind.core.config import Config

__all__ = ["Agent", "Environment", "Config", "__version__"] 