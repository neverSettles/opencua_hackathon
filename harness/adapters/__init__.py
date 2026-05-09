from .base import (
    Action,
    ActionType,
    AdapterStep,
    ModelAdapter,
    execute_action,
)
from .northstar import NorthstarAdapter
from .gemini import GeminiAdapter
from .openai_cua import OpenAIComputerUseAdapter

__all__ = [
    "Action",
    "ActionType",
    "AdapterStep",
    "ModelAdapter",
    "execute_action",
    "NorthstarAdapter",
    "GeminiAdapter",
    "OpenAIComputerUseAdapter",
    "build_adapter",
]


def build_adapter(name: str, **kwargs) -> ModelAdapter:
    """Factory: instantiate an adapter by short name."""
    name = name.lower()
    if name in ("northstar", "tzafon", "lightcone"):
        return NorthstarAdapter(**kwargs)
    if name in ("gemini", "google"):
        return GeminiAdapter(**kwargs)
    if name in ("openai", "gpt", "computer-use"):
        return OpenAIComputerUseAdapter(**kwargs)
    raise ValueError(f"unknown adapter: {name!r}")
