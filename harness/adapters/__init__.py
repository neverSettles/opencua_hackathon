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
from .anthropic_direct import AnthropicDirectAdapter

__all__ = [
    "Action",
    "ActionType",
    "AdapterStep",
    "ModelAdapter",
    "execute_action",
    "NorthstarAdapter",
    "GeminiAdapter",
    "OpenAIComputerUseAdapter",
    "AnthropicDirectAdapter",
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
    if name in ("anthropic", "claude", "opus", "sonnet"):
        return AnthropicDirectAdapter(**kwargs)
    if name in ("local", "local_northstar", "local-northstar", "finetuned"):
        # Lazy import: only available where torch + peft are installed.
        from .local_northstar import LocalNorthstarAdapter
        return LocalNorthstarAdapter(**kwargs)
    raise ValueError(f"unknown adapter: {name!r}")
