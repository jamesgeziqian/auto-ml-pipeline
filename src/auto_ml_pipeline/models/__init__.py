"""Model adapter registry."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type


_ADAPTERS: Dict[str, Type[Any]] = {}


def register_model(name: str) -> Callable[[Type[Any]], Type[Any]]:
    def decorator(cls: Type[Any]) -> Type[Any]:
        _ADAPTERS[name] = cls
        cls.name = name
        return cls

    return decorator


def create_adapter(name: str, **kwargs: Any) -> Any:
    if name not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS))
        raise KeyError(f"Unknown model adapter {name!r}. Available: {available}")
    return _ADAPTERS[name](**kwargs)


def available_models() -> list[str]:
    return sorted(_ADAPTERS)


from . import lightgbm_adapter  # noqa: E402,F401  (register built-in adapters)
