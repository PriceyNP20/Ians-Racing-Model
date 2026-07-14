from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginRegistry:
    """Small named registry so provider/model modules can be replaced cleanly."""

    _plugins: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, plugin: Any) -> None:
        if not name.strip():
            raise ValueError("Plugin name is required.")
        self._plugins[name] = plugin

    def get(self, name: str) -> Any:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise KeyError(f"Plugin '{name}' is not registered.") from exc

    def names(self) -> list[str]:
        return sorted(self._plugins)
