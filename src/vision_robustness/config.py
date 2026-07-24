"""Configuration loading and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve_device(name: str = "auto") -> torch.device:
    """Pick the best available torch device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@dataclass
class Config:
    """Thin wrapper around the YAML config with resolved absolute paths."""

    raw: dict[str, Any]
    root: Path = ROOT

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, key: str) -> Path:
        """Resolve a paths.* entry relative to the project root."""
        rel = self.get("paths", key)
        if rel is None:
            raise KeyError(f"Unknown path key: {key}")
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def device(self) -> torch.device:
        return resolve_device(self.get("project", "device", default="auto"))

    @property
    def seed(self) -> int:
        return int(self.get("project", "seed", default=42))

    def ensure_dirs(self) -> None:
        for key in (
            "raw_dir",
            "processed_dir",
            "distorted_dir",
            "enhanced_dir",
            "results_dir",
            "figures_dir",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)
