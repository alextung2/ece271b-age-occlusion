from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml

@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return Config(raw=raw)

    def get(self, key: str, default=None):
        cur = self.raw
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur