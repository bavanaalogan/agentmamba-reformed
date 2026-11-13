"""Configuration loading. YAML files are deep-merged; values can be overridden with
dotted keys (e.g. ``{"memory.budget_fraction": 0.05}``)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    budget = get_dotted(cfg, "memory.budget_fraction")
    if not (isinstance(budget, (int, float)) and 0.0 < budget <= 1.0):
        raise ValueError(f"memory.budget_fraction must be in (0, 1], got {budget!r}")
    if get_dotted(cfg, "memory.policy") not in {"learned", "heuristic", "random", "store_all"}:
        raise ValueError(f"unknown memory.policy {get_dotted(cfg, 'memory.policy')!r}")
    if int(get_dotted(cfg, "retrieval.top_k", 0)) < 1:
        raise ValueError("retrieval.top_k must be >= 1")
    if get_dotted(cfg, "gate.model") not in {"logreg", "mlp"}:
        raise ValueError("gate.model must be logreg or mlp")


def resolve_path(cfg: dict, key: str) -> Path:
    p = Path(get_dotted(cfg, key))
    return p if p.is_absolute() else PROJECT_ROOT / p


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
