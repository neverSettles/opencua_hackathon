from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "bench"
WORLDS_DIR = BENCH_DIR / "worlds"


def load_structured(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without requiring PyYAML.

    The benchmark files use .yaml extensions for readability, but are kept
    JSON-compatible so the hackathon build has no package-install dependency.
    If PyYAML is installed, regular YAML also works.
    """
    path = Path(path)
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ImportError as import_error:
            raise ValueError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from import_error
        try:
            data = yaml.safe_load(text)
        except Exception as yaml_error:  # pragma: no cover - depends on PyYAML
            raise ValueError(f"Failed to parse {path}") from yaml_error
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain an object")
        return data
    except Exception:
        raise json_error


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True, default=str) + "\n")


def world_dir(world_id: str) -> Path:
    return WORLDS_DIR / world_id


def task_dir(world_id: str, task_id: str) -> Path:
    return world_dir(world_id) / "tasks" / task_id


def load_world(world_id: str) -> dict[str, Any]:
    return load_structured(world_dir(world_id) / "world.yaml")


def load_task(world_id: str, task_id: str) -> dict[str, Any]:
    return load_structured(task_dir(world_id, task_id) / "task.yaml")


def load_rubric(world_id: str, task_id: str) -> dict[str, Any]:
    return load_structured(task_dir(world_id, task_id) / "rubric.yaml")


def load_sellers(world_id: str) -> dict[str, dict[str, Any]]:
    sellers: dict[str, dict[str, Any]] = {}
    sellers_dir = world_dir(world_id) / "sellers"
    for path in sorted(sellers_dir.glob("*.yaml")):
        seller = load_structured(path)
        sellers[seller["id"]] = seller
    return sellers


def load_ground_truth(world_id: str, task_id: str) -> dict[str, Any]:
    return load_structured(task_dir(world_id, task_id) / "ground_truth.json")


def inventory_by_sku(sellers: dict[str, dict[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
    by_sku: dict[str, tuple[str, dict[str, Any]]] = {}
    for seller_id, seller in sellers.items():
        for item in seller.get("inventory", []):
            by_sku[item["sku"]] = (seller_id, item)
    return by_sku


def effective_unit_price(item: dict[str, Any], qty: int, basis: str = "list") -> float:
    if basis == "reservation":
        base = float(item["reservation_price"])
    else:
        base = float(item["list_price"])
    discount = 0.0
    for rule in item.get("volume_discounts", []):
        if qty >= int(rule["min_qty"]):
            discount = max(discount, float(rule["discount_pct"]))
    return round(base * (1 - discount), 2)


def item_matches_request(request: dict[str, Any], inventory_item: dict[str, Any]) -> bool:
    if "accepted_skus" in request and inventory_item.get("sku") in request["accepted_skus"]:
        return True
    if request.get("sku") and inventory_item.get("sku") == request["sku"]:
        return True
    if request.get("item_id") and inventory_item.get("item_id") == request["item_id"]:
        return True
    return False


def item_meets_specs(inventory_item: dict[str, Any], constraints: dict[str, Any]) -> bool:
    specs = inventory_item.get("specs", {})
    for key, minimum in constraints.get("min_specs", {}).items():
        if float(specs.get(key, 0)) < float(minimum):
            return False
    for key, expected in constraints.get("equals", {}).items():
        if specs.get(key) != expected:
            return False
    return True
