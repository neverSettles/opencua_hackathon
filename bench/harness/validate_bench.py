from __future__ import annotations

import argparse

from bench.harness.compile_ground_truth import compile_task
from bench.harness.io import WORLDS_DIR, load_sellers, load_task, load_world


def validate() -> list[str]:
    errors: list[str] = []
    for world_path in sorted(WORLDS_DIR.glob("*/world.yaml")):
        world_id = world_path.parent.name
        try:
            world = load_world(world_id)
            sellers = load_sellers(world_id)
            for seller_id in world.get("sellers", []):
                if seller_id not in sellers:
                    errors.append(f"{world_id}: world references missing seller {seller_id}")
            for task_path in sorted((world_path.parent / "tasks").glob("*/task.yaml")):
                task_id = task_path.parent.name
                task = load_task(world_id, task_id)
                for seller_id in task["config"].get("allowed_sellers", []):
                    if seller_id not in sellers:
                        errors.append(f"{world_id}/{task_id}: missing seller {seller_id}")
                compile_task(world_id, task_id, write=False)
        except Exception as exc:
            errors.append(f"{world_id}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    errors = validate()
    if errors:
        print("Benchmark validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Benchmark validation passed.")


if __name__ == "__main__":
    main()
