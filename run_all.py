"""Run all lab steps sequentially or by step number."""

import argparse
import runpy
from pathlib import Path

STEPS = {
    1: "01_langsmith_rag_pipeline.py",
    2: "02_prompt_hub_ab_routing.py",
    3: "03_ragas_evaluation.py",
    4: "04_guardrails_validator.py",
}


def run_step(step: int) -> None:
    path = Path(__file__).with_name(STEPS[step])
    print(f"\n=== Running step {step}: {path.name} ===")
    runpy.run_path(str(path), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 22 lab steps")
    parser.add_argument("--step", type=int, choices=STEPS.keys())
    args = parser.parse_args()

    if args.step:
        run_step(args.step)
        return

    for step in sorted(STEPS):
        run_step(step)


if __name__ == "__main__":
    main()
