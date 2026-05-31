import os
import subprocess
import sys
from datetime import datetime


PIPELINE_STEPS = [
    {
        "name": "Air4Thai current PM2.5 collector",
        "module": "src.data_pipeline.collect_air4thai_current",
    },
    {
        "name": "OpenWeather current weather collector",
        "module": "src.data_pipeline.collect_openweather_current",
    },
    {
        "name": "NASA FIRMS recent fire collector",
        "module": "src.data_pipeline.collect_firms_recent",
    },
    {
        "name": "Current feature snapshot builder",
        "module": "src.features.build_current_snapshot",
    },
]


def safe_print(text: str) -> None:
    output_encoding = sys.stdout.encoding or "utf-8"
    safe_text = text.encode(output_encoding, errors="replace").decode(
        output_encoding,
        errors="replace",
    )
    print(safe_text)


def run_step(step_number: int, total_steps: int, step: dict) -> None:
    step_name = step["name"]
    module_name = step["module"]

    safe_print("=" * 80)
    safe_print(f"Step {step_number}/{total_steps}: {step_name}")
    safe_print(f"Module: {module_name}")
    safe_print("=" * 80)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.stdout:
        safe_print(result.stdout)

    if result.stderr:
        safe_print("Error output:")
        safe_print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped because this step failed: {step_name}"
        )

    safe_print(f"Completed: {step_name}\n")


def main() -> None:
    started_at = datetime.now()

    safe_print("Starting current PM2.5 data pipeline...")
    safe_print(f"Started at: {started_at.isoformat(timespec='seconds')}\n")

    total_steps = len(PIPELINE_STEPS)

    for index, step in enumerate(PIPELINE_STEPS, start=1):
        run_step(index, total_steps, step)

    finished_at = datetime.now()
    duration = finished_at - started_at

    safe_print("=" * 80)
    safe_print("Current PM2.5 data pipeline completed successfully.")
    safe_print(f"Finished at: {finished_at.isoformat(timespec='seconds')}")
    safe_print(f"Duration: {duration}")
    safe_print("=" * 80)


if __name__ == "__main__":
    main()