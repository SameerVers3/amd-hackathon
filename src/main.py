import json
import logging
import os
import sys
import tempfile
import time
import traceback

from captioner import REQUIRED_STYLES, generate_captions
from utils import fetch_clip

def _get_default_input():
    return "/input/tasks.json" if os.path.exists("/input") else "input/tasks.json"

def _get_default_output():
    return "/output/results.json" if os.path.exists("/output") else "output/results.json"

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", _get_default_input())
OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", _get_default_output())

# limits to 10 minutes by default, but can be overridden by env var for testing
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", 9 * 60))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("captioner")


def load_tasks(path: str):
    with open(path, "r") as f:
        return json.load(f)

def write_results(path: str, results: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path) 

def process_task(task: dict, workdir: str) -> dict:
    task_id = task["task_id"]
    video_url = task["video_url"]
    styles = task.get("styles") or REQUIRED_STYLES

    log.info("Task %s: fetching %s", task_id, video_url)
    local_path = fetch_clip(video_url, workdir)

    log.info("Task %s: generating captions for styles=%s", task_id, styles)
    captions = generate_captions(local_path, styles)

    for style in styles:
        captions.setdefault(style, "")

    return {"task_id": task_id, "captions": captions}

def main() -> int:
    start = time.monotonic()

    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception:
        log.error("Failed to read/parse %s:\n%s", INPUT_PATH, traceback.format_exc())
        write_results(OUTPUT_PATH, [])
        return 1

    results = []
    with tempfile.TemporaryDirectory() as workdir:
        for task in tasks:
            elapsed = time.monotonic() - start
            if elapsed > MAX_RUNTIME_SECONDS:
                log.warning(
                    "Time budget exceeded (%.0fs); skipping remaining tasks", elapsed
                )
                results.append(
                    {
                        "task_id": task.get("task_id", "unknown"),
                        "captions": {s: "" for s in task.get("styles") or REQUIRED_STYLES},
                    }
                )
                continue

            try:
                results.append(process_task(task, workdir))
            except Exception:
                log.error(
                    "Task %s failed:\n%s",
                    task.get("task_id", "unknown"),
                    traceback.format_exc(),
                )
                
                results.append(
                    {
                        "task_id": task.get("task_id", "unknown"),
                        "captions": {s: "" for s in task.get("styles") or REQUIRED_STYLES},
                    }
                )

    write_results(OUTPUT_PATH, results)
    log.info("Wrote %d results to %s in %.1fs", len(results), OUTPUT_PATH, time.monotonic() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
