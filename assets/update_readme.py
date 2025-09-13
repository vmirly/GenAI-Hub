#!/usr/bin/env python3
"""
update_readme.py — Generate/refresh the model index in README.md from per-model metadata.

Conventions:
- Each model lives in a top-level folder: MODEL_NAME/
- Each model folder contains a `metadata.json` (preferred) or `metadata.josn` (typo tolerated)
- The metadata file has at least: {"tasks": ["segmentation", "detection", ...]}

Behavior:
- Scans only immediate subfolders of repo root.
- Replaces content between markers:
    <!-- AUTO-GENERATED:MODEL-INDEX:START -->
    ... generated content ...
    <!-- AUTO-GENERATED:MODEL-INDEX:END -->
  If markers are missing, a new "## Models" section with markers is appended.

Idempotent & safe to run multiple times.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

MARKER_START = "<!-- AUTO-GENERATED:MODEL-INDEX:START -->"
MARKER_END = "<!-- AUTO-GENERATED:MODEL-INDEX:END -->"
README_NAME = "README.md"
PREFERRED_META = "metadata.json"


def find_models_with_metadata(root: Path) -> List[Tuple[str, Path, Path]]:
    """
    Returns a list of (model_name, folder_path, metadata_path) for each model found.
    Only scans direct children of `root`.
    """
    models = []
    for entry in sorted(os.listdir(root)):
        folder = root / entry
        if not folder.is_dir():
            continue
        meta = folder / PREFERRED_META
        if meta.exists():
            models.append((entry, folder, meta))
    print(models)
    return models


def load_tasks(meta_path: Path) -> List[str]:
    """
    Read JSON and return a normalized list of tasks.
    Accepts 'tasks' as list (preferred) or string (coerced); empty list if missing/invalid.
    """
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    tasks = data.get("tasks", [])
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list):
        return []
    # normalize: strip whitespace, drop empties, de-duplicate (case-insensitive) while preserving order
    seen = set()
    normed: List[str] = []
    for t in tasks:
        if not isinstance(t, str):
            continue
        s = t.strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            normed.append(s)
    return normed


def generate_markdown(models: List[Tuple[str, List[str]]]) -> str:
    """
    Build the markdown for the managed section: by model + by task.
    `models` is a list of (model_name, tasks).
    """
    # Stable order
    models = sorted(models, key=lambda x: x[0].lower())

    # Build by-model bullets
    by_model_lines = []
    for name, tasks in models:
        tasks_str = ", ".join(tasks) if tasks else "—"
        by_model_lines.append(f"* [{name}]({name}) — {tasks_str}")

    # Build by-task map
    by_task: Dict[str, List[str]] = {}
    for name, tasks in models:
        for t in tasks:
            key = t.strip()
            if key:
                by_task.setdefault(key, []).append(name)

    # Stable sort for tasks and model lists
    by_task_md = []
    if by_task:
        for task in sorted(by_task.keys(), key=lambda s: s.lower()):
            names = sorted(by_task[task], key=lambda s: s.lower())
            items = "\n".join(f"  * [{n}]({n})" for n in names)
            by_task_md.append(f"\n**{task}**\n{items}")

    by_model_block = by_model_lines or ["_No models found._"]
    by_task_block = by_task_md or ["_No task tags found._"]

    section_lines = []
    section_lines.append(MARKER_START)
    section_lines.append("")
    section_lines.append("### Index (by model)")
    section_lines.append("")
    section_lines.extend(by_model_block)
    section_lines.append("")
    section_lines.append("### Index (by task)")
    section_lines.append("")
    section_lines.extend(by_task_block)
    section_lines.append("")
    section_lines.append(MARKER_END)
    section_lines.append("")

    return "\n".join(section_lines)


def splice_managed_section(readme_text: str, new_block: str) -> str:
    """
    Replace text between markers. If markers are absent, append a new section.
    """
    if MARKER_START in readme_text and MARKER_END in readme_text:
        pre, rest = readme_text.split(MARKER_START, 1)
        _, post = rest.split(MARKER_END, 1)
        return f"{pre}{new_block}{post}"
    else:
        # Append a new Models section at the end
        suffix = "\n\n## Models\n\n" + new_block
        if readme_text.endswith("\n"):
            return readme_text + suffix.lstrip("\n")
        return readme_text + suffix


def main() -> int:
    root = Path(".")
    print(root)
    readme_path = root / README_NAME
    print(readme_path)

    # Gather models
    found = find_models_with_metadata(root)
    models_with_tasks: List[Tuple[str, List[str]]] = []
    for name, _, meta in found:
        tasks = load_tasks(meta)
        models_with_tasks.append((name, tasks))

    # Prepare generated markdown block
    new_block = generate_markdown(models_with_tasks)

    # Read existing README or create a minimal stub
    if readme_path.exists():
        current = readme_path.read_text(encoding="utf-8")
    else:
        current = f"# {root.name}\n\n"

    updated = splice_managed_section(current, new_block)

    # Only write if changed
    if updated != current:
        readme_path.write_text(updated, encoding="utf-8")
        print(f"README updated. Indexed {len(models_with_tasks)} model(s).")
    else:
        print("README already up-to-date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
