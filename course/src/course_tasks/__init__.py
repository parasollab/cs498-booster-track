"""Course task package.

mjlab discovers this package through the `mjlab.tasks` entry point declared in
pyproject.toml and imports it, which runs the imports below. Each hw module
calls `register_mjlab_task(...)` at import time — that is the ONLY thing that
makes a task id appear in `uv run list-envs` / `train` / `play`.

When a new assignment is released, its module gets added here.
"""

# Imported for their side effects: each module calls register_mjlab_task() at
# import time. Nothing reads these names, hence the noqa.
import course_tasks.hw0  # noqa: F401
import course_tasks.hw2  # noqa: F401
