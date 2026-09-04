"""Login-node environment checks. Run: uv run python scripts/check_login_env.py

Everything here must pass on a DeltaAI login node (gh-login0X): no GPU, no
display, aarch64. CUDA being unavailable is EXPECTED and treated as a pass.
"""

from __future__ import annotations

import platform
import sys

PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[warn]"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print(f"{PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))
  if not ok:
    failures.append(name)


def main() -> int:
  # 1. Platform.
  arch = platform.machine()
  check("python >= 3.13", sys.version_info >= (3, 13), platform.python_version())
  if arch != "aarch64":
    print(f"{WARN} architecture is {arch}, not aarch64 — fine on a laptop, "
          "wrong if you think you're on DeltaAI")

  # 2. Torch imports and does CPU work; CUDA absence is expected here.
  try:
    import torch

    x = torch.randn(64, 64, requires_grad=True)
    (x @ x).sum().backward()
    check("torch import + CPU autograd", x.grad is not None, torch.__version__)
    if torch.cuda.is_available():
      print(f"{WARN} CUDA available — you're probably NOT on a login node")
    else:
      print(f"{PASS} CUDA unavailable (expected on login node)")
  except Exception as e:  # noqa: BLE001
    check("torch import + CPU autograd", False, repr(e))

  # 3. Plain MuJoCo physics stepping on CPU.
  try:
    import mujoco

    model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><body><joint type='free'/>"
      "<geom size='0.1'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    for _ in range(100):
      mujoco.mj_step(model, data)
    check("mujoco CPU stepping (100 steps)", True, mujoco.__version__)
  except Exception as e:  # noqa: BLE001
    check("mujoco CPU stepping (100 steps)", False, repr(e))

  # 4. mjlab imports and course tasks are registered via the entry point.
  try:
    import mjlab.tasks  # noqa: F401  (populates registry, incl. entry points)
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    course = [t for t in tasks if t.startswith("Course-")]
    check(
      "course tasks registered",
      len(course) > 0,
      ", ".join(course) if course else f"only found: {tasks[:5]}...",
    )
  except Exception as e:  # noqa: BLE001
    check("mjlab import / task registry", False, repr(e))

  # 5. viser importable (the viewer used by `uv run play` / `demo`).
  try:
    import viser  # noqa: F401

    check("viser import", True)
  except Exception as e:  # noqa: BLE001
    check("viser import", False, repr(e))

  print()
  if failures:
    print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
    return 1
  print("ALL CHECKS PASSED — environment is ready.")
  print("Next: `uv run play Course-Cartpole-Balance --agent zero` and open the")
  print("viser URL through your SSH tunnel (docs/01_workflow.md).")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
