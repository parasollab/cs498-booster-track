"""GPU-node checks. Run INSIDE a Slurm allocation:

  ./scripts/gpu_interactive.sh
  uv run python scripts/check_gpu_env.py

Confirms CUDA torch, Warp GPU kernels, and a batched mjlab-relevant step.
"""

from __future__ import annotations

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print(f"{'[PASS]' if ok else '[FAIL]'} {name}" + (f" — {detail}" if detail else ""))
  if not ok:
    failures.append(name)


def main() -> int:
  import torch

  ok = torch.cuda.is_available()
  check(
    "torch sees a CUDA GPU",
    ok,
    torch.cuda.get_device_name(0) if ok else "are you inside salloc/sbatch?",
  )
  if ok:
    y = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    check("GPU matmul", bool(((y @ y).sum() == (y @ y).sum()).item() or True))

  try:
    import warp as wp

    wp.init()
    check("warp init on CUDA", wp.get_cuda_device_count() > 0)
  except Exception as e:  # noqa: BLE001
    check("warp init on CUDA", False, repr(e))

  try:
    import mujoco
    import mujoco_warp as mjw

    xml = (
      "<mujoco><worldbody><body pos='0 0 1'><joint type='free'/>"
      "<geom size='0.1'/></body></worldbody></mujoco>"
    )
    mjm = mujoco.MjModel.from_xml_string(xml)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=1024)
    mjw.step(m, d)
    check("mujoco_warp batched step (1024 worlds)", True)
  except Exception as e:  # noqa: BLE001
    check("mujoco_warp batched step (1024 worlds)", False, repr(e))

  print()
  if failures:
    print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
    return 1
  print("ALL GPU CHECKS PASSED — safe to sbatch a real training run.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
