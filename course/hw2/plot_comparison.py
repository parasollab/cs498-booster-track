"""HW2 — plot your runs against environment steps. PROVIDED; run it, don't edit it.

Step 8 asks you to compare REINFORCE and PPO *at equal environment steps*, because
runs with different rollout lengths collect different amounts of data per
iteration. This reads the runs already sitting in `logs/` and draws that plot.

    uv run python hw2/plot_comparison.py                      # every HW2 run
    uv run python hw2/plot_comparison.py --metric Policy/mean_std
    uv run python hw2/plot_comparison.py logs/rsl_rl/hw2_ppo/2026-01-01_12-00-00

It reads the TensorBoard event files each run writes, so it needs no network and
works for runs made with WANDB_MODE=offline. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")  # no display on the login node
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
  EventAccumulator,
)

X_TAG = "Train/total_env_steps"
DEFAULT_GLOB = "logs/rsl_rl/hw2_*/*"


def read_run(run_dir: str, metric: str) -> tuple[list[float], list[float]]:
  """Return (env_steps, metric) for one run directory, aligned by iteration."""
  acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
  acc.Reload()
  tags = acc.Tags().get("scalars", [])
  for needed in (X_TAG, metric):
    if needed not in tags:
      raise KeyError(
        f"{run_dir}: no '{needed}' logged. Tags present: {sorted(tags) or 'none'}"
      )
  steps = {e.step: e.value for e in acc.Scalars(X_TAG)}
  ys = [(e.step, e.value) for e in acc.Scalars(metric)]
  # mean_reward only appears once an episode has finished, so keep the
  # iterations that logged both.
  pairs = [(steps[it], v) for it, v in ys if it in steps]
  return [p[0] for p in pairs], [p[1] for p in pairs]


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("runs", nargs="*", help=f"run directories (default: {DEFAULT_GLOB})")
  p.add_argument("--metric", default="Train/mean_reward", help="scalar to plot")
  p.add_argument("--out", default="hw2/comparison.png", help="output image")
  args = p.parse_args()

  runs = args.runs or sorted(d for d in glob.glob(DEFAULT_GLOB) if os.path.isdir(d))
  if not runs:
    raise SystemExit(
      f"No runs found under {DEFAULT_GLOB}. Train something first, or pass run "
      "directories on the command line."
    )

  fig, ax = plt.subplots(figsize=(7, 4.5))
  plotted = 0
  for run in runs:
    label = os.path.basename(run.rstrip("/")) or run
    try:
      x, y = read_run(run, args.metric)
    except KeyError as e:
      print(f"skipping {e}")
      continue
    if not x:
      print(f"skipping {run}: nothing logged yet")
      continue
    ax.plot(x, y, label=label)
    print(f"{label:44s} {len(x):4d} points, final {args.metric} = {y[-1]:.3f}")
    plotted += 1

  if not plotted:
    raise SystemExit("Nothing to plot.")

  ax.set_xlabel("environment steps")
  ax.set_ylabel(args.metric)
  ax.set_title(f"{args.metric} vs environment steps")
  ax.legend(fontsize="small")
  ax.grid(alpha=0.3)
  fig.tight_layout()
  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  fig.savefig(args.out, dpi=150)
  print(f"wrote {args.out} ({plotted} run(s))")


if __name__ == "__main__":
  main()
