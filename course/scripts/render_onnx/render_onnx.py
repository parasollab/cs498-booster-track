"""Render student K1 walking policies (.onnx) to mp4, headless, for grading.

Runs each ONNX in the same mjlab env the students trained in
(Course-Booster-K1-Rewards, play config) with a scripted velocity command,
records an mp4 with a text overlay, and writes summary.csv with commanded vs.
measured velocity and falls per segment.

Run from anywhere (login node is fine, no GPU needed):

  ./render_onnx.sh path/to/onnx_dir_or_files ... [--out-dir videos]

Default commands mirror the deploy repo's render_video.py:
  2 s stand, 5 s forward 0.5 m/s, 5 s turn 0.5 rad/s, 3 s stand.
Override with repeated --segment SECONDS VX VY VYAW.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Headless rendering. With no NVIDIA driver (login nodes), force Mesa's
# software EGL so MuJoCo's offscreen renderer works.
os.environ.setdefault("MUJOCO_GL", "egl")
if not Path("/dev/nvidia0").exists():
    os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    )
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

COURSE_DIR = Path(os.environ.get(
    "COURSE_DIR", Path(__file__).resolve().parents[1] / "cs498-booster-track/course"))
HW3_DIR = COURSE_DIR / "hw3"
sys.path.insert(0, str(HW3_DIR))  # for course_booster (registers the Course tasks)

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import course_booster  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

DEFAULT_SEGMENTS = [(2.0, 0, 0, 0), (5.0, 0.5, 0, 0), (5.0, 0, 0, 0.5), (3.0, 0, 0, 0)]


def build_env(args):
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.viewer.width, cfg.viewer.height = args.width, args.height

    # Make the twist command a pure, externally-set command.
    cmd = cfg.commands["twist"]
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_standing_envs = 0.0
    cmd.rel_world_envs = 0.0
    cmd.rel_forward_envs = 0.0
    cmd.init_velocity_prob = 0.0
    cmd.resampling_time_range = (1e9, 1e9)
    cfg.curriculum = {}
    # Rewards don't affect rollouts; dropping them also avoids the course
    # template's unimplemented reward stubs in course_booster/rewards.py.
    cfg.rewards = {}

    if not args.pushes:
        for name in [n for n in cfg.events if "push" in n]:
            cfg.events.pop(name)

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
    agent_cfg = load_rl_cfg(args.task)
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def check_metadata(model_path: Path, env: ManagerBasedRlEnv) -> list[str]:
    """Compare ONNX export metadata to the env; return human-readable warnings."""
    md = {p.key: p.value for p in onnx.load(str(model_path)).metadata_props}
    warnings = []
    obs_names = env.observation_manager.active_terms["actor"]
    if "observation_names" in md and md["observation_names"].split(",") != list(obs_names):
        warnings.append(f"obs terms differ: onnx={md['observation_names']} env={','.join(obs_names)}")
    robot = env.scene["robot"]
    if "joint_names" in md:
        actuated = {n for a in robot.actuators for n in a.target_names}
        env_joints = [n for n in robot.joint_names if n in actuated]
        if md["joint_names"].split(",") != env_joints:
            warnings.append("joint_names differ from env")
    return warnings


def overlay(frame: np.ndarray, lines: list[str], fell: bool) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        y += 16
    if fell:
        draw.text((8, y), "FELL / RESET", fill=(255, 60, 60), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def run_one(model_path: Path, venv: RslRlVecEnvWrapper, args) -> list[dict]:
    env: ManagerBasedRlEnv = venv.unwrapped
    name = model_path.stem
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    env.reset(seed=args.seed)
    obs = venv.get_observations()["actor"]
    want = sess.get_inputs()[0].shape[-1]
    if isinstance(want, int) and want != obs.shape[-1]:
        raise RuntimeError(f"obs dim mismatch: onnx expects {want}, env gives {obs.shape[-1]}")
    for w in check_metadata(model_path, env):
        print(f"  [WARN] {name}: {w}")

    twist = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    dt = env.step_dt
    render_every = max(1, round(1.0 / (args.fps * dt)))
    fps = 1.0 / (render_every * dt)

    out_path = Path(args.out_dir) / f"{name}.mp4"
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    rows, t, fell_flash = [], 0.0, 0
    try:
        for seg_i, (dur, vx, vy, wz) in enumerate(args.segments):
            command = torch.tensor([vx, vy, wz], dtype=torch.float32)
            n_steps = int(round(dur / dt))
            meas, falls = [], 0
            for k in range(n_steps):
                twist.vel_command_b[:] = command
                action = sess.run(None, {in_name: obs.numpy().astype(np.float32)})[0]
                obs_td, _, dones, _ = venv.step(torch.from_numpy(action))
                obs = obs_td["actor"]
                t += dt
                if bool(dones.any()):
                    falls += 1
                    fell_flash = int(1.0 / dt)
                    # Reset resampled the command; restore it and refresh obs.
                    twist.vel_command_b[:] = command
                    obs = venv.get_observations()["actor"]
                lin = robot.data.root_link_lin_vel_b[0].cpu().numpy()
                ang = robot.data.root_link_ang_vel_b[0].cpu().numpy()
                h = float(robot.data.root_link_pos_w[0, 2])
                if k >= n_steps * 0.4:  # skip the transient at segment start
                    meas.append((lin[0], lin[1], ang[2], h))
                if k % render_every == 0:
                    lines = [
                        f"{name}   t={t:5.2f}s   seg {seg_i + 1}/{len(args.segments)}",
                        f"cmd  vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}",
                        f"act  vx={lin[0]:+.2f} vy={lin[1]:+.2f} wz={ang[2]:+.2f}  h={h:.2f}",
                    ]
                    writer.append_data(overlay(env.render(), lines, fell_flash > 0))
                fell_flash = max(0, fell_flash - 1)
            m = np.mean(meas, axis=0) if meas else [np.nan] * 4
            rows.append(dict(policy=name, segment=seg_i + 1, duration=dur,
                             cmd_vx=vx, cmd_vy=vy, cmd_wz=wz,
                             mean_vx=round(float(m[0]), 3), mean_vy=round(float(m[1]), 3),
                             mean_wz=round(float(m[2]), 3), mean_height=round(float(m[3]), 3),
                             falls=falls))
    finally:
        writer.close()
    total_falls = sum(r["falls"] for r in rows)
    print(f"  wrote {out_path}  (falls: {total_falls})")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help=".onnx files and/or directories containing them")
    p.add_argument("--out-dir", default="onnx_videos")
    p.add_argument("--summary", default=None, help="summary csv path (default OUT_DIR/summary.csv)")
    p.add_argument("--task", default="Course-Booster-K1-Rewards")
    p.add_argument("--segment", nargs=4, type=float, action="append", metavar=("SECONDS", "VX", "VY", "VYAW"))
    p.add_argument("--pushes", action="store_true", help="keep random pushes (off by default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()
    args.segments = [tuple(s) for s in args.segment] if args.segment else DEFAULT_SEGMENTS

    models = []
    for inp in map(Path, args.inputs):
        models += sorted(inp.rglob("*.onnx")) if inp.is_dir() else [inp]
    if not models:
        raise SystemExit("No .onnx files found.")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    venv = build_env(args)
    all_rows = []
    for i, mp in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {mp}")
        try:
            all_rows += run_one(mp, venv, args)
        except Exception as e:  # keep going for the rest of the class
            print(f"  [ERROR] {mp.name}: {e}")
            all_rows.append(dict(policy=mp.stem, segment="ERROR", duration=str(e)))

    summary = Path(args.summary) if args.summary else Path(args.out_dir) / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    keys = ["policy", "segment", "duration", "cmd_vx", "cmd_vy", "cmd_wz",
            "mean_vx", "mean_vy", "mean_wz", "mean_height", "falls"]
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Summary: {summary}")
    sys.stdout.flush()
    # Skip interpreter teardown: Mesa EGL raises noisy errors when MuJoCo's
    # renderer is garbage-collected after the display is gone.
    os._exit(0)


if __name__ == "__main__":
    main()
