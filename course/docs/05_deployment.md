# Deploying your velocity policy on the K1

The repo and its environment are already set up on the robot and the lab machine. You only need to submit a checkpoint, check it in simulation, and run it.

## 1. Submit your checkpoint

After training the K1 flat velocity task in booster_mjlab, submit this file:

```
logs/rsl_rl/course_k1_rewards/<your_run>/<your_run>.onnx
```

- It is the **latest iteration** of that run: booster_mjlab overwrites it each time it saves a checkpoint.
- Rename it to `<yourname>.onnx`. Submitted files go in the repo's `checkpoints/` folder.
- `checkpoints/baseline.onnx` is our reference policy. Running without `--checkpoint` uses it, and it's a good comparison for your own.

## 2. Check it in simulation (recommended)

From the repo folder, with the environment active (`source .venv/bin/activate`).

Render a video of your policy. By default it runs 2 s standing, 5 s walking forward, 5 s turning, then 3 s standing:

```bash
python scripts/render_video.py --task k1_mjlab_velocity_flat \
    --checkpoint checkpoints/<yourname>.onnx --out <yourname>.mp4
```

For your own commands, list `--segment SECONDS VX VY VYAW` in order:

```bash
python scripts/render_video.py --task k1_mjlab_velocity_flat \
    --checkpoint checkpoints/<yourname>.onnx --out <yourname>.mp4 \
    --segment 2 0 0 0 --segment 5 0.5 0 0 --segment 5 0 0.3 0 --segment 5 0 0 0.8
```

If it falls or walks badly in simulation, don't run it on the robot.

## 3. Run on the robot

### Connect the controller

1. Turn on the robot.
2. Hold the controller's home button for a second or two.
3. Wait until the three white bars at the bottom of the controller (between the right thumbstick and the D-pad) are filled.

The robot's own modes are on the controller too:

- **Left trigger + minus**: damp mode
- **Left trigger + plus**: stand

### Start your policy

```bash
source .venv/bin/activate
python scripts/deploy.py --task k1_mjlab_velocity_flat --checkpoint checkpoints/<yourname>.onnx
```

1. Check that it prints `Using checkpoint .../<yourname>.onnx`.
2. Press **X**: the robot moves to the starting pose (knees bent, arms down).
3. Press **A**: your policy starts. The left stick walks forward/back and sideways; the right stick turns.
4. **Ctrl-C** ends the run, and the robot goes back to custom mode.

Start with the robot on the gantry, and use small stick inputs first.

Without the controller, use the keyboard in the same terminal: `x` for custom mode, `r` to start, `w`/`s` forward/back, `a`/`d` sideways, `q`/`e` turn, Space to stop.
