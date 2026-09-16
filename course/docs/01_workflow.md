# 01 — The workflow (and how to see viser)

You will repeat this loop in every single assignment.

## Step 1 — Login node: edit, evaluate, visualize (no GPU needed)

mjlab requires an NVIDIA GPU **for training** only. Building an environment,
stepping it with a scripted agent, and viewing it in the browser all work on
the (GPU-less, ARM) login node:

```bash
uv run list-envs                                  # what tasks exist
uv run play Course-Cartpole-Balance --agent zero    # env sanity check
uv run play Course-Cartpole-Balance --agent random
uv run demo                                       # mjlab's built-in demo
```

`play` serves mjlab's built-in **viser** viewer over a websocket (default
port **8080**). From your
laptop, forward the port when you SSH in:

```bash
ssh -L 8080:localhost:8080 <user>@dtai-login.delta.ncsa.illinois.edu
```

then open http://localhost:8080. This "edit → play → look at it in viser" loop
on the login node is your primary debugging tool. It costs zero GPU-hours.

Etiquette: login nodes are shared. One `play` with 1 env is fine.

## Step 2 — `salloc`: short interactive GPU sessions (debugging only)

Use this when you need a real GPU in the loop, for example: confirming your env
compiles with thousands of parallel instances.
Always start it through the wrapper:

```bash
./scripts/gpu_interactive.sh          # 1 GPU, 1 hour, course account
```

The wrapper prints the **compute node hostname** (it changes every session)
and the exact tunnel command to view viser from your laptop. Compute nodes
are not reachable from outside, so the tunnel must hop through a login node:

```bash
ssh -J <user>@dtai-login.delta.ncsa.illinois.edu \
    <user>@<compute-node> -L 8080:localhost:8080
```

Interactive partitions are deliberately time-capped: they are for
*explore/debug*, not for training runs. When the clock runs out, your session
dies by design.

## Step 3 — `sbatch`: real training runs

All training goes through the course template, which bakes in the account,
partition, a GPU cap, a wall-time cap, and the output paths. Those paths are
**inside the repo**, under `logs/` — not on `/work`:

```bash
sbatch scripts/train.sbatch Course-Cartpole-Swingup --env.scene.num-envs 4096
```

Everything after the template name is passed to `uv run train` verbatim.
Monitor with:

```bash
squeue --me                      # is it queued/running?
tail -f logs/slurm-<jobid>.out
```

Checkpoints and logs land inside the repo, under `logs/rsl_rl/` —
see "Where your output lands" in `hw0/README.md` for the full layout.
Metrics go to W&B, so run `uv run wandb login` on the login node before
your first `sbatch`. View a trained policy afterwards on the login node, not compute node.

```bash
uv run play Course-Cartpole-Swingup --checkpoint-file <path/to/model_xxx.pt> --log-root logs/rsl_rl
# or, if the run logged to W&B:
uv run play Course-Cartpole-Swingup --wandb-run-path <org>/<project>/<run-id>
```

## Notebooks: VS Code on the login node (no GPU)

From HW2 on, some material comes as Jupyter notebooks, starting with
`hw2/reinforce_walkthrough.ipynb`. Run them in VS Code connected to the login
node. The notebook's kernel is the repo's own `.venv`, so it sees exactly the
environment `uv run` does.

1. On your laptop, install VS Code and its **Remote - SSH** and **Jupyter**
   extensions.
2. Command Palette → **Remote-SSH: Connect to Host…** → `deltaai` (the `Host`
   block from `docs/00_deltaai_setup.md`), then open your repo folder.
3. In VS Code's terminal, run `uv sync` once after pulling. The notebook kernel
   (`ipykernel`) is a course dependency as of HW2.
4. Open the notebook, click **Select Kernel** → **Python Environments**, and
   pick the repo's `.venv/bin/python`.

This runs on the login node's CPU and costs zero GPU-hours. Login nodes are
shared: keep `num_envs` as small as the notebook sets it, and shut the kernel
down when you finish (close the notebook tab and choose to shut the kernel
down, or **Restart** → close).

## Budget

GPU accounting is **per-project, not per-student** — one runaway job spends
everyone's hours. Per-user Slurm limits are configured on the course account,
and `train.sbatch` enforces wall-time; treat both as guardrails, not
challenges. Rule of thumb: if you haven't watched it behave under
`--agent random` in viser on the login node, it is not ready for `sbatch`.
