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
partition, a GPU cap, a wall-time cap, and output paths on `/work`:

```bash
sbatch scripts/train.sbatch Course-Cartpole-Swingup --env.scene.num-envs 4096
```

Everything after the template name is passed to `uv run train` verbatim.
Monitor with:

```bash
squeue --me                      # is it queued/running?
tail -f hw0/logs/slurm-<jobid>.out
```

Checkpoints and logs land inside the repo, under `hw0/logs/rsl_rl/`.
See "Where your output lands" in `hw0/README.md` for the full layout.
Metrics go to W&B, so run `uv run wandb login` on the login node before
your first `sbatch`. View a trained policy afterwards on the login node, not compute node.

```bash
uv run play Course-Cartpole-Swingup --checkpoint-file <path/to/model_xxx.pt> --log-root hw0/logs/rsl_rl
# or, if the run logged to W&B:
uv run play Course-Cartpole-Swingup --wandb-run-path <org>/<project>/<run-id>
```

## Budget

GPU accounting is **per-project, not per-student** — one runaway job spends
everyone's hours. Per-user Slurm limits are configured on the course account,
and `train.sbatch` enforces wall-time; treat both as guardrails, not
challenges. Rule of thumb: if you haven't watched it behave under
`--agent random` in viser on the login node, it is not ready for `sbatch`.
