# HW0 — mjlab fundamentals: cartpole and double cartpole

**What you'll learn.**

1. Keep a healthy `uv` environment on DeltaAI
2. Evaluate and *look at* an environment in viser from the login node
3. Read and write an mjlab manager-based environment config
4. Submit training with `sbatch` and monitor it
5. Play back a trained checkpoint

**What you'll build.** A double cartpole (two-link pendulum) environment
with two variants : `Course-Cartpole-Double-Balance` and
`Course-Cartpole-Double-Swingup` with a trained policy that solves swingup.

**Grading.** You will be graded on your write up as well as showing proof the training of your policy is improving at all and has run for sufficient iterations.

**Time estimate.** Steps 1–4 in one sitting (~3–4 h). Steps 5–7 may require more time because training jobs queue, start early.

---

## Assignment Steps

| Step | What | Where | Graded? |
|------|------|-------|---------|
| 1 | Verify your environment | login node | no, but required |
| 2 | Drive the mjlab CLI + viser | login node | no |
|2.5| Read MjLab Docs | Browser | no (but quizzes might test knowledge) |
| 3 | Guided cartpole walkthrough | login node | questions to answer and include in write-up |
| 4 | Implement the double cartpole | login node | **yes** |
| 5 | Train your double cartpole | sbatch + login node | **yes** |
| 6 | Write up and submit to Canvas | — | **yes** |

Six questions appear in Steps 1–5. Answer all of them in your writeup.

---
## Step 0 — Don't waste GPU hours (read this first)

The whole class shares one GPU allocation. Three scripts in `scripts/` help
you not waste it:

```bash
./scripts/my_jobs.sh       # what's running, what finished, what failed
./scripts/my_usage.sh      # my GPU-hours + class total
./scripts/kill_my_jobs.sh  # cancel my jobs (asks first)
```

Rules:

- Run `my_usage.sh` before submitting jobs. Over the assignment budget? Office hours, not more runs.
- Run `my_jobs.sh` before logging off so you don't find a dead job two days later.
- Submitted a bad job (wrong reward, NaNs, duplicate)? `kill_my_jobs.sh` right away. Saved checkpoints are kept.

Anything in viser on the login node costs zero GPU-hours.

You can also manage jobs directly with Slurm:

```bash
squeue --me                  # list your jobs + job ids
scancel <jobid>              # cancel one job
scancel <jobid1> <jobid2>    # cancel several
scancel --me                 # cancel ALL your jobs (no confirmation — careful)
scancel --me --state=PENDING # cancel only queued jobs, keep running ones
```

`scancel` only works on your own jobs.

## Step 1 — Verify your environment

Work through `docs/00_deltaai_setup.md` if you haven't, then from the repo
root on a login node:

```bash
uv run python scripts/check_login_env.py
```

**Done when:** every line is `[PASS]`, including "CUDA unavailable
(expected on login node)".


## Step 2 — Drive the mjlab CLI and viser

mjlab is heavily CLI-based. Four commands cover 90% of this course:

```bash
uv run list-envs                     # every registered task id
uv run play Course-Cartpole-Balance --agent zero
uv run play Course-Cartpole-Balance --agent random
uv run demo                          # mjlab's showcase demo
```

`play` serves the **viser** viewer on port 8080. Open it from your laptop
through the SSH tunnel described in `docs/01_workflow.md`. Orbit the camera,
find the reward readout, watch what `--agent random` does to the pole.

Remember this loop: **zero/random agent + viser on the login node** is how
you debug every environment you will ever build here. It exercises loading,
resets, observations, and rewards (everything except learning) and costs
zero GPU-hours.

**Done when:** you can see the cartpole moving in your browser via the
tunnel, for both `--agent zero` and `--agent random`.

## Step 3 — Guided cartpole walkthrough

Open `src/course_tasks/hw0/cartpole_env_cfg.py` and `cartpole.xml` side by
side, with viser running. The config file is annotated as a tour with five
numbered sections; for each, verify its claims against what you see:

1. **XML → EntityCfg.** Find `slider`, `hinge_1`, and the `slide` motor in
   the XML.
   ★ **Q1:** Which joints are actuated, and how can you tell from the XML
   alone?
2. **Observations.** Terms are functions returning `[num_envs, dim]` tensors.
   ★ **Q2:** What is the total actor observation dimension, and why is the
   pole angle 2 numbers instead of 1?
3. **Rewards.** A product of four shaped factors in [0, 1].
   ★ **Q3:** With `--agent zero` in Balance mode, roughly what reward does
   viser show, and why isn't it exactly 1.0? (Hint: reset events.)
4. **Events & terminations.** Temporarily change `position_range` of
   `reset_hinge` to `(-1.0, 1.0)`, rerun `play --agent zero`, watch the
   resets, then **revert your change**.
5. **Registration.** Trace how `Course-Cartpole-Balance` travels from
   `hw0/__init__.py` into `list-envs` (hint: `pyproject.toml`, entry point
   `mjlab.tasks`). Your task in Step 4 appears the same way.

**Done when:** you've answered Q1–Q3 to be included in your write-up.

## Step 4 — Implement the double cartpole (graded)

`cartpole_double.xml` gives the cart a two-link pole: `hinge_1` joins link 1 to the cart, `hinge_2` joins link 2 to link 1. Complete the three TODOs in `src/course_tasks/hw0/cartpole_double_env_cfg.py`.

**Do not modify:** the `joint_velocity_limit_exceeded` termination, or the PPO runner configs at the bottom (kept fixed for comparable, reproducible results).

**4a.** TODO(1) — entity and initial states. TODO(3) — observations, actions, events, `time_out`, plus the provided velocity termination. Leave the reward returning zeros for now.

*Checkpoint:* `uv run list-envs` lists both `Course-Cartpole-Double-*` tasks. Until then you'll see a warning naming what's missing.

**4b.** Watch it. On the login node pass `--env.scene.num-envs 4`; the config defaults to 1024 parallel envs, which is a lot for CPU.

```bash
uv run play Course-Cartpole-Double-Swingup --agent zero --env.scene.num-envs 4    # links hang straight down, small reset jitter
uv run play Course-Cartpole-Double-Balance --agent zero --env.scene.num-envs 4    # links balance upright until jitter topples them
uv run play Course-Cartpole-Double-Swingup --agent random --env.scene.num-envs 4  # cart shakes, links flail
```

*Checkpoint:* physics looks right in viser. Swingup must hang *straight* down — a kinked chain means your `hinge_2` initial value in TODO(1) is wrong. **Do not continue until it does.**

**4c.** TODO(2) — your reward. Check in viser: hanging down ≈ 0, upright ≈ 1. Then confirm it stays batched:

```bash
uv run play Course-Cartpole-Double-Swingup --agent random --env.scene.num-envs 16
```

**Rules**
- No Python loops over environments.
- Keep the public function names and signatures — grading scripts import them.
- Don't touch the provided termination or PPO configs.
- Scene entity name must be `"cartpole"`.

**Done when:** every checkpoint passes and 16 envs run without errors.

## Step 5 — Sanity-train cartpole swingup

Before spending GPU-hours on *your* env, prove the pipeline with a
known-good task:

```bash
sbatch scripts/train.sbatch Course-Cartpole-Swingup --env.scene.num-envs 4096
squeue --me
tail -f logs/slurm-<jobid>.out
```

### Where your output lands

Everything a training job produces stays **inside this repo**, under
`logs/`. Always submit with `sbatch` from the repo root — the paths
below are relative to it.

> **Changed in HW2.** This used to be `hw0/logs/`. There is now one log root,
> `logs/`, shared by every assignment; runs stay separated by the
> `<experiment>` directory below. If you trained before this change, those runs
> are still in `hw0/logs/` and nothing moved them — the commands here just point
> at the new location from now on.

```
logs/
├── slurm-<jobid>.out                    job stdout/stderr (training curve prints here)
├── wandb/                               W&B's local run cache
└── rsl_rl/
    └── <experiment>/                    hw0_cartpole | cartpole_double
        └── <YYYY-MM-DD_HH-MM-SS>/       one directory per run
            ├── model_<iter>.pt          checkpoints, saved periodically
            ├── params/env.yaml          the exact env config that ran
            ├── params/agent.yaml        the exact PPO config that ran
            ├── git/*.diff               your uncommitted changes at launch
            └── events.out.tfevents.*    TensorBoard scalars (written alongside W&B)
```

`<experiment>` is set by the task's PPO runner config: `hw0_cartpole` for
`Course-Cartpole-*`, `cartpole_double` for `Course-Cartpole-Double-*`.
Runs are timestamped, so a re-train never overwrites an earlier one.

### Log in to W&B first (once)

Training logs metrics to **Weights & Biases**, so reward curves live in the
browser rather than in the `.out` file. Authenticate **on the login node
before your first `sbatch`**. The credential is written to `~/.netrc`, and
because `$HOME` is shared, your compute job picks it up automatically:

```bash
uv run wandb login      # paste the key from https://wandb.ai/authorize
```

Skip this and the job has no credential to use, which costs you the GPU-hours
the run burns before you notice. Checkpoints are unaffected — they are always
written to `logs/rsl_rl/` regardless of what W&B does.

Two things:

* **Checkpoints are git-ignored** (`*.pt` and everything under `logs/`
  except the directory itself).
* Find your most recent run without typing a timestamp:

  ```bash
  ls -t logs/rsl_rl/cartpole_double/*/model_*.pt | head -1
  ```

Cartpole swingup should visibly learn within the run. If it doesn't, your
problem is setup.

★ **Q4:** From the log: roughly how many env-steps/second did you get?
★ **Q5:** Play the resulting checkpoint (use the Step 6 command with the
cartpole task). Describe in one or two sentences what the policy does.

**Done when:** you've watched a trained cartpole swingup succeed in viser.

## Step 6 — Train your double cartpole (graded, iterative)

Train Balance first, then Swingup:

```bash
sbatch scripts/train.sbatch Course-Cartpole-Double-Balance --env.scene.num-envs 4096
# once Balance visibly works:
sbatch scripts/train.sbatch Course-Cartpole-Double-Swingup --env.scene.num-envs 4096
```

The training hyperparameters are fixed course-wide (they're the provided
configs in your file). If training fails, the fix is in your **reward or
environment**, not in PPO settings.

When a run has saved checkpoints you can play it back with:

```bash
uv run play Course-Cartpole-Double-Swingup \
  --checkpoint-file logs/rsl_rl/cartpole_double/<run>/model_<iter>.pt \
  --log-root logs/rsl_rl
```

Expect to iterate: watch → adjust your reward → retrain. Balancing one link
while windmilling the other is a classic first outcome; what you changed and
why is exactly what the writeup wants.

**Budget:** Balance + Swingup should be solvable well within 2–3 sbatch runs
of the template's wall-time. If you're on run 6, come to office hours
instead of submitting run 7.

**Done when:** in viser, a *Swingup* checkpoint swings both links up and
balances them with the cart staying near the center.

## Step 7 — Write up and submit

Write `hw0/writeup.md/txt/word/doc`, at most 2 pages, containing:

- Answers to ★ Q1–Q5
- Your reward design: the terms, why you chose them, and what you changed
  between training iterations (with what you observed that prompted each
  change) — remember the PPO configs are fixed, so reward design is your
  only lever
- Two short explanations: (a) why the provided velocity-limit termination
  exists and when it fires, and (b) why there is deliberately *no*
  "pole fell over" failure termination in the swingup task
- A screenshot or exported plot of your reward curve
- The task id, run directory name, and checkpoint filename you want graded

### What to submit (exactly these four things)

Via the course submission page (not email, not a git push to `main`):

1. **`cartpole_double_env_cfg.py`** 
2. **`writeup.md/txt/word/doc`** 
3. **the latest checkpoint folder**, found at logs/rsl_rl/<env_name>/latest_checkpoint_timestamp


Before submitting, run the same smoke test we will:

```bash
uv run python scripts/check_login_env.py &&
uv run play Course-Cartpole-Double-Swingup --agent random --env.scene.num-envs 16
```

---
