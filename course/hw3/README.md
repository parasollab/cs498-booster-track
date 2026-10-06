# HW3 — Booster K1: implement and evaluate locomotion rewards

**What you'll learn.** Turn physical control objectives into batched reward
functions, check them before spending GPU hours, and see how a reward change
alters learned behavior. PPO and the simulator are provided. Your work is seven
reward implementations, a trained walking policy, and one justified reward variant.

**This homework feeds the robot deployment.** The policy you train here is
the one your team will put on the real Booster K1. Checkpoint 1 is about
getting a policy that walks in simulation and packaging it for deployment.

## Deadlines and checkpoints

| Checkpoint | Due | What you submit |
|---|---|---|
| **Checkpoint 1** | **Monday 10/12** | A video of your policy walking in the viewer, the files needed for deployment (see Step 5), and which team member's policy your team will deploy for Checkpoint 2 |
| **Checkpoint 2** | **Monday 10/19** | Deployment on the real robot, by signup slot |

A deployment guide and signup sheet will be posted in this repository and on
Canvas later this week. **[TA: add link once posted.]**

## Read this first: cluster time is the bottleneck

- **Queue times on Delta and DeltaAI are currently long**, often a day or more.
  Submit your first real training job as early as you can, ideally by midweek.
- **Use the full four-hour allocation.** The course sbatch templates request
  `--time=04:00:00`. If you shortened your copy for HW2, restore it: a 30-minute
  job will not get you a usable walking policy.
- **Log in to W&B before every GPU submission.** Run `uv run wandb login` on
  the login node (see the [workflow guide](../docs/01_workflow.md)). The CPU
  smoke test in Step 3 deliberately sets `WANDB_MODE=offline`; do **not** carry
  that into your sbatch command.
- Use `scripts/my_jobs.sh`, `scripts/my_usage.sh` and `scripts/kill_my_jobs.sh`
  to monitor jobs. Do not alter the fixed cluster resource requests.

## Assignment steps

| Step | Work | Where |
|---|---|---|
| 1 | Set up the isolated project and read the MDP | login node or laptop |
| 2 | Implement and check seven rewards | laptop or login node (CPU) |
| 3 | Run the CPU smoke test | laptop or login node (CPU) |
| 4 | Train on the cluster | course sbatch |
| 5 | Play back, record, export, and package for deployment | login node + viewer |
| 6 | One reward variant and a short write-up | cluster + write-up |

As in HW2, debug the math before interpreting a reward curve.

## Step 1 — Set up and understand the task

Pull the latest `main`. Run every command below from the **nested course
Python-project root** (the folder with `pyproject.toml`, `scripts/`, `hw3/`):

```bash
cd <your-repo>/course
uv sync --project hw3/vendor/booster_mjlab --locked --no-dev     # first run downloads Torch + MuJoCo, several minutes
uv run --project hw3/vendor/booster_mjlab --locked --no-dev \
  python hw3/run.py list
uv run --project hw3/vendor/booster_mjlab --locked --no-dev \
  python hw3/run.py check
```

**Every HW3 command uses `uv run --project hw3/vendor/booster_mjlab`.** Booster
pins a different mjlab revision from HW0/HW2, so it has its own environment
inside the vendored project. Plain `uv run train ...` from HW2 will not find
the Booster tasks. Leave the main course lockfile alone. The vendor README is
historical upstream documentation, not this assignment's instructions.
Expect the first `run.py check` and the reward checks to take a minute or
two while Warp compiles kernels.

Setup guides: [DeltaAI](../docs/00_deltaai_setup.md), [Delta](../docs/03_delta_setup.md),
[workflow and viewer](../docs/01_workflow.md).

### The three tasks

| Task | What it is | Experiment folder |
|---|---|---|
| `Course-Booster-K1-Rewards` | **Your task.** Seven rewards you implement plus eight provided shaping terms | `course_k1_rewards` |
| `Mjlab-Velocity-Flat-Booster-K1` | The unchanged upstream 15-reward baseline, for comparison | `k1_velocity` |
| `Course-Booster-K1-Rewards-Candidate` | Optional ablation: your task minus the angular-momentum and swing-height shaping | `course_k1_rewards_candidate` |

Your implementations go into `Course-Booster-K1-Rewards`. It intentionally
uses different reward formulas from upstream while keeping every other MDP
and PPO setting identical, so you can compare against the baseline.

### The control problem

The robot receives body-frame commands for forward/lateral velocity and yaw
rate. The policy outputs **22 scaled joint-position targets** offset from the
default standing pose. Physics runs at 200 Hz; the policy runs at 50 Hz.
Actuator delay and domain randomization are provided and stay on.

| Actor observation | Dims | Meaning |
|---|---:|---|
| IMU angular velocity | 3 | body-frame roll, pitch, yaw rates |
| Projected gravity | 3 | unit gravity vector in the body frame; `(0, 0, -1)` when upright |
| Joint positions | 22 | relative to the default pose |
| Joint velocities | 22 | |
| Previous actions | 22 | the last raw policy output |
| Command | 3 | commanded vx, vy, yaw rate |

Actor total **75** with observation noise and encoder bias. The critic gets
**90** inputs: the same terms without noise, plus privileged linear velocity,
foot heights, air times, contacts and contact forces. The deployed actor never
sees those. **Rewards do not use the noisy observations.** They read the true
simulator state through `env.scene["robot"].data`; the docstring at the top of
[rewards.py](course_booster/rewards.py) lists every tensor you need with its
shape and meaning. The [MDP inventory](MDP.md) goes deeper on observations,
rewards, events, terminations and curricula.

## Step 2 — Implement seven rewards

Edit only [course_booster/rewards.py](course_booster/rewards.py) and, when you
get to the variant, `STUDENT_WEIGHTS` / `STUDENT_PARAMS` in
[course_booster/env_cfg.py](course_booster/env_cfg.py). Keep function names
and signatures. Do not edit vendor code, observations, sensors, PPO or launchers.

Each function already fetches the state it needs and has a TODO spelling out
the formula. The table says what each one is for and what sign to expect.

| Function | What it rewards or penalizes | Sign | Weight |
|---|---|---|---:|
| `track_linear_velocity` | Moving at the commanded planar velocity, with a tolerance that scales with commanded speed; also penalizes vertical bouncing and gives partial credit for "getting closer" | score in [0, 1] | +2.25 |
| `track_angular_velocity` | Turning at the commanded yaw rate | score in [0, 1] | +2.0 |
| `upright` | Keeping the trunk vertical; inverted must score 0, not 1 | score in [0, 1] | +1.0 |
| `action_smoothness` | Jerky actions: squared step-to-step change plus a weighted squared second difference | penalty ≥ 0 | −0.1 |
| `standing_pose` | Drifting from the default pose or moving joints when commanded to stand still; fades out as the command grows | penalty ≥ 0 | −0.5 |
| `air_time` | Swing phases of a sensible duration while walking; zero when standing | score in [0, 2] | +0.1 |
| `foot_slip` | A planted foot sliding horizontally, even when standing | penalty ≥ 0 | −0.2 |

Penalties return a **positive** magnitude; the negative weight in `env_cfg.py`
makes them costs. The reward manager multiplies by the weight and the control
timestep; never do that inside your function.

Rules for all seven: return a finite tensor of shape `(env.num_envs,)` on
`env.device`. Reduce over joints, feet or axes, never over environments. No
Python loops over environments, no `.cpu()`, no `.item()`. The checker rejects
`for`/`while` loops and comprehensions in the file.

Run the public checks:

```bash
uv run --project hw3/vendor/booster_mjlab --locked --no-dev \
  python hw3/tests/check_rewards.py
```

These check shapes, devices, signs and a handful of properties (better
tracking scores higher, inverted is not upright, gates switch off, and so on).
**They are necessary conditions, not a proof of correctness.** Several wrong
formulas pass them. Build your own small tests with
[tests/reward_fixtures.py](tests/reward_fixtures.py): set a perfect-tracking
state and a bad one, a turning command versus a standing one, an upside-down
gravity vector, one foot moving while the other is planted. Write down why you
expect each number. You will include a few of these in your write-up.

`run.py check` confirms the tasks register and that nothing outside the
reward layer drifted from the baseline. It does not judge your rewards.

## Step 3 — Smoke-test on CPU before touching a GPU

```bash
uv run --project hw3/vendor/booster_mjlab --locked --no-dev \
  python hw3/tests/smoke_env.py --task Course-Booster-K1-Rewards

WANDB_MODE=offline OMP_NUM_THREADS=1 \
uv run --project hw3/vendor/booster_mjlab --locked --no-dev \
  python hw3/run.py train Course-Booster-K1-Rewards \
  --gpu-ids None --env.scene.num-envs 2 --agent.max-iterations 1 \
  --agent.save-interval 1 --agent.logger tensorboard --agent.run-name cpu_smoke
```

The first command resets the environment and steps it three times with your
rewards in the loop. The second runs one PPO iteration on two environments,
saves a checkpoint and exports ONNX. Both finish in a couple of minutes on a
laptop. `WANDB_MODE=offline` is only for this throwaway run; your cluster
jobs must log online. This tiny run says nothing about policy quality.

See all options with:

```bash
uv run --project hw3/vendor/booster_mjlab --locked --no-dev \
  python hw3/run.py train Course-Booster-K1-Rewards --help
```

## Step 4 — Train on the cluster

Log in to W&B first, then submit from the nested project root using the
template for the cluster you are on:

```bash
uv run wandb login     # once per login session; paste your key from https://wandb.ai/authorize

# DeltaAI
sbatch scripts/train_deltaai.sbatch Course-Booster-K1-Rewards \
  --env.scene.num-envs 4096 --agent.run-name reference
# Delta
sbatch scripts/train_delta.sbatch Course-Booster-K1-Rewards \
  --env.scene.num-envs 4096 --agent.run-name reference
```

Arguments after the task ID are forwarded unchanged. Both templates request
one GPU, 16 CPUs and **four hours**. The sbatch script routes Booster tasks
through the isolated project automatically. Single-GPU only.

**How long to train.** Default PPO is 30,000 iterations with 24 steps per
environment per iteration. That will not finish inside four hours. For scale,
1,000 iterations at 4,096 environments took about 32 minutes on a desktop
RTX 5090, so expect roughly **5,000 to 7,000 iterations per four-hour job**
depending on the cluster GPU. Checkpoints and ONNX exports are saved every
50 iterations, so you keep the latest one if the job hits its time limit.
If the policy is not walking well yet, resume from the last checkpoint
(Step 5) in a second job. Budget for **at least two four-hour jobs plus
queue time** before Checkpoint 1.

Before your first big job, a quick sanity submission catches path and login
problems without burning hours:

```bash
sbatch scripts/train_deltaai.sbatch Course-Booster-K1-Rewards \
  --env.scene.num-envs 4096 --agent.max-iterations 3 --agent.run-name sanity
```

Its Slurm log prints `Iteration time:` for each iteration. Divide 3.5 hours
by that number and pass the result as `--agent.max-iterations` on your real
job. A run that finishes on its own writes a final checkpoint and ONNX; a run
killed by the time limit only has its last 50-iteration save.

## Step 5 — Inspect, resume, play back, and package for deployment

Everything a run produces lands in one folder:

```text
logs/slurm-<jobid>.out
logs/rsl_rl/course_k1_rewards/<timestamp>_<run-name>/
  model_<iteration>.pt          checkpoints, every 50 iterations
  params/env.yaml               the exact environment that ran
  params/agent.yaml             the exact PPO settings that ran
  events.out.tfevents.*         every logged scalar
  <timestamp>_<run-name>.onnx   exported policy, rewritten at every save
```

Watch `Train/mean_reward`, the per-term episode rewards and
`Policy/mean_std` in W&B. The ONNX export can print warnings and still
succeed: check the file exists rather than assuming. The saved git diff does
not include untracked files, so keep your edited `rewards.py` and
`env_cfg.py` with your run record.

**Resume** a run to add more iterations:

```bash
sbatch scripts/train_deltaai.sbatch Course-Booster-K1-Rewards \
  --env.scene.num-envs 4096 --agent.resume True \
  --agent.load-run '<timestamp>_<run-name>' \
  --agent.load-checkpoint 'model_<iteration>.pt' \
  --agent.max-iterations 5000 --agent.run-name resumed
```

Resume creates a new run folder, restores the policy, optimizer and
curriculum stage, and trains for **additional** iterations. It does not
restore the simulator state or RNG. Iteration numbers restart at zero in the
new folder, so count segments when you compare runs.

**Play back and record.** Use the SSH tunnel from the viewer guide:

```bash
uv run --project hw3/vendor/booster_mjlab --locked --no-dev \
  python hw3/run.py play Course-Booster-K1-Rewards \
  --checkpoint-file 'logs/rsl_rl/course_k1_rewards/<run>/model_<iteration>.pt' \
  --num-envs 4 --device cpu --viewer viser
```

Playback uses `--num-envs` (train uses `--env.scene.num-envs`). Recording
goes to `logs/videos/viser` or wherever `--record-dir` points. Playback turns
off observation noise and the illegal-contact termination, keeps pushes, and
widens the command range, so it looks a bit more forgiving than training.

**Checkpoint 1 deliverables** (one submission per team on Canvas):

1. A short video of your policy walking forward, turning, and standing still
   from the viewer recording. **[TA: confirm length and required commands.]**
2. The run folder for the policy you are submitting: the final `model_*.pt`,
   the `.onnx` export, `params/`, and the Slurm log.
3. Your `rewards.py` and `env_cfg.py` as trained.
4. The name of the team member whose policy will be deployed in Checkpoint 2.

The deployment guide will state the exact files the robot needs.
**[TA: link once posted.]**

## Step 6 — One reward variant and the write-up

Pick **one** change and justify it before you train it: a weight, a parameter
such as the air-time window or a tolerance, or a formula change in one of your
seven functions. Write a one-sentence hypothesis about how the gait will
change. Train it with the **same** environment count, iterations and PPO
settings as your reference run, then compare in matched conditions.

Compare behavior, not just total reward. After a reward change the summed
reward means something different. Look at the per-term episode rewards,
tracking error, falls, standing stillness, foot slip, landing impacts and
upper-body posture, and watch both policies side by side in the viewer.

The runner logs per-iteration scalars but not HW2's `Train/total_env_steps`.
For an unresumed run, iteration `k` corresponds to `(k + 1) * num_envs * 24`
transitions; add segments together for resumed runs. The HW2 plot script does
not work unchanged.

**Write-up** (two pages, PDF on Canvas, due with Checkpoint 1
**[TA: confirm due date]**):

1. For three of your seven rewards, one hand-built test state, the value you
   expected, and why.
2. A reward or training problem you hit and how you found it.
3. Your variant: the hypothesis, the comparison plot or table, and what
   actually happened.
4. Task IDs, run folders, checkpoint names and seeds for every run you cite.

Optional for the curious: train `Course-Booster-K1-Rewards-Candidate` with the
same settings and report whether dropping the two shaping terms changes the gait.
