# HW2 — Policy gradients: from REINFORCE to PPO

**Learning Objectives**

1. Read a foundational RL paper and map every symbol in it to working code
2. Why a Monte-Carlo policy gradient is unbiased, and what makes it noisy
3. How a custom algorithm plugs into mjlab's `train` / `play` workflow
4. Extend an algorithm from its paper alone: turn REINFORCE into PPO
5. Compare algorithms fairly, and read a run's health off W&B

**Task** 

HW2 gives you a complete, working REINFORCE. In Part I
you take it apart with Sutton & Barto's REINFORCE chapter, the original Williams
(1992) paper, and a walkthrough notebook. In
Part II you turn a copy of it into PPO using the Schulman et al. (2017) paper,
register your own task variant, and compare REINFORCE against your PPO on the
double cartpole.

**Grading.** Your `ppo.py`, your three training runs (Step 8), and your
writeup. The Part I notebook is **not graded** but it is highly reccommended for understanding.

**Time estimate.** Part I (Steps 1–4) ~3–4 h, mostly reading. Part II (Steps
5–7) ~5–6 h. **Training might queue more than usual due to confrence deadlines, start work early .**

---

## Assignment Steps

| Step | What | Where | Graded? |
|------|------|-------|---------|
| **Part I** | **REINFORCE** | | |
| 1 | Pull HW2, verify, set up notebooks | login node + VS Code | no, but required |
| 2 | Read the REINFORCE sources | — | questions for the writeup |
| 3 | Work through the REINFORCE notebook | VS Code on the login node | no |
| 4 | Train the REINFORCE baseline | sbatch | the run is used in Step 8 |
| **Part II** | **PPO** | | |
| 5 | How algorithms plug into mjlab; register your own task | login node | **yes** (via Step 8) |
| 6 | Read Schulman et al. (2017) | — | questions for the writeup |
| 7 | Turn `ppo.py` into PPO | login node | **yes** |
| 8 | Train and compare | sbatch + W&B | **yes** |
| 9 | Write up and submit to Canvas | — | **yes** |

Ten questions (★ Q1–Q10) appear in Steps 2–8. Answer all of them in your writeup.

---

## Step 0 — Don't waste GPU hours (read this first)

```bash
./scripts/my_jobs.sh       # what's running, what finished, what failed
./scripts/my_usage.sh      # my GPU-hours + class total
./scripts/kill_my_jobs.sh  # cancel my jobs (asks first)
```

Everything in Steps 1–3, 5 and 7 is **free**: it runs on the login node's CPU.
Only Steps 4 and 8 use GPU-hours. Every training command in this handout has a
free CPU smoke test next to it — run that first, every time.

> **Changed in HW2.** Training output now lands in `logs/` at the repo root, not
> `hw0/logs/`. One log root for every assignment; runs stay separated by
> experiment name. Your old HW0 runs are untouched where they are.

---

# Part I — REINFORCE

## Step 1 — Pull HW2, verify, set up notebooks

From the repo root, on a login node:

```bash
git pull
uv sync                                   # HW2 adds the notebook kernel (ipykernel)
uv run wandb login                        # paste the key from https://wandb.ai/authorize
uv run python scripts/check_login_env.py
uv run list-envs | grep HW2
```

**Do the `wandb login` now, before you train anything.** Every training job logs
to Weights & Biases, and a job that starts without a stored key dies immediately
with `No API key configured` — after it has waited in the queue. W&B is there for
*you*: it is how you watch a run that is still going and compare runs afterwards.
You will not submit W&B links; grading is from the log files on disk (Step 9).

You should see four tasks:

| Task | Algorithm | Code |
|---|---|---|
| `Course-HW2-Cartpole-Balance-Reinforce` | REINFORCE | `src/course_tasks/hw2/reinforce.py` (provided) |
| `Course-HW2-DoubleCartpole-Balance-Reinforce` | REINFORCE | same |
| `Course-HW2-Cartpole-Balance-PPO` | yours | `src/course_tasks/hw2/ppo.py` |
| `Course-HW2-DoubleCartpole-Balance-PPO` | yours | same |

`ppo.py` starts as an exact copy of `reinforce.py`, so right now the PPO tasks
train exactly like REINFORCE. That changes in Part II.

The files you'll work with, and which ones you may edit:

| File | You |
|---|---|
| `hw2/reinforce_walkthrough.ipynb` | edit freely (Part I) |
| `src/course_tasks/hw2/ppo.py` | **modify into PPO** (Part II) |
| `src/course_tasks/hw2/__init__.py` | edit **only** the `# --- YOUR REGISTRATION ---` block (Part II) |
| `reinforce.py`, `runner.py`, `modules.py`, `storage.py`, `env_cfg.py` | read, don't edit |
| `hw2/plot_comparison.py` | run it (Step 8), don't edit |

Then set up VS Code for notebooks: follow **"Notebooks: VS Code on the login
node"** in `docs/01_workflow.md`.

**Done when:** all four task ids are listed, `wandb login` reports success, and
`hw2/reinforce_walkthrough.ipynb` opens in VS Code with the repo's `.venv`
kernel selected.

## Step 2 — Read the REINFORCE sources

REINFORCE is the textbook policy gradient optimization architecture for RL. We will use this as a foundation in which we build the more complicated archticture, Proximal Policy Optimization(PPO), on top of

Read the textbook first: it is the clearest
statement of what `reinforce.py` does. Then read the original paper, reading papaers will be an essential skill for your final project and work in ML and Robotics in general. This is a skill that must be trained and cannot be subsituted without doing.

### 2a. Sutton & Barto, Chapter 13 — the clear version

R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed.,
MIT Press (2018), Chapter 13, "Policy Gradient Methods". Free from the authors:
[RLbook2020.pdf](http://incompleteideas.net/book/RLbook2020.pdf).

**Read §13.1 through §13.7.** §13.3 and §13.4 are REINFORCE itself, eq. 13.8,
the boxed algorithms, and the baseline, and §13.7 is the Gaussian policy that
`GaussianActor` implements. Two things to notice on the way past: S&B call
∇ ln π the *eligibility vector*, which is Williams' "characteristic eligibility"
under a newer name, and second, the boxed algorithms carry a γᵗ factor that the chapter's
own text leaves out, as does HW2. The note right after the §13.3 box in the book explains why.

### 2b. Williams (1992) — the original

R. J. Williams, *Simple Statistical Gradient-Following Algorithms for
Connectionist Reinforcement Learning*, Machine Learning 8, 229–256 (1992).
[doi:10.1007/BF00992696](https://doi.org/10.1007/BF00992696) — free through the
UIUC library.

**Read pages 1–16** (229–244 in the journal's numbering). This paper is the first introduction of REINFORCE, much before modern deep RL terms were collocialized. Hence you will see terms that might not seem familar but rest assure the analogs exist in the table below. 

Important sections are: §4, the REINFORCE update and where the name comes from; the
remark about *causal* reinforcement at the end of §5, which is why we weight step
t by the return from t onward; and §6, the Gaussian unit, including footnote 2 on
adapting ln σ rather than σ. You can skip every proof.

| Williams (1992) | Sutton & Barto | HW2 code |
|---|---|---|
| weights w_ij | policy parameter θ | `actor` parameters |
| unit output y | action A_t | `storage.actions` |
| output distribution g | policy π(a \| s, θ) | `GaussianActor.distribution` |
| reinforcement r | return G_t | `storage.returns` |
| reinforcement baseline b_ij | baseline b(S_t), e.g. v̂(S_t, w) | `storage.values` |
| characteristic eligibility ∂ ln g / ∂ w_ij | eligibility vector ∇ ln π(A_t \| S_t, θ) | autograd through `evaluate_actions` |
| learning rate factor α_ij | step size α | `learning_rate` |

### What HW2 does that neither source's pseudocode does

- **Batched updates.** S&B's boxed algorithms update θ at every step of a single
  episode. HW2 collects 100 steps from each of 1024 environments and takes *one*
  gradient step on the average.
- **No γᵗ factor.** S&B's boxed update multiplies by γᵗ; the note after the §13.3
  box explains why. HW2, like nearly all deep-RL code, leaves it out.
- **One loss, one optimizer, for actor and critic.** S&B §13.4 updates the policy
  and the value function with separate step sizes α_θ and α_w. HW2 adds the
  critic's squared error to the policy loss (`value_loss_coef`) and takes one Adam
  step.
- **Advantage normalization, gradient-norm clipping and Adam** come from later
  practice and are in neither source. The entropy bonus has a precursor in
  Williams' §8.1 remark.

★ **Q1:** Williams' Theorem 1 says (r − b_ij) ∂ ln g_i / ∂ w_ij is an *unbiased*
estimate of ∂E{r | W}/∂w_ij. In your own words, what does unbiased mean here, and
why must the baseline b_ij not depend on the output y_i? (S&B's baseline b(s) in
eq. 13.10 likewise may depend on the state, but not on the action.)

★ **Q2:** Williams' episodic update (eq. 11) multiplies *every* step's
eligibility by the *whole* episode's reinforcement. S&B's REINFORCE (eq. 13.8) and
HW2's `compute_returns` weight step t by the return *from t onward* instead. Why
is that still an unbiased estimate? (Williams' "causal" remark at the end of §5
is the idea.) Answer after the notebook, and use the variance you measured there.

★ **Q3:** `GaussianActor` learns `log_std`, not σ. Connect that choice to
Williams' footnote 2 of §6 and to S&B's eq. 13.20. What is ∂ ln π / ∂ (ln σ)?

## Step 3 — Work through the REINFORCE notebook

Open `hw2/reinforce_walkthrough.ipynb` in VS Code and run it top to bottom. It
builds the real HW2 environment on the CPU and walks through `reinforce.py`
with a few small TODOs you fill in. 

The notebook is **not graded** and is **not submitted**. It's where you build
the understanding Part II needs.

★ **Q4:** Section 7 of the notebook shows two different errors from trying to
backpropagate through log-probabilities produced during collection. Explain
both. Then: `update()` never uses the stored `old_log_probs` — so why does the
storage keep them?

**Done when:** every check cell in the notebook passes.

## Step 4 — Train the REINFORCE baseline

This is the first of the three runs you compare in Step 8. Free smoke test
first, then submit:

```bash
WANDB_MODE=offline uv run train Course-HW2-DoubleCartpole-Balance-Reinforce \
  --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3

sbatch scripts/train.sbatch Course-HW2-DoubleCartpole-Balance-Reinforce \
  --env.scene.num-envs 1024 --agent.run-name reinforce
```

Watch it with `squeue --me` and `tail -f logs/slurm-<jobid>.out`. Watch
`Train/mean_reward` **and** `Policy/mean_std`, the policy's exploration noise.
If the std collapses toward zero early, the policy has stopped exploring.

Rewards are scaled by the 0.05 s control step, so a perfect 100-step episode
scores about **5.0**.

### Where your numbers show up

**Setup.** You logged in during Step 1. Every HW2 run logs to a project called
**`cs498-hw2`** *in your own W&B account* — the name is set for you in the runner
config, and nothing of yours is visible to anyone else until you share it.
`--agent.run-name <name>` names the run's directory on disk, and the run's name in
W&B is that directory name, `<timestamp>_<run-name>`. That is how you tell three
otherwise identical runs apart, which is why the commands here pass one.

The same numbers reach you in three places:

| Where | What you get |
|---|---|
| **W&B**, project `cs498-hw2` → your run | Charts grouped by name prefix: `Train/mean_reward`, `Train/mean_episode_length`, `Train/total_env_steps`; `Loss/*` (one per key your algorithm reports, plus `Loss/learning_rate`); `Policy/mean_std`; `Perf/*` for throughput; and the environment's own per-episode terms, such as its reward and termination counts |
| **The Slurm log**, `logs/slurm-<jobid>.out` | The same values printed as a table every iteration: `Mean reward`, `Mean episode length`, `Mean action std`, `Total steps`, and one `Mean <key> loss` line per reported key. No browser needed |
| **On disk**, `logs/rsl_rl/<experiment>/<timestamp>_<run-name>/` | Checkpoints (`model_*.pt`), the resolved config in `params/`, and TensorBoard event files holding every scalar above — which is what `hw2/plot_comparison.py` reads |

One naming quirk to keep in mind: a key your algorithm reports as `foo` appears
in W&B as `Loss/foo` and in the Slurm log as `Mean foo loss`, whether or not it
is a loss.

**Why the smoke tests say `WANDB_MODE=offline`.** Those three-iteration CPU runs
exist only to prove the code path works; their numbers are meaningless. Offline
mode keeps that noise out of your `cs498-hw2` project and needs no network
or credentials, while running exactly the same logging code — the run still
writes its event files under `logs/`, so `plot_comparison.py` can read it. The
real runs, in this step and Step 8, are online, and those are the ones you
submit. To push an offline run up afterwards: `wandb sync wandb/offline-run-*`.

★ **Q5:** How many environment steps per second did this run get (from the
Slurm log)? Describe the reward curve's shape in two sentences.

---

# Part II — PPO

## Step 5 — How algorithms plug into mjlab

Read **`docs/02_custom_algorithms.md`**. It explains how a task id reaches
`uv run train`, the two places an algorithm can plug in, exactly what `train`
and `play` call on HW2's runner, and the pitfalls. Then read
`src/course_tasks/hw2/__init__.py` and `runner.py` with that page open.

**Exercise — register your own task.** In the `# --- YOUR REGISTRATION ---`
block at the bottom of `src/course_tasks/hw2/__init__.py`, register
`Course-HW2-DoubleCartpole-Balance-PPO-T24`:

- the double cartpole environment, with its play variant,
- your PPO (start from `ppo_double_cfg()`),
- `num_steps_per_env = 24` instead of 100,
- `max_iterations` scaled so it collects the **same total environment steps**
  as the 100-step PPO run,
- `experiment_name="hw2_ppo"` and `run_name="T24"`,
- `runner_cls=ReinforceRunner`, like every other HW2 task.

*Test your registration:*

```bash
uv run list-envs | grep T24
uv run train Course-HW2-DoubleCartpole-Balance-PPO-T24 --help | grep -A1 num-steps-per-env
WANDB_MODE=offline uv run train Course-HW2-DoubleCartpole-Balance-PPO-T24 \
  --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3
```

The id is listed, `--help` shows a default of 24, and the smoke test runs.
Registration runs at import time: a mistake in your block breaks
`uv run list-envs` for every task, so check right after you write it.

★ **Q6:** Trace `uv run train Course-HW2-DoubleCartpole-Balance-PPO` from the
shell to your `PPO.update`. Name each file and function the call passes
through.

## Step 6 — Read Schulman et al. (2017)

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, *Proximal Policy
Optimization Algorithms*, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
(2017).



| Section | How | What it means for you |
|---|---|---|
| **§2.1 Policy Gradient Methods**, eq. 1–2 | **read closely** | L^PG is the loss in your `reinforce.py`. Read the paragraph after eq. 2: why not just take several steps on L^PG? |
| §2.2 Trust Region Methods, eq. 3–5 | read | the idea PPO approximates. Not implemented |
| **§3 Clipped Surrogate Objective**, eq. 6–7, Figs. 1–2 | **implement** | the probability ratio and the clipped objective |
| §4 Adaptive KL Penalty Coefficient, eq. 8 | read | not implemented. **Name clash:** HW0's rsl_rl config has `schedule="adaptive"` and `desired_kl`. That adapts the *learning rate* from the KL. It is not §4's penalty |
| **§5 Algorithm**, eq. 9–12, Algorithm 1 | **implement** | the combined loss, the truncated advantage estimator, and the epochs-over-minibatches loop |
| §6.1 Comparison of Surrogate Objectives, Table 1 | read | why clipping with ε = 0.2 |
| Appendix A, Table 3 | read | the MuJoCo hyperparameters (Q8) |

**Heads-up on eq. 10–11 in the paper, because you are about to implement them.** Both
estimate the advantage over a rollout of T steps, and eq. 11 , with δ_t defined
in eq. 12 , is the general form you will write; eq. 10 is what it reduces to at
λ = 1. Both bootstrap from V(s_T) where collection stopped. That term does
nothing when the rollout ends on a terminal step (your T = 100 runs) and matters
when it stops mid-episode (the T = 24 task you registered in Step 5). They are
also the one place in the paper you would reasonably copy a formula straight
into code.

As printed, the exponent on each sum's last term is `T − t + 1`. That term is
r_{T−1} (or δ_{T−1}), so the exponent should be `T − t − 1`: transcribed
literally, your last term picks up two extra factors of γ (of γλ in eq. 11).
Nothing crashes: the run trains and the reward curve moves, and the advantages
are just quietly weighted wrong.

To solve this, first, write the **recursion** rather than the closed-form sum,
this is implemented in practice (`reinforce.py`'s `compute_returns` is the same
shape). And do Step 7's piece-by-piece checks, which compare your estimator
against code you already trust and would catch the stray factor immediately.

★ **Q7:** What does §2.1 say goes wrong if you take many optimization steps on
L^PG with the same batch? Which case in Figure 1 does the `min` in eq. 7 guard
against, and why is it a *pessimistic* bound?

## Step 7 — Turn `ppo.py` into PPO

Edit `src/course_tasks/hw2/ppo.py` until it implements PPO.
How you get there is yours to decide. What follows is a suggestion.

**Where the changes live.** Three functions in `ppo.py`, and nothing else:

- `compute_returns(last_obs)`
- `compute_advantages()`
- `update()`

`act` and `process_env_step` stay exactly as they are, as do
checkpointing and logging. The constructor already receives every hyperparameter
you need.

**Work in pieces you can check.**

 A policy-gradient bug is hard to catch so don't write all of PPO and then stare
at a reward curve.

**Test at three levels.**

1. **On numbers you can check by hand** — build a `RolloutStorage` of a handful of
   steps and one or two environments, fill `rewards`, `dones` and `values`
   yourself, and check the properties below.

   - **λ = 1 with V(s) = 0 everywhere** must reproduce `reinforce.py`'s
     discounted return-to-go on the same rewards and dones. This is the setting
     where your new estimator and the old one have to agree.
   - **A done in the middle of your buffer** must stop credit leaking backwards
     across it, exactly as the `(1 - done)` factor already does in
     `compute_returns`.
   - **A rollout that stops mid-episode** must pick up the value of `last_obs`;
     one whose last step is a done must not. Give the critic a constant output
     and the difference is easy to read off.
   - **The ratio on the first minibatch of the first epoch, before any optimizer
     step, must be 1** to within about 1e-6 . If it isn't, `old_log_probs` and your recomputed
     log-probabilities disagree about something. If it is *still* exactly 1 on
     later epochs, your update isn't changing the policy.
   - **With one epoch, one minibatch and clipping wide enough never to bite**,
     your loss must produce the same gradient as REINFORCE's on the same batch.
2. **On the real environment, on CPU** - three iterations with 8 environments
   exercises the whole path (config, runner, your code, logging) for free:

   ```bash
   WANDB_MODE=offline uv run train Course-HW2-Cartpole-Balance-PPO \
     --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3
   WANDB_MODE=offline uv run train Course-HW2-DoubleCartpole-Balance-PPO-T24 \
     --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3
   ```

3. **On the GPU**, in Step 8 — only once 1 and 2 are clean.

**Make your update report on itself.** Every key your `update()` returns is
logged, with no extra work: see "Where your numbers show up" in Step 4 for the
three places it lands and what it gets called in each. Two diagnostics are worth
far more than they cost: how far the probability ratio drifts from 1, and what
fraction of samples hit the clip. Together they tell you whether your update is
doing anything at all, and whether it is doing too much.

**Rules:**

- Keep the contract in `ppo.py`'s docstring: the class name, the method names and
  signatures, and the keys `state_dict` saves (or `uv run play` stops loading your
  checkpoints).
- Batched torch only.
- Don't edit any other provided file.

**Move on when:** The CPU runs above finish; the
diagnostics look sane on the first iterations; and the warm-up task's reward goes
up rather than sideways.

## Step 8 — Train and compare

Three runs on the double cartpole:

```bash
# Run 1 is your Step 4 REINFORCE run.
sbatch scripts/train.sbatch Course-HW2-DoubleCartpole-Balance-PPO \
  --env.scene.num-envs 1024 --agent.run-name ppo_T100
sbatch scripts/train.sbatch Course-HW2-DoubleCartpole-Balance-PPO-T24 \
  --env.scene.num-envs 1024
```

| Run | Rollout | Isolates |
|---|---|---|
| REINFORCE | 100 | the baseline |
| PPO, T = 100 | 100 | clipping + several epochs, with the **same data per iteration** as REINFORCE |
| PPO, T = 24 | 24 | shorter, bootstrapped rollouts |

**Compare against environment steps, not iterations.** A 24-step run collects
about a quarter of the data per iteration that a 100-step run does, so plotted by
iteration the curves are not comparable. HW2's runner logs `Train/total_env_steps`
for exactly this.

A script draws the comparison for you, straight from `logs/`:

```bash
uv run python hw2/plot_comparison.py                        # -> hw2/comparison.png
uv run python hw2/plot_comparison.py --metric Policy/mean_std
```

With no arguments it picks up every HW2 run under `logs/rsl_rl/`, plots each one's
`Train/mean_reward` against `Train/total_env_steps`, and writes a PNG you can drop
into your writeup. Pass run directories explicitly to plot a subset. It reads the
event files your runs already write, so it needs no network and works offline.

You can do the same thing by hand in W&B: add a line plot of `Train/mean_reward`,
open the panel's settings, and set its **X axis** to `Train/total_env_steps`.

View your PPO policy:

```bash
uv run play Course-HW2-DoubleCartpole-Balance-PPO \
  --checkpoint-file "$(ls -t logs/rsl_rl/hw2_ppo/*/model_*.pt | head -1)" \
  --log-root logs/rsl_rl --env.scene.num-envs 4
```

★ **Q9:** At equal environment steps, how do REINFORCE and PPO at T = 100
compare? They saw identical data per iteration, so what explains the difference
you observe?

★ **Q10:** What did shortening the rollout to 24 steps change? Describe what you
*measured*, not what you expected. Then consider: the environment has a fixed
100-step horizon, but the observation contains no information about how many
steps remain. What does bootstrapping from V(s_T) at a mid-episode cut need to
know that the critic cannot see, and did that show up in your results?

## Step 9 — Write up and submit

Your writeup contains:

- Answers to ★ Q1–Q10.
- The Step 8 comparison plot, x-axis `Train/total_env_steps`, all three runs.
- `Policy/mean_std` for your two PPO runs.
- One paragraph on what broke while you built PPO and how you found it.
- The task ids, run directories and checkpoint filenames you want graded.

### What to submit: your run folders

 Everything needed to grade a run is already on disk,
in the run's own directory:

```
logs/rsl_rl/<experiment>/<timestamp>_<run-name>/
├── model_<iter>.pt        checkpoints, about 120 KB each
├── params/env.yaml        the exact environment that ran
├── params/agent.yaml      the exact hyperparameters that ran
├── git/                   a diff of your repo at submission time
└── events.out.tfevents.*  every scalar, which is what plot_comparison.py reads
```

Find your three runs, newest first:

```bash
ls -td logs/rsl_rl/hw2_reinforce/*/ | head -3
ls -td logs/rsl_rl/hw2_ppo/*/ | head -3
```

Then bundle those three, their Slurm logs, and your plot:

```bash
tar czf hw2_runs.tar.gz \
  logs/rsl_rl/hw2_reinforce/<your-reinforce-run> \
  logs/rsl_rl/hw2_ppo/<your-ppo_T100-run> \
  logs/rsl_rl/hw2_ppo/<your-T24-run> \
  logs/slurm-<jobid>.out logs/slurm-<jobid>.out logs/slurm-<jobid>.out \
  hw2/comparison.png
```

Expect roughly 15–20 MB. Name the three runs in your writeup so it is obvious
which folder is which, and check the archive before you upload it:

```bash
tar tzf hw2_runs.tar.gz | head
```

Submit on the canvas:

1. The writeup PDF.
2. Your `src/course_tasks/hw2/ppo.py` and `src/course_tasks/hw2/__init__.py`.
3. `hw2_runs.tar.gz` — your three run folders, their Slurm logs, and the comparison plot.

Before you submit, run what the staff will run:

```bash
uv run python scripts/check_login_env.py &&
uv run list-envs | grep -c HW2 &&
WANDB_MODE=offline uv run train Course-HW2-DoubleCartpole-Balance-PPO-T24 \
  --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3
```

The `grep -c` must print `5`: the four provided tasks plus yours.
