# CS 498 - Robotics Team Project

This repository is everything you need for the programming assignments:
environment configs built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp + an Isaac-Lab-style manager API), Slurm job templates for
NCSA's **DeltaAI** cluster, and per-assignment handouts.

**You are probably reading this on GitHub, on your laptop, with no cluster
access yet. That's the expected starting point.** All coursework runs on
DeltaAI. You will not install anything on your own machine except an SSH
client. Follow the numbered path below in order.

---

## Start here

### Step 1 — Get access to DeltaAI (do this today; approvals take time)


Follow **[`docs/00_deltaai_setup.md`](docs/00_deltaai_setup.md), Part A**:
create your NCSA identity, get added to the course allocation, set up Duo
MFA, and make your first SSH connection to a login node. Account creation
and allocation approval can take a day or more — do not leave this until
the assignment is due.

### Step 2 — Clone this repo ON THE CLUSTER and install

Once you can SSH in, follow **`docs/00_deltaai_setup.md`, Part B**. In
short, on a DeltaAI login node (not your laptop):

for more info on [uv](https://docs.astral.sh/uv/)
```bash
git clone <COURSE_REPO_URL> ~/course-tasks
cd ~/course-tasks
curl -LsSf https://astral.sh/uv/install.sh | sh    # then restart your shell
uv sync
```

Do not `pip install` anything, do not create conda envs, do not install
mjlab by hand. `uv sync` reads `pyproject.toml` + `uv.lock` and builds the
exact environment the course expects.

### Step 3 — Verify

Still on the login node:

```bash
uv run python scripts/check_login_env.py
```

Every line must say `[PASS]` (note: CUDA being *unavailable* here is a
pass — login nodes have no GPU; that's normal and explained in the docs).

### Step 4 — Learn the workflow

Read **[`docs/01_workflow.md`](docs/01_workflow.md)**: where code runs
(login node vs. interactive GPU vs. batch jobs), where files go, and how to
see the 3D viewer (viser) in your laptop's browser through an SSH tunnel.
Ten minutes of reading that will save you hours all semester.

### Step 5 — Do HW0

Open **[`hw0/README.md`](hw0/README.md)**. It walks you through everything
step by step and doubles as your training for every later assignment.

---

## How this repo is used all semester

- You **clone it once** and `git pull` at the start of each assignment to
  receive new `hwN/` handouts and `src/course_tasks/hwN/` starter code.
- Do your work on your own branch (`git switch -c dev`) and rebase or merge
  when we push updates. **Never commit to `main`**. Releases only add files, so pulls
  won't conflict with your work.
- All commands run through `uv` from the repo root, on DeltaAI.
  
---

## Managing your jobs and GPU usage
 
Three helper scripts in `scripts/`. Run them from the repo root; they only
affect your own jobs.
 
```bash
./scripts/my_jobs.sh       # are all my training jobs done?
./scripts/my_usage.sh      # how many GPU-hours have I used?
./scripts/kill_my_jobs.sh  # cancel my jobs (asks before acting)
```
 
- **`my_jobs.sh`** — shows what's running, what finished, and what failed
  (with the log file to check). Run it before logging off.
- **`my_usage.sh`** — your GPU-hours and the whole class's. The allocation
  is shared, so check it before submitting extra runs.
- **`kill_my_jobs.sh`** — cancels your jobs after showing you the list and
  asking for confirmation. Use it the moment you spot a bad run — saved
  checkpoints are kept, so you lose nothing.
  
## Repo layout

```
docs/
  00_deltaai_setup.md      Part A: get cluster access · Part B: install this repo
  01_workflow.md           login node / salloc / sbatch, viser tunnel, notebooks
  02_custom_algorithms.md  how an algorithm plugs into train/play (from HW2)
scripts/
  check_login_env.py       sanity checks for the login node (no GPU)
  check_gpu_env.py         sanity checks to run inside salloc / sbatch
  gpu_interactive.sh       wrapper for salloc; prints your tunnel command
  train.sbatch             the ONLY way you should submit training jobs
  cluster.env              course account / partition / storage settings
src/course_tasks/          the Python package with all course environments
  hw0/                     cartpole walkthrough + double pendulum starter
  hw2/                     double pendulum envs, REINFORCE + your PPO
hw0/README.md              assignment 0 handout
hw2/README.md              assignment 2 handout
hw2/reinforce_walkthrough.ipynb   REINFORCE walkthrough notebook (HW2 Part I)
hw2/plot_comparison.py     plots your runs against environment steps
logs/                      all training output lands here (git-ignored)
```
