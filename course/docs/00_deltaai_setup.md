# Getting onto DeltaAI and installing the course environment

This guide assumes you have made your ACCESS account and given 
Part A gets you from your laptop to a shell on a DeltaAI login node.
Part B installs the course environment there. Do them in order.

---

# Part A — Get access to DeltaAI


### A.1 First SSH connection

From your laptop's terminal (macOS/Linux: built in; Windows: use Windows
Terminal + OpenSSH, or WSL):

```bash
ssh <your_ncsa_username>@dtai-login.delta.ncsa.illinois.edu
```

Enter your NCSA password, approve the Duo push, and you should land in a
shell whose prompt looks like `<you>@gh-login0X`. Congratulations — every
future instruction that says "on the login node" means *here*.

While you're at it, add this to `~/.ssh/config` **on your laptop** — you'll
need the port forwarding constantly for the 3D viewer (see
`01_workflow.md`):

```
Host deltaai
    HostName dtai-login.delta.ncsa.illinois.edu
    User <your_ncsa_username>
    LocalForward 8080 localhost:8080
```

After that, `ssh deltaai` both logs you in and forwards the viewer port.

### A.2 Know where your files live

| Location            | Use for                          | Do NOT use for |
|---------------------|----------------------------------|----------------|
| `$HOME` (`/u/...`)  | this repo, dotfiles, uv caches   | large or long-lived datasets — 100 GB quota, snapshots purged after ~30 days |
| `/work/hdd/<acct>/` | large datasets, shared data      | — |
| `/work/nvme/...`    | fast scratch                     | anything you can't afford to lose |

**HW0 is a deliberate exception to the usual "outputs go on `/work`" advice.**
Cartpole checkpoints are small, so `scripts/train.sbatch` writes everything
into the repo under `hw0/logs/`. Later assignments with bigger artifacts will
say so explicitly.

Keep an eye on your quota anyway (`quota` or `du -sh hw0/logs`), and delete
runs you no longer need — checkpoints accumulate faster than you'd think.

---

# Part B — Install the course environment (on the login node)

Everything below happens **on the login node**, in the SSH session from
A.3. Nothing is installed on your laptop.

### B.1 Clone the repo

```bash
git clone <COURSE_REPO_URL> ~/(your_repo_name)
cd ~/(your_repo_name)
```



### B.2 Install uv

`uv` is the Python package/environment manager the course standardizes on:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell, and confirm `uv --version` prints something.

### B.3 Sync the environment

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and builds the exact environment
the course and autograder use — including the correct **aarch64 + CUDA**
PyTorch build (verified on DeltaAI: torch 2.13.0+cu130). The first sync
downloads a few GB; subsequent ones are fast.

Rules that keep everyone's environment identical:

- Never `pip install` into this project, never conda, never edit
  `pyproject.toml` or `uv.lock` unless an assignment explicitly says to.
- If `uv sync` proposes changing `uv.lock`, you edited something you shouldn't.
- The environment lives in `.venv/` inside the repo on your (shared)
  `$HOME`, so the same `uv run ...` commands work on login *and* compute
  nodes. You sync once, here, and never on a GPU node.

### B.4 Verify

```bash
uv run python scripts/check_login_env.py
```

Expected: every line `[PASS]`, including
`CUDA unavailable (expected on login node)`. Common first-run issues:

- `uv: command not found` → B.2's PATH step; restart your shell.
- mjlab import errors → you're not in `~/course-tasks`, or sync didn't
  finish; rerun `uv sync` and read its last lines.
- `course tasks registered` fails but mjlab imports → post on the forum
  with the full output.

### B.5 Where to go next

Read `docs/01_workflow.md` (how to actually *use* the cluster and see the
3D viewer in your browser), then start `hw0/README.md`.
