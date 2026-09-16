# 02 — Custom algorithms in mjlab

HW0 trained with rsl_rl's PPO without ever naming it. From HW2 on you write
algorithms yourself, and they still run through the same `uv run train`,
`uv run play` and `sbatch scripts/train.sbatch`. This page is how that works and
what you must get right when you add one.

Everything here was checked against the mjlab commit pinned in `pyproject.toml`
and the rsl_rl version in `uv.lock`. File references below are relative to
`.venv/lib/python3.13/site-packages/`. If the pin changes, re-check them.

## 1. How a task id reaches `uv run train`

1. `pyproject.toml` declares an entry point in the group `mjlab.tasks` pointing
   at `course_tasks`. mjlab imports every package in that group at startup.
2. `src/course_tasks/__init__.py` imports each assignment package
   (`course_tasks.hw0`, `course_tasks.hw2`, ...).
3. Importing an assignment package runs its `register_mjlab_task(...)` calls
   (`mjlab/tasks/registry.py:22`):

   ```python
   register_mjlab_task(
     task_id="Course-HW2-DoubleCartpole-Balance-PPO",
     env_cfg=...,        # the environment used for training
     play_env_cfg=...,   # the environment used by `play`
     rl_cfg=...,         # the runner/algorithm config (a dataclass)
     runner_cls=...,     # the class that trains; None means rsl_rl's PPO
   )
   ```

Consequences worth knowing:

- **Registration is an import side effect.** If a task is "missing" from
  `uv run list-envs`, the package was never imported, or its import raised.
- **Task ids are global and unique.** Registering the same id twice raises
  `ValueError` (`registry.py:39`). That is why course ids start with `Course-`.
  To try a variant, register a *new* id.
- **Registered configs are templates.** `load_env_cfg` / `load_rl_cfg` return
  deep copies (`registry.py:53`, `:63`), so CLI overrides never leak back into
  the registry.
- **`runner_cls=None` means the default.** `train` and `play` fall back to
  mjlab's `MjlabOnPolicyRunner`, which is rsl_rl's PPO.

## 2. Two places to plug in an algorithm

| Hook | You replace | You must match | Example |
|---|---|---|---|
| `runner_cls=` in `register_mjlab_task` | the whole training loop: collection, updates, logging, checkpoints | the runner contract in §3 | every HW2 task; mjlab's own velocity and tracking tasks |
| `algorithm.class_name` in an rsl_rl runner config | only the learning rule, inside rsl_rl's `OnPolicyRunner` | rsl_rl's algorithm interface: a static `construct_algorithm(obs, env, cfg, device)` plus `act`, `process_env_step`, `compute_returns(obs)`, `update`, `train_mode`, `eval_mode`, `save`, `load`, `get_policy` — built on rsl_rl's `MLPModel` and TensorDict observations (see `rsl_rl/algorithms/ppo.py`) | none in this course yet |

HW2 uses the first hook so its networks and rollout buffer can be plain, readable
PyTorch. The second is the lighter option when rsl_rl's models suit you.

## 3. The runner contract: exactly what the scripts call

A runner does not subclass anything. It only has to respond to these calls.

**`uv run train`** (`mjlab/scripts/train.py`):

| Line | Call | Notes |
|---|---|---|
| 167 | `runner_cls(env, agent_cfg, str(log_dir), device)` | `agent_cfg` is `asdict(...)` of your config — a **plain dict** |
| 170 | `runner.add_git_repo_to_log(__file__)` | record repo state with the run |
| 173 | `runner.load(str(resume_path))` | **only** with `--agent.resume` |
| 175 | `runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)` | the flag is **always True**; honour or ignore it deliberately |

Before any of that, lines 164–165 write `params/env.yaml` and
`params/agent.yaml` into the run directory.

**`uv run play`** (`mjlab/scripts/play.py`):

| Line | Call | Notes |
|---|---|---|
| 204 | `runner_cls(env, asdict(agent_cfg), device=device)` | **no `log_dir`**, so no logger writer exists — don't assume one |
| 205 | `runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)` | |
| 208 | `runner.get_inference_policy(device=device)` | must return a callable taking the observation **TensorDict** and returning actions |
| 216–222 | `load(...)` + `get_inference_policy(...)` again | **on every checkpoint swap** in the viser viewer |

**The `env` you are given** is mjlab's `RslRlVecEnvWrapper`
(`mjlab/rl/vecenv_wrapper.py`):

- `get_observations()` returns a TensorDict keyed by observation group
  (`"actor"`, `"critic"`), each `[num_envs, dim]`.
- `step(actions)` returns `(obs, rewards, dones, extras)`. `rewards` and `dones`
  are `[num_envs]`; `dones` is a long tensor combining terminations and timeouts.
- `extras["time_outs"]` exists only for **infinite-horizon** tasks. HW2's tasks
  are finite-horizon, so it is absent.
- **Environments reset themselves.** On a done step, the returned observation is
  already the first observation of the next episode.
- Rewards are multiplied by the control timestep by default
  (`scale_rewards_by_dt`).

## 4. Config logistics

- **Your runner config must subclass `RslRlBaseRunnerCfg`** (`mjlab/rl/config.py`).
  `train.py` reads `seed`, `max_iterations`, `resume`, `load_run`,
  `load_checkpoint`, `experiment_name`, `run_name`, `wandb_tags` and
  `clip_actions` from it.
- **Every field is a CLI flag.** tyro builds flags from the registered default's
  concrete type, so adding a field is how you expose a hyperparameter. Nested
  dataclasses become nested flags: a field `clip_param` inside `algorithm` is
  `--agent.algorithm.clip-param`. Check with `uv run train <task> --help`.
- **Your runner receives a dict, not your dataclass.** Read `cfg["algorithm"]["gamma"]`,
  not `cfg.algorithm.gamma`.
- **Where output goes.** `experiment_name` picks `logs/rsl_rl/<experiment>/`.
  `run_name` is appended to the timestamped run directory and becomes the W&B
  display name.
- **If you use rsl_rl's `Logger`** (HW2's runner does), your config dict must
  contain `algorithm` as a dict (`rsl_rl/utils/logger.py:59`) and
  `num_steps_per_env` (`:176`). For W&B, set `logger` to the dict
  `{"class_name": "WandbLogWriter", "project_name": ...}`; the plain string
  `"wandb"` is deprecated in rsl_rl.
- **Logged scalars use the iteration as their step.** If runs collect different
  amounts of data per iteration, also log a step counter (HW2 logs
  `Train/total_env_steps`) and use it as the W&B x-axis.

## 5. Pitfalls checklist

- [ ] **`inference_mode` tensors can't be backpropagated through.** Collect under
      `torch.inference_mode()` for speed, but compute anything that feeds a loss
      outside it, and re-evaluate log-probabilities during the update.
- [ ] **The checkpoint format is a contract.** `play` loads with `strict=True`,
      again on every checkpoint swap. Rename or remove a key your `state_dict`
      saves and every existing checkpoint stops loading. Add keys; don't rename.
- [ ] **`get_inference_policy` should act with the mean**, not a sample.
- [ ] **`play` gives you no writer.** Anything in `__init__`, `load` or
      `get_inference_policy` that touches the logger must handle `None`.
- [ ] **New variant, new id.** Don't edit a registered task to try something;
      register another one.
- [ ] **A registration error breaks every task.** Registration runs at import, so
      one bad call takes down `uv run list-envs` for the whole course. Run it
      straight after editing.
- [ ] **Smoke test on CPU before any `sbatch`.** Free, on the login node:

  ```bash
  WANDB_MODE=offline uv run train <task-id> \
    --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3
  ```
