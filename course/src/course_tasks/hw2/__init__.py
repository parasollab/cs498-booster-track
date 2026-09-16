"""HW2 task registration.

Every task here is registered with `runner_cls=ReinforceRunner`. That one
argument is what makes `uv run train` drive HW2's own algorithms instead of
rsl_rl's PPO — see docs/02_custom_algorithms.md for the full story.

  Course-HW2-*-Reinforce   the provided REINFORCE (reinforce.py). Your baseline.
  Course-HW2-*-PPO         YOUR algorithm (ppo.py). Until you change ppo.py it
                           is a copy of REINFORCE and trains exactly like it.

Which algorithm a task runs is decided by `algorithm.class_name` in its runner
config ("Reinforce" or "PPO"); runner.py picks the class from that.

Task ids are namespaced `Course-HW2-...` so they can't collide with mjlab's
`Mjlab-...` tasks or HW0's `Course-Cartpole-...` (mjlab raises on a duplicate id).

HW0 wraps its registrations in `try/except NotImplementedError` because its
starter code raises until finished. Nothing in HW2 raises, so there is no guard.
"""

from mjlab.tasks.registry import register_mjlab_task

from course_tasks.hw2.env_cfg import (
  STEPS_PER_EPISODE,
  cartpole_balance_env_cfg,
  cartpole_double_balance_env_cfg,
)
from course_tasks.hw2.runner import PpoRunnerCfg, ReinforceRunner, ReinforceRunnerCfg

# Placeholder budgets until the staff GPU runs measure real convergence. PPO at
# T=100 gets the same iterations as REINFORCE, so both see the same total
# environment steps.
_CARTPOLE_ITERATIONS = 300
_DOUBLE_ITERATIONS = 1000


def reinforce_cartpole_cfg() -> ReinforceRunnerCfg:
  return ReinforceRunnerCfg(
    experiment_name="hw2_reinforce_cartpole",
    num_steps_per_env=STEPS_PER_EPISODE,
    max_iterations=_CARTPOLE_ITERATIONS,
    save_interval=50,
  )


def reinforce_double_cfg() -> ReinforceRunnerCfg:
  cfg = reinforce_cartpole_cfg()
  cfg.experiment_name = "hw2_reinforce"
  cfg.max_iterations = _DOUBLE_ITERATIONS
  return cfg


def ppo_cartpole_cfg() -> PpoRunnerCfg:
  return PpoRunnerCfg(
    experiment_name="hw2_ppo_cartpole",
    num_steps_per_env=STEPS_PER_EPISODE,
    max_iterations=_CARTPOLE_ITERATIONS,
    save_interval=50,
  )


def ppo_double_cfg() -> PpoRunnerCfg:
  """Start here for your own registration below."""
  cfg = ppo_cartpole_cfg()
  cfg.experiment_name = "hw2_ppo"
  cfg.max_iterations = _DOUBLE_ITERATIONS
  return cfg


register_mjlab_task(
  task_id="Course-HW2-Cartpole-Balance-Reinforce",
  env_cfg=cartpole_balance_env_cfg(),
  play_env_cfg=cartpole_balance_env_cfg(play=True),
  rl_cfg=reinforce_cartpole_cfg(),
  runner_cls=ReinforceRunner,
)
register_mjlab_task(
  task_id="Course-HW2-DoubleCartpole-Balance-Reinforce",
  env_cfg=cartpole_double_balance_env_cfg(),
  play_env_cfg=cartpole_double_balance_env_cfg(play=True),
  rl_cfg=reinforce_double_cfg(),
  runner_cls=ReinforceRunner,
)
register_mjlab_task(
  task_id="Course-HW2-Cartpole-Balance-PPO",
  env_cfg=cartpole_balance_env_cfg(),
  play_env_cfg=cartpole_balance_env_cfg(play=True),
  rl_cfg=ppo_cartpole_cfg(),
  runner_cls=ReinforceRunner,
)
register_mjlab_task(
  task_id="Course-HW2-DoubleCartpole-Balance-PPO",
  env_cfg=cartpole_double_balance_env_cfg(),
  play_env_cfg=cartpole_double_balance_env_cfg(play=True),
  rl_cfg=ppo_double_cfg(),
  runner_cls=ReinforceRunner,
)


# --- YOUR REGISTRATION ---
#
# Part II, Step 0 of hw2/README.md. Register
# `Course-HW2-DoubleCartpole-Balance-PPO-T24` here: the double cartpole, your
# PPO, rollouts of 24 steps instead of 100, and enough iterations that it sees
# the same total environment steps as the T=100 run.
#
# Everything above this marker is provided; this block is yours. It runs at
# import time like the rest of the file, so a mistake here (a typo, a duplicate
# id) will break `uv run list-envs` for every task — run it right after you
# write it.
