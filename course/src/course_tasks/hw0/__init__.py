"""HW0 task registration.

Task ids are namespaced `Course-...` so they can't collide with mjlab's
built-in `Mjlab-...` tasks (mjlab raises if an id is registered twice).
"""

from mjlab.tasks.registry import register_mjlab_task

from course_tasks.hw0.cartpole_double_env_cfg import (
  cartpole_double_balance_env_cfg,
  cartpole_double_balance_ppo_runner_cfg,
  cartpole_double_swingup_env_cfg,
  cartpole_double_swingup_ppo_runner_cfg,
)
from course_tasks.hw0.cartpole_env_cfg import (
  cartpole_balance_env_cfg,
  cartpole_ppo_runner_cfg,
  cartpole_swingup_env_cfg,
)

register_mjlab_task(
  task_id="Course-Cartpole-Balance",
  env_cfg=cartpole_balance_env_cfg(),
  play_env_cfg=cartpole_balance_env_cfg(play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Course-Cartpole-Swingup",
  env_cfg=cartpole_swingup_env_cfg(),
  play_env_cfg=cartpole_swingup_env_cfg(play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)

# Registration happens at import time, so an unfinished starter must not
# crash `uv run list-envs` for everything else. Once you remove the last
# NotImplementedError in cartpole_double_env_cfg.py, both tasks appear
# automatically.
try:
  register_mjlab_task(
    task_id="Course-Cartpole-Double-Balance",
    env_cfg=cartpole_double_balance_env_cfg(),
    play_env_cfg=cartpole_double_balance_env_cfg(play=True),
    rl_cfg=cartpole_double_balance_ppo_runner_cfg(),
  )
  register_mjlab_task(
    task_id="Course-Cartpole-Double-Swingup",
    env_cfg=cartpole_double_swingup_env_cfg(),
    play_env_cfg=cartpole_double_swingup_env_cfg(play=True),
    rl_cfg=cartpole_double_swingup_ppo_runner_cfg(),
  )
except NotImplementedError:
  import warnings

  warnings.warn(
    "Course-Cartpole-Double-* not registered yet: finish the TODOs in "
    "course_tasks/hw0/cartpole_double_env_cfg.py",
    stacklevel=1,
  )
