"""Course snapshot: explicitly register only the unchanged serial K1 Flat task."""
from mjlab.tasks.registry import register_mjlab_task
from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from .velocity.config.k1.env_cfgs import booster_k1_flat_env_cfg
from .velocity.config.k1.rl_cfg import booster_k1_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Booster-K1",
    env_cfg=booster_k1_flat_env_cfg(), play_env_cfg=booster_k1_flat_env_cfg(play=True),
    rl_cfg=booster_k1_ppo_runner_cfg(), runner_cls=BoosterOnPolicyRunner,
)
