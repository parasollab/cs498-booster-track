"""Imported explicitly by the lab launcher, in the isolated Booster environment."""
import mjlab.tasks  # Populate upstream task registrations first.
from mjlab.tasks.registry import register_mjlab_task
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import booster_k1_ppo_runner_cfg
from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from .env_cfg import course_env_cfg

for task_id, candidate in (
    ("Course-Booster-K1-Rewards", False),
    ("Course-Booster-K1-Rewards-Candidate", True),
):
    agent = booster_k1_ppo_runner_cfg()
    agent.experiment_name = "course_k1_rewards" + ("_candidate" if candidate else "")
    register_mjlab_task(
        task_id=task_id,
        env_cfg=course_env_cfg(candidate=candidate),
        play_env_cfg=course_env_cfg(play=True, candidate=candidate),
        rl_cfg=agent,
        runner_cls=BoosterOnPolicyRunner,
    )
