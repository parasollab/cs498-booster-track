"""HW0 — YOUR TASK: double cartpole (two-link pendulum), balance and swingup.

Build this the way `cartpole_env_cfg.py` is built. The physical system is `cartpole_double.xml`:
the same cart and rail, but the pole is now two links. Open the XML and
note the joint names, which link each hinge belongs to, and that `hinge_2` is defined *relative to* `pole_1`, not relative to the world.

Two task variants share one config function, exactly like the cartpole:
  * Balance: start with both links up, keep them up.
  * Swingup: start hanging down, swing up and balance.

Provided for you (do NOT modify):
  * `joint_velocity_limit_exceeded` = a safety termination. The double
    pendulum is chaotic, and runaway joint velocities can blow up the
    fixed-timestep integrator; this guard ends those episodes cleanly.
  * Both PPO runner configs at the bottom — everyone trains with the same
    settings, so results are comparable and the autograder can reproduce
    your run. Do not tune them.

Your work is the three TODOs. The registration in hw0/__init__.py and all
public function names/signatures must not change (the autograder imports
them). Grading: see hw0/README.md.
"""

from __future__ import annotations

import math  # noqa: F401  (you will want it)
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,  # noqa: F401
  joint_vel_rel,  # noqa: F401
  reset_joints_by_offset,  # noqa: F401
  time_out,  # noqa: F401
)
from mjlab.envs.mdp.actions import JointEffortActionCfg  # noqa: F401
from mjlab.managers.action_manager import ActionTermCfg  # noqa: F401
from mjlab.managers.event_manager import EventTermCfg  # noqa: F401
from mjlab.managers.observation_manager import (  # noqa: F401
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: F401
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: F401
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg  # noqa: F401
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: F401
from mjlab.terrains import TerrainEntityCfg  # noqa: F401
from mjlab.viewer import ViewerConfig  # noqa: F401

from course_tasks.hw0.cartpole_env_cfg import (
  _gaussian_tolerance,  # noqa: F401
  _quadratic_tolerance,  # noqa: F401
  cartpole_ppo_runner_cfg,
  pole_angle_cos_sin,  # noqa: F401  (works for MULTIPLE joints — check its shape!)
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_CARTPOLE_DOUBLE_XML: Path = Path(__file__).parent / "cartpole_double.xml"
_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider",))
_HINGE1_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_1",))
_HINGE2_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_2",))

# Entity.


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_CARTPOLE_DOUBLE_XML))


_CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=("slider",)),),
)


def _get_cartpole_cfg(swing_up: bool = False) -> EntityCfg:
  # TODO(1): build the EntityCfg with the right initial state for each
  # variant (compare _BALANCE_INIT / _SWINGUP_INIT in cartpole_env_cfg.py).
  # All three joints need entries. Think about hinge_2: it is RELATIVE to
  # pole_1 — what value makes the chain hang straight down when hinge_1 is
  # at pi? Verify your answer in viser before moving on.
  raise NotImplementedError


# Rewards.


def cartpole_double_smooth_reward(
  env: ManagerBasedRlEnv,
  cart_cfg: SceneEntityCfg = _CART_CFG,
  hinge1_cfg: SceneEntityCfg = _HINGE1_CFG,
  hinge2_cfg: SceneEntityCfg = _HINGE2_CFG,
) -> torch.Tensor:
  """Shaped reward in [0, 1]. Shape: [num_envs].

  TODO(2): extend `cartpole_smooth_reward` to the second link. Start from
  the single-pole version (upright * centered * small_control *
  small_velocity) and ask: what does "upright" mean when hinge_2 is a
  RELATIVE angle? (Hint: both local angles at 0 is the unique fully
  extended, vertical configuration.) Keep every factor in [0, 1], keep it
  batched, and sanity-check the value in viser with --agent zero: hanging
  down ≈ 0, balanced upright ≈ 1.
  """
  raise NotImplementedError


# Terminations. (PROVIDED — do not modify.)


def joint_velocity_limit_exceeded(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  limit: float,
) -> torch.Tensor:
  """Terminate if any of the selected joints' velocity exceeds `limit`.

  Safety guard against the double pendulum's chaotic dynamics driving joint
  velocity into the regime where mujoco-warp's fixed-timestep integrator
  diverges. `limit` is set far above anything a normal swing-up/balance
  needs, so this only fires on genuinely runaway states.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return (vel.abs() > limit).any(dim=-1)


# Environment config.


def _make_env_cfg(swing_up: bool = False) -> ManagerBasedRlEnvCfg:
  # TODO(3): assemble, following _make_env_cfg in cartpole_env_cfg.py.
  #  - observations: cart pos/vel, and (cos, sin) + velocity for BOTH
  #    hinges. You do not need a new observation function — reuse
  #    `pole_angle_cos_sin` with a SceneEntityCfg selecting both hinge
  #    joints, and check the resulting shape.
  #  - actions: effort on the slider actuator (unchanged from cartpole).
  #  - events: reset offsets for the slider AND both hinges (the cartpole
  #    only had one hinge to jitter; the slider range still depends on
  #    swing_up the same way).
  #  - rewards: your cartpole_double_smooth_reward.
  #  - terminations: time_out, PLUS the provided
  #    joint_velocity_limit_exceeded over both hinges with limit=50.0.
  #  - scene: entity name must be "cartpole" (the SceneEntityCfgs above use
  #    it); num_envs=1024, env_spacing=4.0, plane terrain.
  #  - viewer/sim/decimation/episode_length_s: same as the cartpole, but
  #    distance=5.0 frames both links better.
  raise NotImplementedError


def cartpole_double_balance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=False)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


def cartpole_double_swingup_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=True)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


# RL config. (PROVIDED, do not modify)


def cartpole_double_balance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = cartpole_ppo_runner_cfg()
  cfg.experiment_name = "cartpole_double"
  cfg.max_iterations = 1500
  return cfg


def cartpole_double_swingup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = cartpole_ppo_runner_cfg()
  cfg.experiment_name = "cartpole_double"
  cfg.max_iterations = 5000
  return cfg
