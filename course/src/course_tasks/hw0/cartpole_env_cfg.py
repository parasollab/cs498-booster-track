"""HW0 — Cartpole, annotated.

This file is the worked example the HW0 handout walks through, adapted from
mjlab's own `Mjlab-Cartpole-*` tasks. Read it top to bottom alongside
`hw0/README.md`. Every mjlab environment you build in this course — up to the
humanoid — is made of exactly these pieces:

  MJCF XML  ->  EntityCfg  ->  SceneCfg
                                  + observations   (what the policy sees)
                                  + actions        (what the policy controls)
                                  + events         (what happens on reset)
                                  + rewards        (what we optimize)
                                  + terminations   (when an episode ends)
                              ->  ManagerBasedRlEnvCfg  ->  register_mjlab_task
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---------------------------------------------------------------------------
# 1. The entity: wrap an MJCF model so mjlab can batch it on the GPU.
#
# cartpole.xml defines the physical system: a cart on a rail (slide joint
# "slider", actuated by motor "slide") with one pole ("hinge_1", unactuated).
# SceneEntityCfg("cartpole", joint_names=(...)) is how every other piece of
# the config points at *specific joints* of a *named entity* in the scene.
# ---------------------------------------------------------------------------

_CARTPOLE_XML: Path = Path(__file__).parent / "cartpole.xml"
_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider",))
_HINGE_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_1",))


def _get_spec() -> mujoco.MjSpec:
  # mjlab consumes an MjSpec (MuJoCo's editable model format), not a path,
  # so entities can be procedurally modified before compilation.
  return mujoco.MjSpec.from_file(str(_CARTPOLE_XML))


# Which actuators from the XML does this entity expose? Names must match
# <actuator> targets in the XML ("slider" is the joint the motor drives).
_CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=("slider",)),),
)

# Two variants of the same physical system, differing only in where the pole
# starts: hanging up (balance) vs hanging down (swingup). Same trick you'll
# use for the double pendulum.
_BALANCE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": 0.0},
  joint_vel={".*": 0.0},  # regexes over joint names are allowed
)
_SWINGUP_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": math.pi},
  joint_vel={".*": 0.0},
)


def _get_cartpole_cfg(swing_up: bool = False) -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_CARTPOLE_ARTICULATION,
    init_state=_SWINGUP_INIT if swing_up else _BALANCE_INIT,
  )


# ---------------------------------------------------------------------------
# 2. A custom observation term.
#
# Terms are plain functions env -> tensor of shape [num_envs, dim]. mjlab
# ships many (joint_pos_rel, joint_vel_rel, ...); you write your own when the
# built-ins aren't enough. Feeding (cos θ, sin θ) instead of raw θ removes
# the 2π wrap-around discontinuity — do the same for your pendulum angles.
# ---------------------------------------------------------------------------


def pole_angle_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
  """Cosine and sine of the pole hinge angle. Shape: [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


# ---------------------------------------------------------------------------
# 3. Reward terms.
#
# This is dm_control's classic smooth cartpole reward: a PRODUCT of four
# shaped factors in [0, 1]. Products act like soft ANDs — the reward is only
# high when the pole is up AND the cart is centered AND control is small AND
# the pole is slow. Note everything is written with torch ops over the whole
# batch: one tensor op services all num_envs environments at once. Never
# write a Python loop over environments.
# ---------------------------------------------------------------------------

_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))  # value_at_margin = 0.1
_QUADRATIC_SCALE = math.sqrt(1 - 0.1)


def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """1 at x=0, decaying to 0.1 at |x| = margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _GAUSSIAN_SCALE
  return torch.exp(-0.5 * scaled**2)


def _quadratic_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """1 at x=0, hitting exactly 0 for |x| >= margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _QUADRATIC_SCALE
  return torch.clamp(1 - scaled**2, min=0.0)


def cartpole_smooth_reward(
  env: ManagerBasedRlEnv,
  cart_cfg: SceneEntityCfg = _CART_CFG,
  hinge_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
  """upright * centered * small_control * small_velocity, all in [0, 1]."""
  asset: Entity = env.scene[cart_cfg.name]

  hinge_angle = asset.data.joint_pos[:, hinge_cfg.joint_ids].squeeze(-1)
  upright = (torch.cos(hinge_angle) + 1) / 2

  cart_pos = asset.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
  centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2

  control = env.action_manager.action.squeeze(-1)
  small_control = (4 + _quadratic_tolerance(control, margin=1.0)) / 5

  hinge_vel = asset.data.joint_vel[:, hinge_cfg.joint_ids].squeeze(-1)
  small_velocity = (1 + _gaussian_tolerance(hinge_vel, margin=5.0)) / 2

  return upright * centered * small_control * small_velocity


# ---------------------------------------------------------------------------
# 4. Assemble the environment config.
# ---------------------------------------------------------------------------


def _make_env_cfg(swing_up: bool = False) -> ManagerBasedRlEnvCfg:
  cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
  hinge_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1",))

  # Observations: named terms, concatenated in order. "actor" is what the
  # policy sees (with noise/corruption during training); "critic" can see a
  # privileged, clean copy. Here they're identical.
  actor_terms = {
    "cart_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": cart_cfg}),
    "pole_angle": ObservationTermCfg(
      func=pole_angle_cos_sin, params={"asset_cfg": hinge_cfg}
    ),
    "cart_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": cart_cfg}),
    "pole_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": hinge_cfg}),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  # Actions: one scalar effort on the cart's slide actuator. The policy
  # outputs in roughly [-1, 1]; scale maps that to actuator units.
  actions: dict[str, ActionTermCfg] = {
    "effort": JointEffortActionCfg(
      entity_name="cartpole",
      actuator_names=("slider",),
      scale=1.0,
    ),
  }

  # Events with mode="reset" run at every episode reset. Randomizing the
  # initial state (a small offset around init_state) is what stops the
  # policy from memorizing one trajectory.
  slider_range = (-0.1, 0.1) if not swing_up else (0.0, 0.0)
  events = {
    "reset_slider": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": slider_range,
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("slider",)),
      },
    ),
    "reset_hinge": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.034, 0.034),
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_1",)),
      },
    ),
  }

  rewards = {
    "smooth_reward": RewardTermCfg(
      func=cartpole_smooth_reward,
      weight=1.0,
      params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg},
    ),
  }

  # time_out=True marks a truncation (episode ran out of time) rather than a
  # failure — PPO bootstraps the value function differently for the two.
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"cartpole": _get_cartpole_cfg(swing_up=swing_up)},
      num_envs=1,  # overridden at train time: --env.scene.num-envs 4096
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="cartpole",
      body_name="cart",
      distance=4.0,
      elevation=-15.0,
      azimuth=0.0,
    ),
    sim=SimulationCfg(
      # Contacts disabled: nothing in this scene needs them, and it's faster.
      mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",)),
    ),
    # Policy acts every `decimation` physics steps: control dt = 0.05 s.
    decimation=5,
    episode_length_s=50.0,
  )


def cartpole_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=False)
  if play:
    # In play mode: run forever, no observation noise.
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


def cartpole_swingup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=True)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


# ---------------------------------------------------------------------------
# 5. The RL side: a small PPO config (rsl_rl). You will reuse this shape all
# semester; only sizes and horizons change as tasks get harder.
# ---------------------------------------------------------------------------


def cartpole_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="hw0_cartpole",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )
