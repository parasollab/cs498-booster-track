"""Seven batched Torch reward exercises; do not call existing reward helpers.

Every function receives the live environment and returns ONE number per
environment: a tensor of shape [B] where B = env.num_envs, on env.device.
The reward manager multiplies your result by the weight in env_cfg.py and by
the control timestep. Do not apply those yourself.

The first lines of each function already fetch the state you need and note
each tensor's shape and meaning. Your job is the math underneath the TODO.

State you will see:
  env.command_manager.get_command("twist")        [B, 3]  commanded vx, vy (m/s), yaw rate (rad/s), body frame
  env.scene["robot"].data.root_link_lin_vel_b     [B, 3]  measured linear velocity, body frame
  env.scene["robot"].data.root_link_ang_vel_b     [B, 3]  measured roll/pitch/yaw rates, body frame
  env.scene["robot"].data.projected_gravity_b     [B, 3]  unit gravity vector in body frame; (0, 0, -1) when upright
  env.scene["robot"].data.joint_pos / joint_vel   [B, 22] joint angles (rad) and rates
  env.scene["robot"].data.default_joint_pos       [B, 22] the standing pose
  env.scene["robot"].data.site_lin_vel_w          [B, sites, 3] site velocities, world frame
  env.scene["feet_ground_contact"].data.found     [B, 2]  > 0 while that foot touches the ground
  env.scene["feet_ground_contact"].data.current_air_time  [B, 2]  seconds the foot has been airborne
  env.action_manager.action / prev_action / prev_prev_action  [B, 22] raw policy outputs, newest first

Rules: use batched Torch ops over the whole batch. No Python loops over
environments, no .cpu(), no .item(). Reduce over joints/feet/axes, never over B.
"""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

_ROBOT = SceneEntityCfg("robot")


def track_linear_velocity(env, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT,
                          std: float = 0.5, std_at_rest: float = 0.1,
                          relative_std: float = 0.75, lateral_std: float = 0.3,
                          vertical_std: float = 0.5, progress_weight: float = 0.5):
    """Direction-aware body-frame velocity tracking, including zero commands.

    Positive score in [0, 1]; 1 means the robot moves exactly as commanded.
    """
    command = env.command_manager.get_command(command_name)        # [B, 3]: commanded vx, vy, yaw rate
    velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b  # [B, 3]: measured vx, vy, vz (body frame)
    speed = torch.linalg.vector_norm(command[:, :2], dim=1)        # [B]: commanded planar speed
    # TODO(student): build the score in five steps.
    # 1. Reference direction: unit vector along command[:, :2]. When speed is 0
    #    use body +x, i.e. (1, 0), so there is no division by zero (torch.where).
    # 2. Planar error = velocity[:, :2] - command[:, :2]. Split it into a
    #    longitudinal part (dot with the direction) and a transverse part
    #    (dot with the direction rotated 90 degrees).
    # 3. tracking = exp(-(long / sigma)^2 - (trans / lateral_std)^2 - (vz / vertical_std)^2)
    #    where sigma = relative_std * speed, clamped to [std_at_rest, std].
    # 4. progress = 1 - |planar error| / max(speed, std_at_rest), clamped to [0, 1].
    # 5. Return (1 - progress_weight) * tracking + progress_weight * progress, shape [B].
    raise NotImplementedError("TODO: implement track_linear_velocity")


def track_angular_velocity(env, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT,
                           variance: float = 0.5):
    """Track the commanded yaw rate. Positive score in [0, 1].

    A separate provided term penalizes roll and pitch rates, so this one looks
    at the yaw axis only.
    """
    command = env.command_manager.get_command(command_name)                 # [B, 3]; yaw-rate command is command[:, 2]
    angular_velocity = env.scene[asset_cfg.name].data.root_link_ang_vel_b  # [B, 3]: measured roll, pitch, yaw rates
    # TODO(student): return exp(-(measured yaw rate - commanded yaw rate)^2 / variance).
    # Perfect tracking gives 1, large errors approach 0. Only index 2 of each
    # tensor matters; changing roll or pitch rate must not change the result.
    raise NotImplementedError("TODO: implement track_angular_velocity")


def upright(env, asset_cfg: SceneEntityCfg = _ROBOT):
    """Score how upright the trunk is. Positive score in [0, 1].

    Gravity expressed in the body frame tells you which way "down" is from the
    robot's point of view. Upright: gravity is (0, 0, -1). Lying on its side:
    gravity is horizontal, z component 0. Upside down: gravity is (0, 0, +1).
    """
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b  # [B, 3]: unit gravity vector, body frame
    # TODO(student): alignment = -gravity[:, 2] (so upright gives +1). Clamp it to
    # [0, 1] FIRST, then square. Upright scores 1; horizontal and inverted both
    # score 0. Squaring before clamping would give an inverted robot a score of 1.
    raise NotImplementedError("TODO: implement upright")


def action_smoothness(env, acceleration_weight: float = 0.1):
    """Positive penalty for jerky actions (weighted negatively in env_cfg.py).

    Smooth actions are easier on real motors. The action manager keeps the
    last three raw policy outputs; histories are zeroed on reset.
    """
    manager = env.action_manager
    action = manager.action                      # [B, 22]: this step's raw action a_t
    prev_action = manager.prev_action            # [B, 22]: a_{t-1}
    prev_prev_action = manager.prev_prev_action  # [B, 22]: a_{t-2}
    # TODO(student): penalty = sum over the 22 joints of (a_t - a_{t-1})^2
    #   + acceleration_weight * sum over joints of (a_t - 2 a_{t-1} + a_{t-2})^2.
    # Return shape [B], nonnegative. A constant action costs 0. An action that
    # changes by the same step every tick has rate cost but zero acceleration cost.
    raise NotImplementedError("TODO: implement action_smoothness")


def standing_pose(env, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT,
                  linear_threshold: float = 0.05, yaw_threshold: float = 0.05,
                  velocity_weight: float = 0.01):
    """Positive penalty for fidgeting when told to stand still (weighted negatively).

    Only applies when the command is (close to) zero; walking is handled by the
    tracking and gait terms.
    """
    data = env.scene[asset_cfg.name].data
    command = env.command_manager.get_command(command_name)       # [B, 3]
    joint_pos = data.joint_pos[:, asset_cfg.joint_ids]            # [B, J]: current joint angles
    default_pos = data.default_joint_pos[:, asset_cfg.joint_ids]  # [B, J]: standing pose
    joint_vel = data.joint_vel[:, asset_cfg.joint_ids]            # [B, J]: joint rates
    # TODO(student):
    #   cost = mean over joints of (joint_pos - default_pos)^2
    #        + velocity_weight * mean over joints of joint_vel^2
    #   gate = clamp(1 - max(|command_xy| / linear_threshold, |command_yaw| / yaw_threshold), 0, 1)
    # Return cost * gate, shape [B]. The gate is 1 for a zero command and falls
    # to 0 once EITHER the linear or the yaw command reaches its own threshold.
    # Normalize linear (m/s) and yaw (rad/s) separately; never add them together.
    raise NotImplementedError("TODO: implement standing_pose")


def air_time(env, sensor_name: str, command_name: str, threshold_min: float = 0.05,
             threshold_max: float = 0.5, linear_threshold: float = 0.2,
             yaw_threshold: float = 0.2):
    """Positive reward for swing phases of a sensible duration while walking.

    Encourages real steps instead of shuffling. Too-short swings score little;
    too-long swings (hopping, holding a foot up) score 0.
    """
    contact = env.scene[sensor_name].data
    airborne_time = contact.current_air_time                 # [B, feet]: seconds this foot has been off the ground
    found = contact.found                                    # [B, feet]: > 0 while the foot touches the ground
    command = env.command_manager.get_command(command_name)  # [B, 3]
    # TODO(student): per foot, a triangular window on airborne_time: 0 at or
    # outside [threshold_min, threshold_max], rising linearly to 1 at the midpoint
    # (hint: 1 - |t - midpoint| / half_width, clamped at 0). Feet in contact score 0.
    # Sum over feet, so the result lies in [0, number of feet]. Then multiply by a
    # moving mask: 1 if |command_xy| > linear_threshold OR |command_yaw| > yaw_threshold
    # (strict), else 0. Return shape [B].
    raise NotImplementedError("TODO: implement air_time")


def foot_slip(env, sensor_name: str, asset_cfg: SceneEntityCfg = _ROBOT):
    """Positive penalty for feet sliding while on the ground (weighted negatively).

    A planted foot should not move horizontally. There is no command gate:
    slipping while standing still is penalized too.
    """
    foot_velocity = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids]  # [B, feet, 3]: world-frame foot velocity
    found = env.scene[sensor_name].data.found                                            # [B, feet]: > 0 while touching; same foot order
    # TODO(student): for each grounded foot take the squared horizontal speed
    # (x and y components only; vertical motion must not count). Airborne feet
    # contribute 0. Sum over feet and return shape [B], nonnegative.
    raise NotImplementedError("TODO: implement foot_slip")
