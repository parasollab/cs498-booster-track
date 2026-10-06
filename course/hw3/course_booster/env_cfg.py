"""Course rewards; all non-reward settings come from the unchanged Flat factory."""
from copy import deepcopy
from mjlab.managers.reward_manager import RewardTermCfg
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import booster_k1_flat_env_cfg
from . import rewards

STUDENT_BASELINE_TERMS = {
    "track_linear_velocity": "track_linear_velocity",
    "track_angular_velocity": "track_angular_velocity", "upright": "upright",
    "action_smoothness": "action_rate_l2", "standing_pose": "standing_pose_l1",
    "air_time": "air_time", "foot_slip": "foot_slip",
}
STUDENT_TERMS = tuple(STUDENT_BASELINE_TERMS)
STUDENT_WEIGHTS = {
    "track_linear_velocity": 2.25, "track_angular_velocity": 2.0, "upright": 1.0,
    "action_smoothness": -0.1, "standing_pose": -0.5, "air_time": 0.1, "foot_slip": -0.2,
}
STUDENT_PARAMS = {
    "track_linear_velocity": dict(command_name="twist", std=0.5, std_at_rest=0.1,
        relative_std=0.75, lateral_std=0.3, vertical_std=0.5, progress_weight=0.5),
    "track_angular_velocity": dict(command_name="twist", variance=0.5),
    "upright": {}, "action_smoothness": dict(acceleration_weight=0.1),
    "standing_pose": dict(command_name="twist", linear_threshold=0.05,
        yaw_threshold=0.05, velocity_weight=0.01),
    "air_time": dict(sensor_name="feet_ground_contact", command_name="twist",
        threshold_min=0.05, threshold_max=0.5, linear_threshold=0.2, yaw_threshold=0.2),
    "foot_slip": dict(sensor_name="feet_ground_contact"),
}
# Optional ablation task: drops two provided shaping terms, keeps everything else.
CANDIDATE_OMISSIONS = ("angular_momentum", "foot_swing_height")


def course_env_cfg(play=False, candidate=False):
    cfg = booster_k1_flat_env_cfg(play=play)
    for course_name, baseline_name in STUDENT_BASELINE_TERMS.items():
        old = cfg.rewards.pop(baseline_name)
        params = deepcopy(STUDENT_PARAMS[course_name])
        if course_name in {"standing_pose", "foot_slip"}:
            params["asset_cfg"] = deepcopy(old.params["asset_cfg"])
        cfg.rewards[course_name] = RewardTermCfg(func=getattr(rewards, course_name),
            weight=STUDENT_WEIGHTS[course_name], params=params)
    if candidate:
        for name in CANDIDATE_OMISSIONS:
            cfg.rewards.pop(name)
    return cfg
