"""Login-node checks: no simulation or training."""
import argparse
from dataclasses import fields, is_dataclass
import json
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from booster_mjlab.robots.booster_k1.k1_constants import get_k1_robot_cfg
from mjlab.entity import Entity
from .env_cfg import STUDENT_TERMS, STUDENT_BASELINE_TERMS, CANDIDATE_OMISSIONS
from . import rewards

BASELINE = "Mjlab-Velocity-Flat-Booster-K1"
TEACHING = "Course-Booster-K1-Rewards"
CANDIDATE = TEACHING + "-Candidate"


def normalize(obj):
    if is_dataclass(obj):
        return {f.name: normalize(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [normalize(v) for v in obj]
    if callable(obj):
        return obj.__module__ + ":" + obj.__qualname__
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", action="store_true", help="Print baseline config inventory")
    args = parser.parse_args()
    for name in (BASELINE, TEACHING, CANDIDATE):
        assert name in list_tasks(), name
    for play in (False, True):
        baseline = load_env_cfg(BASELINE, play=play)
        teaching = load_env_cfg(TEACHING, play=play)
        candidate = load_env_cfg(CANDIDATE, play=play)
        expected_names = (set(baseline.rewards) - set(STUDENT_BASELINE_TERMS.values())) | set(STUDENT_TERMS)
        assert set(teaching.rewards) == expected_names
        assert set(candidate.rewards) == expected_names - set(CANDIDATE_OMISSIONS)
        for name in STUDENT_TERMS:
            assert teaching.rewards[name].func is getattr(rewards, name)
            assert candidate.rewards[name].func is getattr(rewards, name)
        for name in expected_names - set(STUDENT_TERMS):
            assert normalize(teaching.rewards[name]) == normalize(baseline.rewards[name]), name
        assert normalize(candidate.rewards) == normalize({k: v for k, v in teaching.rewards.items()
            if k not in CANDIDATE_OMISSIONS}), "Unexpected candidate reward drift"
        # Reward formulas/parameters/weights are the assignment's edit surface.
        teaching.rewards = baseline.rewards
        candidate.rewards = baseline.rewards
        assert normalize(baseline) == normalize(teaching), "Teaching non-reward MDP drift"
        assert normalize(baseline) == normalize(candidate), "Candidate non-reward MDP drift"
    for name in (TEACHING, CANDIDATE):
        reference = load_rl_cfg(BASELINE)
        agent = load_rl_cfg(name)
        agent.experiment_name = reference.experiment_name
        assert normalize(reference) == normalize(agent), "PPO drift"
        assert load_runner_cls(name) is load_runner_cls(BASELINE)
    model = Entity(get_k1_robot_cfg()).spec.compile()
    assert model.nu == 22 and model.nq == 29 and model.nv == 28
    print("PASS: registry, train/play MDP parity, candidate scope, PPO parity, K1 assets")
    if not args.dump:
        return
    env = load_env_cfg(BASELINE)
    print(json.dumps({"actions": normalize(env.actions),
        "scene": normalize(env.scene), "sim": normalize(env.sim),
        "decimation": env.decimation, "episode_length_s": env.episode_length_s,
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu},
        "observations": normalize(env.observations), "rewards": normalize(env.rewards),
        "terminations": normalize(env.terminations), "events": normalize(env.events),
        "commands": normalize(env.commands), "curriculum": normalize(env.curriculum),
        "ppo": normalize(load_rl_cfg(BASELINE))}, indent=2))
