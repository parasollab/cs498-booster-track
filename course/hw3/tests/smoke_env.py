"""Bounded reset/steps, sensor order and action history; CPU or allocated GPU."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import course_booster  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Mjlab-Velocity-Flat-Booster-K1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = 2
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (2, 75) and obs["critic"].shape == (2, 90)
        contact = env.scene["feet_ground_contact"]
        assert [name.rsplit("/", 1)[-1] for name in contact.primary_names] == ["left_foot_link", "right_foot_link"]
        sites = env.reward_manager.get_term_cfg("foot_slip").params["asset_cfg"].site_ids
        robot = env.scene["robot"]
        assert [robot.site_names[i].rsplit("/", 1)[-1] for i in sites] == ["left_foot", "right_foot"]
        assert contact.data.found.shape == (2, 2) and contact.data.current_air_time.shape == (2, 2)
        manager = env.action_manager
        assert all(torch.count_nonzero(value) == 0 for value in (manager.action, manager.prev_action, manager.prev_prev_action))
        for level in (0.0, 0.05, -0.05):
            previous, older = manager.action.clone(), manager.prev_action.clone()
            actions = torch.full((2, 22), level, device=args.device)
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert reward.shape == (2,) and torch.isfinite(reward).all()
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
            valid = ~(terminated | truncated)
            torch.testing.assert_close(manager.action[valid], actions[valid])
            torch.testing.assert_close(manager.prev_action[valid], previous[valid])
            torch.testing.assert_close(manager.prev_prev_action[valid], older[valid])
        env.reset()
        assert all(torch.count_nonzero(value) == 0 for value in (manager.action, manager.prev_action, manager.prev_prev_action))
        print(f"PASS: {args.task}, reset + 3 steps, actor=75, critic=90, actions=22, foot order/history/reset")
    finally:
        env.close()


if __name__ == "__main__":
    main()
