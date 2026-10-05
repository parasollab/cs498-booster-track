"""Student property checks: necessary conditions, not a complete correctness oracle."""
import ast
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import course_booster
import torch
from mjlab.tasks.registry import load_env_cfg
from course_booster import rewards
from course_booster.env_cfg import STUDENT_TERMS
from reward_fixtures import fixture


def parameters():
    cfg = load_env_cfg("Course-Booster-K1-Rewards")
    result = {}
    for name in STUDENT_TERMS:
        params = deepcopy(cfg.rewards[name].params)
        if "asset_cfg" in params:
            params["asset_cfg"].joint_ids = slice(None)
            params["asset_cfg"].site_ids = [0, 1]
        result[name] = params
    return result


def main():
    params = parameters()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for batch in (1, 8):
            env, cmd, data, contact = fixture(batch, device)
            for name in STUDENT_TERMS:
                out = getattr(rewards, name)(env, **params[name])
                assert out.shape == (batch,) and out.device == data.joint_pos.device, name
                assert torch.isfinite(out).all() and (out >= 0).all(), name
            # Tracking improves as error shrinks; zero command remains well-defined.
            for name, state in (("track_linear_velocity", data.root_link_lin_vel_b),
                                ("track_angular_velocity", data.root_link_ang_vel_b)):
                state[:] = 1
                far = getattr(rewards, name)(env, **params[name])
                state[:] = 0.1
                near = getattr(rewards, name)(env, **params[name])
                state[:] = 0
                perfect = getattr(rewards, name)(env, **params[name])
                assert (perfect > near).all() and (near > far).all()
            data.root_link_ang_vel_b[:, :2] = 10
            assert torch.allclose(rewards.track_angular_velocity(env, **params["track_angular_velocity"]), perfect)
            up = rewards.upright(env, **params["upright"])
            data.projected_gravity_b[:] = torch.tensor([1., 0., 0.], device=device)
            horizontal = rewards.upright(env, **params["upright"])
            data.projected_gravity_b[:] = torch.tensor([0., 0., 1.], device=device)
            inverted = rewards.upright(env, **params["upright"])
            assert (up > horizontal).all() and torch.equal(horizontal, inverted)
            data.joint_pos[:] = 0.3
            rest = rewards.standing_pose(env, **params["standing_pose"])
            cmd[:, 0] = 0.025
            partial = rewards.standing_pose(env, **params["standing_pose"])
            cmd[:, 0] = 1
            moving = rewards.standing_pose(env, **params["standing_pose"])
            assert (rest > partial).all() and (partial > moving).all() and (moving == 0).all()
            cmd[:] = 0; cmd[:, 2] = 1
            assert (rewards.standing_pose(env, **params["standing_pose"]) == 0).all()
            env.action_manager.action[:] = 0.1
            small = rewards.action_smoothness(env, **params["action_smoothness"])
            env.action_manager.action[:] = 1
            large = rewards.action_smoothness(env, **params["action_smoothness"])
            assert (large > small).all()
            env.action_manager.prev_action[:] = 1; env.action_manager.prev_prev_action[:] = 1
            assert (rewards.action_smoothness(env, **params["action_smoothness"]) == 0).all()
            env.action_manager.prev_prev_action[:] = 0
            assert (rewards.action_smoothness(env, **params["action_smoothness"]) > 0).all()
            contact.current_air_time[:] = 0.275
            peak = rewards.air_time(env, **params["air_time"])
            contact.current_air_time[:] = 0.05
            boundary = rewards.air_time(env, **params["air_time"])
            assert (peak > boundary).all() and torch.allclose(boundary, torch.zeros_like(boundary), atol=1e-6)
            contact.current_air_time[:] = 0.7
            assert (rewards.air_time(env, **params["air_time"]) == 0).all()
            contact.current_air_time[:] = 0.275; contact.found[:] = 1
            assert (rewards.air_time(env, **params["air_time"]) == 0).all()
            contact.found[:] = 0; cmd[:] = 0
            assert (rewards.air_time(env, **params["air_time"]) == 0).all()
            data.site_lin_vel_w[:, :, 2] = 10; contact.found[:] = 1
            assert (rewards.foot_slip(env, **params["foot_slip"]) == 0).all()
            data.site_lin_vel_w[:, :, 0] = 1
            assert (rewards.foot_slip(env, **params["foot_slip"]) > 0).all()
            contact.found[:] = 0
            assert (rewards.foot_slip(env, **params["foot_slip"]) == 0).all()
        # Mixed batches must behave independently of order.
        env, cmd, data, contact = fixture(8, device)
        torch.manual_seed(11)
        cmd[:] = torch.randn_like(cmd)
        for value in vars(data).values(): value[:] = torch.randn_like(value)
        contact.found[:] = torch.randint(0, 2, contact.found.shape, device=device)
        contact.current_air_time[:] = torch.rand_like(contact.current_air_time)
        for value in vars(env.action_manager).values(): value[:] = torch.randn_like(value)
        before = {n: getattr(rewards, n)(env, **params[n]) for n in STUDENT_TERMS}
        permutation = torch.randperm(8, device=device)
        cmd[:] = cmd[permutation].clone()
        for obj in (data, contact, env.action_manager):
            for value in vars(obj).values(): value[:] = value[permutation].clone()
        for n in STUDENT_TERMS:
            torch.testing.assert_close(getattr(rewards, n)(env, **params[n]), before[n][permutation])
    tree = ast.parse(Path(rewards.__file__).read_text())
    assert not any(isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp)) for node in ast.walk(tree)), "Use batched Torch rather than environment loops"
    print("PASS: seven reward properties, B=1/8, batch permutation, device, gates, contacts, action history")


if __name__ == "__main__": main()
