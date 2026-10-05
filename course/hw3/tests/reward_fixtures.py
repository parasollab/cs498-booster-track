"""Small state fixtures: simulator-free and reusable for your own checks."""
from types import SimpleNamespace as NS
import torch


def fixture(batch=4, device="cpu"):
    zeros = lambda *shape: torch.zeros(*shape, device=device)
    command = zeros(batch, 3)
    data = NS(root_link_lin_vel_b=zeros(batch, 3), root_link_ang_vel_b=zeros(batch, 3),
        projected_gravity_b=zeros(batch, 3), joint_pos=zeros(batch, 22),
        joint_vel=zeros(batch, 22), default_joint_pos=zeros(batch, 22),
        site_lin_vel_w=zeros(batch, 2, 3))
    data.projected_gravity_b[:, 2] = -1
    contacts = NS(found=zeros(batch, 2), current_air_time=zeros(batch, 2))
    manager = NS(action=zeros(batch, 22), prev_action=zeros(batch, 22), prev_prev_action=zeros(batch, 22))
    env = NS(num_envs=batch, device=device,
        scene={"robot": NS(data=data), "feet_ground_contact": NS(data=contacts)},
        command_manager=NS(get_command=lambda name: command), action_manager=manager)
    return env, command, data, contacts
