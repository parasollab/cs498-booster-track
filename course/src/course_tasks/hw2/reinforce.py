"""HW2 — REINFORCE (vanilla policy gradient). PROVIDED and COMPLETE. Read it; don't edit it.

This is the algorithm from Williams (1992), in the form the HW2 notebook
(`hw2/reinforce_walkthrough.ipynb`) walks through step by step. Do the notebook
first; this file is the same code without the commentary.

You do not modify this file. `ppo.py` starts as a copy of it, and turning that
copy into PPO is Part II of the assignment. Keeping this file untouched is what
lets you still train the REINFORCE baseline (`Course-HW2-*-Reinforce`) for the
comparison — and `diff reinforce.py ppo.py` is exactly your PPO.

    grad J(theta)  =  E[ sum_t  grad log pi(a_t | s_t)  *  A_t ]

    A_t = G_t                baseline="none"
    A_t = G_t - V(s_t)       baseline="value"  (the default)
    G_t = r_t + gamma * (1 - done_t) * G_{t+1}

The environment is finite-horizon and each iteration collects exactly one full
episode per environment, so G_t is an exact Monte-Carlo return.

The four collection/learning methods — act, process_env_step, compute_returns,
update — are the same four methods rsl_rl's PPO implements, so the structure
carries straight over to PPO. Two differences from rsl_rl's exact signatures are
deliberate: `process_env_step` takes only (rewards, dones), and every tensor is a
plain [num_envs, ...] tensor rather than a TensorDict; the runner unwraps those.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from course_tasks.hw2.modules import Critic, GaussianActor
from course_tasks.hw2.storage import RolloutStorage


class Reinforce:
  """Vanilla policy gradient, optionally with a learned value baseline."""

  def __init__(
    self,
    actor: GaussianActor,
    critic: Critic,
    storage: RolloutStorage,
    gamma: float = 0.99,
    learning_rate: float = 1.0e-3,
    entropy_coef: float = 0.005,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    baseline: str = "value",
    normalize_advantage: bool = True,
    device: str | torch.device = "cpu",
  ) -> None:
    self.actor = actor
    self.critic = critic
    self.storage = storage
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.entropy_coef = entropy_coef
    self.value_loss_coef = value_loss_coef
    self.max_grad_norm = max_grad_norm
    self.baseline = baseline
    self.normalize_advantage = normalize_advantage
    self.device = device

    self.params = list(actor.parameters())
    if baseline == "value":
      self.params += list(critic.parameters())
    self.optimizer = torch.optim.Adam(self.params, lr=learning_rate)

  # -- Collection. ----------------------------------------------------------

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    """Sample an action for every environment and stash what update() needs.

    Called inside `torch.inference_mode()` by the runner, so nothing produced
    here carries gradients. That is why `update()` re-scores the actions.
    """
    actions, log_probs = self.actor.act(obs)
    values = self.critic(obs) if self.baseline == "value" else torch.zeros_like(log_probs)
    self._transition = (obs, actions, log_probs, values)
    return actions

  def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    """Record the outcome of the action `act` just returned."""
    obs, actions, log_probs, values = self._transition
    self.storage.add_transition(obs, actions, log_probs, rewards, dones, values)

  # -- Learning. ------------------------------------------------------------

  def compute_returns(self, last_obs: torch.Tensor) -> None:
    """Discounted return-to-go into `storage.returns`, computed backwards.

    `last_obs` is the observation after the final collected step. REINFORCE
    never uses it: every rollout ends on a genuine terminal, so there is no
    "value of what comes after" to bootstrap. It is in the signature because an
    algorithm whose rollouts can stop mid-episode does need it.
    """
    del last_obs
    st = self.storage
    running = torch.zeros(st.num_envs, device=st.returns.device)
    for t in reversed(range(st.num_transitions_per_env)):
      # (1 - done) stops a return leaking backwards across an episode boundary.
      running = st.rewards[t] + self.gamma * (1.0 - st.dones[t]) * running
      st.returns[t] = running

  def compute_advantages(self) -> None:
    """`storage.advantages` from returns, minus the baseline, optionally normalized."""
    st = self.storage
    if self.baseline == "value":
      advantages = st.returns - st.values
    else:
      advantages = st.returns.clone()
    if self.normalize_advantage:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    st.advantages = advantages

  def update(self) -> dict[str, float]:
    """One full-batch gradient step. Returns losses for logging."""
    obs, actions, _old_log_probs, returns, advantages = self.storage.flatten()

    # Re-score the stored actions under the CURRENT policy: this is the copy of
    # log pi that carries gradients. The stored `old_log_probs` came out of
    # inference_mode and cannot be backpropagated through.
    log_probs, entropy = self.actor.evaluate_actions(obs, actions)

    policy_loss = -(log_probs * advantages).mean()
    entropy_loss = -entropy.mean()
    if self.baseline == "value":
      value_loss = F.mse_loss(self.critic(obs), returns)
    else:
      value_loss = torch.zeros((), device=obs.device)

    loss = (
      policy_loss
      + self.value_loss_coef * value_loss
      + self.entropy_coef * entropy_loss
    )

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
    self.optimizer.step()

    return {
      "policy": policy_loss.item(),
      "value": value_loss.item(),
      "entropy": entropy.mean().item(),
    }

  # -- Plumbing. ------------------------------------------------------------

  def train_mode(self) -> None:
    self.actor.train()
    self.critic.train()

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critic.eval()

  def state_dict(self) -> dict:
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
    }

  def load_state_dict(self, loaded: dict, strict: bool = True) -> None:
    self.actor.load_state_dict(loaded["actor_state_dict"], strict=strict)
    self.critic.load_state_dict(loaded["critic_state_dict"], strict=strict)
    if "optimizer_state_dict" in loaded:
      self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
