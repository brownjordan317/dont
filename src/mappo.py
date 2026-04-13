from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    output_dim: int,
    activation_cls=nn.Tanh,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(activation_cls())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class RunningMeanStd:
    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8
        )
        return np.clip(normalized, -clip, clip)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Iterable[int],
        action_dim: int,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        feature_dim = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.backbone = build_mlp(obs_dim, hidden_sizes, feature_dim)
        self.mean_head = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def _distribution(self, obs: torch.Tensor) -> Normal:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std, -5.0, 1.5)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self._distribution(obs)
        raw_action = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        entropy = dist.entropy().sum(dim=-1)
        return action, raw_action, log_prob.sum(dim=-1), entropy

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(obs)
        return torch.tanh(dist.mean)

    def evaluate_actions(self, obs: torch.Tensor, raw_action: torch.Tensor):
        dist = self._distribution(obs)
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob.sum(dim=-1), entropy


class CentralValueCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        self.network = build_mlp(state_dim, hidden_sizes, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


@dataclass
class RolloutBatch:
    obs: np.ndarray
    states: np.ndarray
    raw_actions: np.ndarray
    log_probs: np.ndarray
    masks: np.ndarray
    reference_actions: np.ndarray
    reference_masks: np.ndarray
    values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


class MAPPOPolicy:
    def __init__(
        self,
        *,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        actor_hidden_sizes: Iterable[int],
        critic_hidden_sizes: Iterable[int],
        device: str = "cpu",
        log_std_init: float = -0.5,
        obs_clip: float = 10.0,
        state_clip: float = 10.0,
    ):
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.actor_hidden_sizes = list(actor_hidden_sizes)
        self.critic_hidden_sizes = list(critic_hidden_sizes)
        self.device = torch.device(device)
        self.obs_clip = float(obs_clip)
        self.state_clip = float(state_clip)

        self.actor = SquashedGaussianActor(
            obs_dim=self.obs_dim,
            hidden_sizes=self.actor_hidden_sizes,
            action_dim=self.action_dim,
            log_std_init=log_std_init,
        ).to(self.device)
        self.critic = CentralValueCritic(
            state_dim=self.state_dim,
            hidden_sizes=self.critic_hidden_sizes,
        ).to(self.device)

        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.state_rms = RunningMeanStd((self.state_dim,))

    def update_normalizers(self, obs: np.ndarray, state: np.ndarray):
        obs = np.asarray(obs, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if obs.size:
            self.obs_rms.update(obs)
        if state.ndim == 1:
            state = state[None, :]
        if state.size:
            self.state_rms.update(state)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        return self.obs_rms.normalize(obs, clip=self.obs_clip)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        return self.state_rms.normalize(state, clip=self.state_clip)

    def act_parallel(
        self,
        obs_batches: List[np.ndarray],
        states: np.ndarray,
        *,
        deterministic: bool = False,
        update_stats: bool = False,
    ) -> dict:
        split_sizes = [int(len(obs_batch)) for obs_batch in obs_batches]
        if split_sizes:
            non_empty = [
                np.asarray(obs_batch, dtype=np.float32)
                for obs_batch in obs_batches
                if len(obs_batch) > 0
            ]
            merged_obs = (
                np.concatenate(non_empty, axis=0)
                if non_empty
                else np.zeros((0, self.obs_dim), dtype=np.float32)
            )
        else:
            merged_obs = np.zeros((0, self.obs_dim), dtype=np.float32)

        states = np.asarray(states, dtype=np.float32)
        if update_stats:
            self.update_normalizers(merged_obs, states)

        merged_obs_norm = self.normalize_obs(merged_obs)
        states_norm = self.normalize_state(states)

        if merged_obs_norm.shape[0] > 0:
            obs_tensor = torch.as_tensor(
                merged_obs_norm,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            obs_tensor = None

        states_tensor = torch.as_tensor(
            states_norm,
            dtype=torch.float32,
            device=self.device,
        )
        if states_tensor.ndim == 1:
            states_tensor = states_tensor.unsqueeze(0)

        with torch.no_grad():
            if obs_tensor is not None:
                actions, raw_actions, log_probs, _ = self.actor.sample(
                    obs_tensor,
                    deterministic=deterministic,
                )
                actions_np = actions.cpu().numpy()
                raw_actions_np = raw_actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
            else:
                actions_np = np.zeros((0, self.action_dim), dtype=np.float32)
                raw_actions_np = np.zeros((0, self.action_dim), dtype=np.float32)
                log_probs_np = np.zeros((0,), dtype=np.float32)

            values_np = self.critic(states_tensor).squeeze(-1).cpu().numpy()

        action_splits = []
        raw_action_splits = []
        log_prob_splits = []
        normalized_obs_splits = []
        start = 0
        for size in split_sizes:
            stop = start + size
            action_splits.append(actions_np[start:stop])
            raw_action_splits.append(raw_actions_np[start:stop])
            log_prob_splits.append(log_probs_np[start:stop])
            normalized_obs_splits.append(merged_obs_norm[start:stop])
            start = stop

        return {
            "actions": action_splits,
            "raw_actions": raw_action_splits,
            "log_probs": log_prob_splits,
            "normalized_obs": normalized_obs_splits,
            "values": values_np,
            "normalized_states": states_norm,
        }

    def value(self, state: np.ndarray) -> float:
        state_norm = self.normalize_state(np.asarray(state, dtype=np.float32))
        state_tensor = torch.as_tensor(
            state_norm,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1)
        return float(value.item())

    def evaluate_actor(
        self,
        obs: torch.Tensor,
        raw_actions: torch.Tensor,
    ):
        return self.actor.evaluate_actions(obs, raw_actions)

    def actor_mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.mean_action(obs)

    def evaluate_critic(self, states: torch.Tensor):
        return self.critic(states).squeeze(-1)

    def save(
        self,
        path: str,
        *,
        extra: Optional[dict] = None,
    ):
        payload = {
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "actor_hidden_sizes": self.actor_hidden_sizes,
            "critic_hidden_sizes": self.critic_hidden_sizes,
            "obs_clip": self.obs_clip,
            "state_clip": self.state_clip,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "state_rms": self.state_rms.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @staticmethod
    def _load_checkpoint_payload(path: str, device: str):
        """
        Load local trusted checkpoints across PyTorch versions.

        PyTorch 2.6 switched `torch.load()` to `weights_only=True` by default,
        which breaks older checkpoints containing numpy-based payload fields.
        We first try the default path, then fall back to full deserialization
        for trusted local checkpoints.
        """
        try:
            return torch.load(path, map_location=device)
        except pickle.UnpicklingError:
            try:
                return torch.load(
                    path,
                    map_location=device,
                    weights_only=False,
                )
            except TypeError:
                return torch.load(path, map_location=device)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MAPPOPolicy":
        payload = cls._load_checkpoint_payload(path, device)
        policy = cls(
            obs_dim=int(payload["obs_dim"]),
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            actor_hidden_sizes=payload["actor_hidden_sizes"],
            critic_hidden_sizes=payload["critic_hidden_sizes"],
            device=device,
            obs_clip=float(payload.get("obs_clip", 10.0)),
            state_clip=float(payload.get("state_clip", 10.0)),
        )
        policy.actor.load_state_dict(payload["actor_state_dict"])
        policy.critic.load_state_dict(payload["critic_state_dict"])
        policy.obs_rms.load_state_dict(payload["obs_rms"])
        policy.state_rms.load_state_dict(payload["state_rms"])
        return policy

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for step in reversed(range(len(rewards))):
        next_value = bootstrap_value if step == len(rewards) - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def build_rollout_batch(
    *,
    obs: List[np.ndarray],
    states: List[np.ndarray],
    raw_actions: List[np.ndarray],
    log_probs: List[np.ndarray],
    masks: List[np.ndarray],
    reference_actions: List[np.ndarray],
    reference_masks: List[np.ndarray],
    values: List[float],
    rewards: List[float],
    dones: List[float],
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> RolloutBatch:
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=np.float32)
    advantages, returns = compute_gae(
        rewards=rewards_arr,
        values=values_arr,
        dones=dones_arr,
        bootstrap_value=bootstrap_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return RolloutBatch(
        obs=np.asarray(obs, dtype=np.float32),
        states=np.asarray(states, dtype=np.float32),
        raw_actions=np.asarray(raw_actions, dtype=np.float32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        masks=np.asarray(masks, dtype=np.float32),
        reference_actions=np.asarray(reference_actions, dtype=np.float32),
        reference_masks=np.asarray(reference_masks, dtype=np.float32),
        values=values_arr,
        returns=returns.astype(np.float32),
        advantages=advantages.astype(np.float32),
    )
