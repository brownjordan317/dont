from __future__ import annotations

from typing import Dict

import numpy as np
from gymnasium import spaces

from mappo import MAPPOPolicy


class RouteSkillOnlyEnv:
    """Run a compact route-follow skill directly on every active drone."""

    def __init__(self, *, base_env, route_skill_policy: MAPPOPolicy):
        self.base_env = base_env
        self.route_skill_policy = route_skill_policy

        self.dt = float(base_env.dt)
        self.max_agents = int(base_env.max_agents)
        self.possible_agents = list(base_env.possible_agents)
        self.agent_name_to_index = dict(base_env.agent_name_to_index)
        self.agents = list(base_env.agents)
        self.obs_dim = int(route_skill_policy.obs_dim)
        self.state_dim = int(route_skill_policy.state_dim)
        self.action_dim = int(route_skill_policy.action_dim)

        self.observation_spaces = {
            agent: spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.action_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self._obs_cache = np.zeros((self.max_agents, self.obs_dim), dtype=np.float32)
        self._state_cache = np.zeros(self.state_dim, dtype=np.float32)
        self._last_policy_train_mask = np.zeros(self.max_agents, dtype=np.float32)
        self._skill_step_totals = np.zeros(self.max_agents, dtype=np.int32)
        self._cache_valid = False

    def __getattr__(self, name: str):
        return getattr(self.base_env, name)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        return self.base_env.seed(seed)

    def _sync_agents(self) -> None:
        self.agents = list(self.base_env.agents)

    def _invalidate_cache(self) -> None:
        self._cache_valid = False

    def _project_obs_matrix(self, base_obs: np.ndarray) -> np.ndarray:
        if int(self.route_skill_policy.obs_dim) == int(self.base_env.obs_dim):
            return np.asarray(base_obs, dtype=np.float32).copy()

        obs_stack_size = int(self.base_env.obs_stack_size)
        source_base_obs_dim = int(self.route_skill_policy.obs_dim) // max(
            obs_stack_size,
            1,
        )
        if (
            obs_stack_size <= 0
            or source_base_obs_dim <= 0
            or source_base_obs_dim * obs_stack_size
            != int(self.route_skill_policy.obs_dim)
        ):
            raise ValueError(
                "Route skill observation shape is incompatible with test env: "
                f"policy_obs_dim={self.route_skill_policy.obs_dim}, "
                f"env_obs_dim={self.base_env.obs_dim}"
            )

        copied_self_features = min(
            int(self.base_env.self_feature_count),
            source_base_obs_dim,
            int(self.base_env.base_obs_dim),
        )
        projected_obs = np.zeros(
            (self.max_agents, int(self.route_skill_policy.obs_dim)),
            dtype=np.float32,
        )
        self._fill_neutral_non_self_features(
            projected_obs,
            source_base_obs_dim=source_base_obs_dim,
            copied_self_features=copied_self_features,
        )
        for agent in self.base_env.agents:
            agent_idx = self.agent_name_to_index[agent]
            for stack_idx in range(obs_stack_size):
                source_offset = stack_idx * int(self.base_env.base_obs_dim)
                target_offset = stack_idx * source_base_obs_dim
                projected_obs[
                    agent_idx,
                    target_offset: target_offset + copied_self_features,
                ] = base_obs[
                    agent_idx,
                    source_offset: source_offset + copied_self_features,
                ]
        return projected_obs

    def _fill_neutral_non_self_features(
        self,
        projected_obs: np.ndarray,
        *,
        source_base_obs_dim: int,
        copied_self_features: int,
    ) -> None:
        tail_width = int(source_base_obs_dim) - int(copied_self_features)
        if tail_width <= 0:
            return
        neutral_neighbor = np.asarray(
            getattr(self.base_env, "NO_NEIGHBOR_FEATURE_VECTOR", ()),
            dtype=np.float32,
        )
        if neutral_neighbor.size <= 0:
            return

        neutral_tail = np.zeros(tail_width, dtype=np.float32)
        cursor = 0
        while cursor < tail_width:
            copy_width = min(int(neutral_neighbor.size), tail_width - cursor)
            neutral_tail[cursor: cursor + copy_width] = neutral_neighbor[:copy_width]
            cursor += copy_width

        obs_stack_size = int(self.base_env.obs_stack_size)
        for stack_idx in range(obs_stack_size):
            target_offset = stack_idx * int(source_base_obs_dim)
            tail_start = target_offset + int(copied_self_features)
            tail_stop = target_offset + int(source_base_obs_dim)
            projected_obs[:, tail_start:tail_stop] = neutral_tail

    def _project_state(self, projected_obs: np.ndarray) -> np.ndarray:
        if int(self.route_skill_policy.state_dim) == int(self.base_env.state_dim):
            return np.asarray(self.base_env.state(), dtype=np.float32).copy()

        source_max_agents = 0
        denom = int(self.route_skill_policy.obs_dim) + 1
        numerator = int(self.route_skill_policy.state_dim) - 2
        if denom > 0 and numerator > 0 and numerator % denom == 0:
            source_max_agents = int(numerator // denom)
        if source_max_agents <= 0:
            raise ValueError(
                "Route skill state shape is incompatible with test env: "
                f"policy_state_dim={self.route_skill_policy.state_dim}, "
                f"env_state_dim={self.base_env.state_dim}"
            )

        projected_state = np.zeros(int(self.route_skill_policy.state_dim), dtype=np.float32)
        copied_agents = min(source_max_agents, len(self.base_env.agents))
        for local_idx, agent in enumerate(list(self.base_env.agents)[:copied_agents]):
            agent_idx = self.agent_name_to_index[agent]
            start = local_idx * int(self.route_skill_policy.obs_dim)
            stop = start + int(self.route_skill_policy.obs_dim)
            projected_state[start:stop] = projected_obs[agent_idx]

        active_mask_start = source_max_agents * int(self.route_skill_policy.obs_dim)
        projected_state[active_mask_start: active_mask_start + copied_agents] = 1.0
        projected_state[-2:] = np.asarray(
            [
                self.base_env.box_width_m / max(self.base_env.map_size_scale, 1.0),
                self.base_env.box_height_m / max(self.base_env.map_size_scale, 1.0),
            ],
            dtype=np.float32,
        )
        return projected_state

    def _refresh_cache(self) -> None:
        if self._cache_valid:
            return
        self._sync_agents()
        base_obs = self.base_env.get_obs_matrix()
        self._obs_cache[:] = self._project_obs_matrix(base_obs)
        self._state_cache[:] = self._project_state(self._obs_cache)
        self._cache_valid = True

    def get_obs_matrix(self) -> np.ndarray:
        self._refresh_cache()
        return self._obs_cache.copy()

    def state(self) -> np.ndarray:
        self._refresh_cache()
        return self._state_cache.copy()

    def reset(self, seed=None, options=None):
        self.base_env.reset(seed=seed, options=options)
        self._sync_agents()
        self._last_policy_train_mask[:] = 0.0
        self._skill_step_totals[:] = 0
        self._invalidate_cache()
        self._refresh_cache()
        obs = {
            agent: self._obs_cache[self.agent_name_to_index[agent]].copy()
            for agent in self.agents
        }
        infos = {
            agent: {
                **self.base_env._shared_info(),
                "selected_skill": "route_follow",
                "selected_skill_id": 0,
                "deconfliction_enabled": False,
            }
            for agent in self.agents
        }
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):
        self._sync_agents()
        active_agents = list(self.agents)
        for agent in active_agents:
            idx = self.agent_name_to_index[agent]
            self._last_policy_train_mask[idx] = 1.0
            self._skill_step_totals[idx] += 1

        obs, rewards, terminations, truncations, infos = self.base_env.step(actions)
        self._sync_agents()
        self._invalidate_cache()

        if self.agents:
            self._refresh_cache()
            obs = {
                agent: self._obs_cache[self.agent_name_to_index[agent]].copy()
                for agent in self.agents
            }
        else:
            obs = {}

        augmented_infos = {}
        for agent, info in infos.items():
            augmented_infos[agent] = {
                **info,
                "selected_skill": "route_follow",
                "selected_skill_id": 0,
                "deconfliction_enabled": False,
            }
        return obs, rewards, terminations, truncations, augmented_infos

    def get_episode_metrics(self) -> dict:
        metrics = self.base_env.get_episode_metrics()
        episode_summary = metrics.setdefault("episode_summary", {})
        episode_summary["route_skill_steps_total"] = int(np.sum(self._skill_step_totals))
        episode_summary["avoid_skill_steps_total"] = 0
        for mission_stat in metrics.get("mission_stats", []):
            agent = mission_stat["id"]
            idx = self.agent_name_to_index[agent]
            mission_stat["route_skill_steps"] = int(self._skill_step_totals[idx])
            mission_stat["avoid_skill_steps"] = 0
        return metrics

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()
