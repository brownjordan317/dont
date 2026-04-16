from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config_utils import load_mode_config
from inference_setup import create_test_environment, load_policy_for_config
from mappo_runtime import (
    action_dict_from_step,
    sample_info,
    select_actions,
    validate_policy_env,
)


@dataclass
class RuntimeStepResult:
    rewards: dict[str, float]
    terminations: dict[str, bool]
    truncations: dict[str, bool]
    info: dict
    agent_states: dict[str, dict]
    episode_metrics: Optional[dict]

    @property
    def terminated(self) -> bool:
        return bool(any(self.terminations.values()))

    @property
    def truncated(self) -> bool:
        return bool(any(self.truncations.values()))


class TrainedPolicyRuntime:
    def __init__(self, config: dict):
        self.config = config
        self.policy = load_policy_for_config(config)
        (
            self.env,
            self.top_left,
            self.bottom_right,
            self._default_reset_options,
        ) = create_test_environment(
            config,
            terminate_on_all_waypoints_complete=False,
            allow_live_waypoint_updates=True,
        )
        validate_policy_env(self.policy, self.env)
        self.deterministic = bool(config["test"].get("deterministic", True))
        self.last_info: dict = {}
        self.last_agent_states: dict[str, dict] = {}

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "TrainedPolicyRuntime":
        runtime_config = load_mode_config("test") if config is None else config
        runtime = cls(runtime_config)
        runtime.reset()
        return runtime

    @property
    def agents(self) -> list[str]:
        return list(self.env.agents)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.last_info = {}
        reset_options = dict(self._default_reset_options or {})
        if options:
            reset_options.update(options)
        if reset_options:
            result = self.env.reset(seed=seed, options=reset_options)
        else:
            result = self.env.reset(seed=seed)
        self.last_agent_states = self.env.runtime_agent_snapshots()
        return result

    def step(
        self,
        *,
        deterministic: Optional[bool] = None,
    ) -> RuntimeStepResult:
        if not self.env.agents:
            raise RuntimeError("No active agents are available. Call reset() first.")

        step = select_actions(
            self.policy,
            self.env,
            deterministic=self.deterministic if deterministic is None else deterministic,
            update_stats=False,
        )
        _, rewards, terminations, truncations, infos = self.env.step(
            action_dict_from_step(step)
        )
        self.last_info = sample_info(infos) if infos else {}
        self.last_agent_states = self.env.runtime_agent_snapshots()
        return RuntimeStepResult(
            rewards={agent: float(value) for agent, value in rewards.items()},
            terminations=terminations,
            truncations=truncations,
            info=self.last_info,
            agent_states=dict(self.last_agent_states),
            episode_metrics=self.last_info.get("episode_metrics"),
        )

    def append_waypoints(self, agent: str, waypoints) -> dict:
        snapshot = self.env.append_runtime_waypoints(agent, waypoints)
        self.last_agent_states = self.env.runtime_agent_snapshots()
        return snapshot

    def replace_waypoint_queue(
        self,
        agent: str,
        waypoints,
        *,
        replace_current: bool = False,
    ) -> dict:
        snapshot = self.env.replace_runtime_waypoint_queue(
            agent,
            waypoints,
            replace_current=replace_current,
        )
        self.last_agent_states = self.env.runtime_agent_snapshots()
        return snapshot

    def agent_state(self, agent: str) -> dict:
        if self.env.aircraft_by_agent:
            state = self.env.runtime_agent_snapshot(agent)
            self.last_agent_states[agent] = state
            return state
        if agent in self.last_agent_states:
            return dict(self.last_agent_states[agent])
        raise KeyError(f"Agent {agent!r} is not available in the runtime.")

    def agent_states(self) -> dict[str, dict]:
        if self.env.aircraft_by_agent:
            self.last_agent_states = self.env.runtime_agent_snapshots()
        return dict(self.last_agent_states)

    def waypoint_state(self, agent: str) -> dict:
        return self.agent_state(agent)

    def waypoint_states(self) -> dict[str, dict]:
        return self.agent_states()

    def target_state(self, agent: str, target_id: str) -> dict:
        return self.env.runtime_target_snapshot(agent, target_id)

    def target_states(self, agent: str) -> dict[str, dict]:
        return self.env.runtime_target_snapshots(agent)

    def get_episode_metrics(self) -> dict:
        return self.env.get_episode_metrics()

    def close(self) -> None:
        self.env.close()
