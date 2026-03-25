import os
from typing import Callable, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _candidate_vecnormalize_paths(model_path: str) -> list[str]:
    root, _ = os.path.splitext(model_path)
    return [
        f"{root}_vecnormalize.pkl",
        f"{root}_vecnorm.pkl",
    ]


def resolve_vecnormalize_path(
    model_path: str,
    explicit_path: Optional[str] = None,
) -> Optional[str]:
    """Return the first existing VecNormalize stats path for a model."""
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else None

    for candidate in _candidate_vecnormalize_paths(model_path):
        if os.path.exists(candidate):
            return candidate
    return None


def default_vecnormalize_save_path(model_path: str) -> str:
    """Standard save path used for new VecNormalize statistics files."""
    return _candidate_vecnormalize_paths(model_path)[0]


def make_training_env(env_fn: Callable[[], object], train_config: dict):
    """Create the training VecEnv, optionally wrapping it with VecNormalize."""
    vec_env = DummyVecEnv([env_fn])

    vec_cfg = train_config.get("vecnormalize", {})
    if not vec_cfg.get("enabled", True):
        return vec_env

    return VecNormalize(
        vec_env,
        norm_obs=vec_cfg.get("norm_obs", True),
        norm_reward=vec_cfg.get("norm_reward", True),
        clip_obs=float(vec_cfg.get("clip_obs", 10.0)),
        clip_reward=float(vec_cfg.get("clip_reward", 10.0)),
        gamma=float(train_config.get("gamma", 0.99)),
    )


def make_eval_env(
    env_fn: Callable[[], object],
    model_path: str,
    vecnormalize_path: Optional[str] = None,
    norm_reward: bool = False,
) -> Tuple[object, Optional[str]]:
    """
    Create a VecEnv for inference/evaluation and load saved normalization stats
    when available.
    """
    vec_env = DummyVecEnv([env_fn])
    stats_path = resolve_vecnormalize_path(model_path, vecnormalize_path)

    if stats_path is None:
        return vec_env, None

    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = norm_reward
    return vec_env, stats_path


def save_vecnormalize_stats(model, model_path: str) -> Optional[str]:
    """Persist VecNormalize running statistics beside a saved model."""
    vec_env = model.get_vec_normalize_env()
    if vec_env is None:
        return None

    stats_path = default_vecnormalize_save_path(model_path)
    vec_env.save(stats_path)
    return stats_path


def unwrap_env(env):
    """Return the base Gym env from DummyVecEnv/VecNormalize wrappers."""
    current = env
    while hasattr(current, "venv"):
        current = current.venv

    if hasattr(current, "envs"):
        return current.envs[0].unwrapped

    return current.unwrapped if hasattr(current, "unwrapped") else current


def reset_rollout_env(env):
    """Reset a raw Gym env or VecEnv and return the observation."""
    obs = env.reset()
    return obs[0] if isinstance(obs, tuple) else obs


def step_rollout_env(env, action):
    """Step a raw Gym env or VecEnv and return a consistent rollout tuple."""
    step_result = env.step(action)

    if len(step_result) == 4:
        obs, rewards, dones, infos = step_result
        reward = float(rewards[0]) if np.ndim(rewards) > 0 else float(rewards)
        done = bool(dones[0]) if np.ndim(dones) > 0 else bool(dones)
        info = infos[0] if isinstance(infos, list) else infos
        return obs, reward, done, info

    obs, reward, done, _, info = step_result
    return obs, float(reward), bool(done), info
