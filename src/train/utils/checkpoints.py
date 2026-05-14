from __future__ import annotations

import os

from mappo import MAPPOPolicy


def checkpoint_filename(model_name: str) -> str:
    if model_name.endswith(".pt") or model_name.endswith(".pth"):
        return model_name
    return f"{model_name}.pt"


def checkpoint_stem(model_name: str) -> str:
    filename = checkpoint_filename(model_name)
    return os.path.splitext(filename)[0]


def save_checkpoint(
    policy: MAPPOPolicy,
    *,
    save_dir: str,
    filename: str,
    config: dict,
    timesteps: int,
    updates: int,
    interrupted: bool = False,
) -> str:
    path = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(path) or save_dir, exist_ok=True)
    policy.save(
        path,
        extra={
            "config": config,
            "timesteps": int(timesteps),
            "updates": int(updates),
            "interrupted": bool(interrupted),
        },
    )
    return path


def latest_checkpoint_filename(train_cfg: dict) -> str:
    latest_name = train_cfg.get("latest_model_name")
    if latest_name:
        return checkpoint_filename(str(latest_name))
    return f"{checkpoint_stem(train_cfg['model_name'])}_latest.pt"


def best_checkpoint_filename(train_cfg: dict) -> str:
    best_name = train_cfg.get("best_model_name")
    if best_name:
        return checkpoint_filename(str(best_name))
    return f"{checkpoint_stem(train_cfg['model_name'])}_best.pt"


def best_checkpoint_score(
    metrics: dict,
    *,
    primary_metric: str = "completion_rate",
) -> tuple[float, float, float, float, float]:
    """Lexicographic score for keeping the best RL checkpoint.

    Mission-style tasks usually want completion first. Long-horizon survival
    tasks can choose mean_length first because full completion is intentionally
    sparse early in training. Return and safety break ties.
    """
    score_fields = {
        "completion_rate": float(metrics.get("completion_rate", 0.0)),
        "mean_return": float(metrics.get("mean_return", 0.0)),
        "mean_length": float(metrics.get("mean_length", 0.0)),
        "crash_rate": -float(metrics.get("crash_rate", 0.0)),
        "geofence_outside_steps": -float(metrics.get("geofence_outside_steps", 0.0)),
    }
    ordered_fields = [
        str(primary_metric or "completion_rate"),
        "completion_rate",
        "mean_return",
        "mean_length",
        "crash_rate",
        "geofence_outside_steps",
    ]
    score = []
    used = set()
    for field in ordered_fields:
        if field in used:
            continue
        used.add(field)
        score.append(score_fields.get(field, float(metrics.get(field, 0.0))))
    return tuple(score)


def should_update_best_checkpoint(
    metrics: dict,
    *,
    best_score: tuple[float, ...] | None,
    recent_episode_count: int,
    min_recent_episodes: int,
    primary_metric: str = "completion_rate",
) -> bool:
    if recent_episode_count < max(int(min_recent_episodes), 1):
        return False
    score = best_checkpoint_score(metrics, primary_metric=primary_metric)
    return best_score is None or score > best_score


def scheduled_learning_rate(initial_lr: float, final_lr: float, progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    return float(initial_lr) + (float(final_lr) - float(initial_lr)) * progress


def should_stop_for_stale_best_checkpoint(
    *,
    timesteps: int,
    best_timestep: int | None,
    patience_timesteps: int,
    min_timesteps: int,
) -> bool:
    if best_timestep is None or patience_timesteps <= 0:
        return False
    if timesteps < max(int(min_timesteps), 0):
        return False
    return int(timesteps) - int(best_timestep) >= int(patience_timesteps)
