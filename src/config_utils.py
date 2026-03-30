import copy
import os

import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base, override):
    result = copy.deepcopy(base)

    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_mode_config(mode, config_dir="config"):
    mode_path = os.path.join(config_dir, f"{mode}_config.yaml")
    tuning_path = os.path.join(config_dir, "tuning_config.yaml")

    mode_config = load_yaml(mode_path)
    tuning_config = load_yaml(tuning_path)

    config = _deep_merge(mode_config, tuning_config)
    config.setdefault("tuning", tuning_config.get("tuning", {}))
    return config


def get_tuning_section(config, mode):
    tuning = config.get("tuning", {})
    env = tuning.get("env", {})
    shared_env = env.get("shared", {})
    mode_env = env.get(mode, {})
    flight = tuning.get("flight", {})
    shared_flight = flight.get("shared", {})
    mode_flight = flight.get(mode, {})
    rewards = tuning.get("rewards", {})
    hard_safety = tuning.get("hard_safety", {})
    anti_circling = tuning.get("anti_circling", {})

    return {
        "env": _deep_merge(shared_env, mode_env),
        "flight": _deep_merge(shared_flight, mode_flight),
        "rewards": copy.deepcopy(rewards),
        "hard_safety": copy.deepcopy(hard_safety),
        "anti_circling": copy.deepcopy(anti_circling),
    }
