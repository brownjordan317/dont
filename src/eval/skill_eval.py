from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from config_utils import get_tuning_section, load_mode_config
from mappo import MAPPOPolicy
from mappo.runtime import validate_policy_env
from train import build_role_training_env

from .utils.results import (
    ResultsWriter,
    build_benchmark,
    build_empty_history,
    print_benchmark,
    update_history,
)
from .utils.runners import run_policy_episode

console = Console()


@dataclass(frozen=True)
class SkillEvalSpec:
    name: str
    role: str
    model_path: str


_SKILL_TO_ROLE = {
    "route": "route_skill",
    "avoid": "avoid_skill",
}
_SKILL_TO_CONFIG_MODE = {
    "route": "train_route_skill",
    "avoid": "train_avoid_skill",
}
_SKILL_TO_MANAGER_PATH_KEY = {
    "route": "route_skill_path",
    "avoid": "avoid_skill_path",
}


def build_skill_eval_specs(config: dict) -> list[SkillEvalSpec]:
    eval_cfg = config.get("eval", {})
    skill_cfg = eval_cfg.get("skill_eval", {}) or {}
    requested = skill_cfg.get("skills", ["route", "avoid"])
    manager_cfg = eval_cfg.get("manager", {})
    specs: list[SkillEvalSpec] = []
    for raw_name in requested:
        name = str(raw_name).strip().lower()
        if name not in _SKILL_TO_ROLE:
            raise ValueError(f"Unknown skill eval target: {raw_name!r}")
        path_key = _SKILL_TO_MANAGER_PATH_KEY[name]
        model_path = str(skill_cfg.get(f"{name}_model_path") or manager_cfg.get(path_key) or "")
        if not model_path:
            raise ValueError(f"Missing eval skill checkpoint path for {name!r}")
        specs.append(SkillEvalSpec(name=name, role=_SKILL_TO_ROLE[name], model_path=model_path))
    return specs


def _load_skill_training_config(spec: SkillEvalSpec, eval_config: dict) -> tuple[dict, dict]:
    skill_config = load_mode_config(_SKILL_TO_CONFIG_MODE[spec.name])
    train_cfg = dict(skill_config["train"])
    drone_cfg = eval_config.get("drone_settings", {}) or {}
    skill_eval_cfg = eval_config.get("eval", {}).get("skill_eval", {}) or {}
    train_cfg["device"] = eval_config.get("eval", {}).get("device", train_cfg.get("device", "cpu"))
    if "num_missions" in skill_eval_cfg:
        train_cfg["num_eval_missions"] = int(skill_eval_cfg["num_missions"])
    if spec.name == "route" and "num_wps_per_drone" in drone_cfg:
        train_cfg["route_skill_mission_waypoint_count"] = int(
            drone_cfg["num_wps_per_drone"]
        )
    return skill_config, train_cfg


def build_skill_eval_env(spec: SkillEvalSpec, eval_config: dict):
    skill_config, train_cfg = _load_skill_training_config(spec, eval_config)
    tuning = get_tuning_section(skill_config, "train")
    return build_role_training_env(
        tuning=tuning,
        train_cfg=train_cfg,
        role=spec.role,
        device=str(train_cfg.get("device", "cpu")),
    )


def _skill_result_path(base_path: str, skill_name: str) -> str:
    path = Path(base_path)
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    parent = path.parent if str(path.parent) != "." else Path("results")
    return str(parent / f"{stem}_{skill_name}{suffix}")


def _visual_path(visual_dir: str, skill_name: str, mission_index: int) -> str:
    return str(Path(visual_dir) / skill_name / f"{skill_name}_mission_{mission_index + 1:03d}.mp4")


def eval_single_skill(config: dict, spec: SkillEvalSpec) -> dict:
    eval_cfg = config["eval"]
    skill_cfg = eval_cfg.get("skill_eval", {}) or {}
    num_missions = int(skill_cfg.get("num_missions", eval_cfg.get("num_missions", 10)))
    deterministic = bool(skill_cfg.get("deterministic", eval_cfg.get("deterministic", True)))
    save_visuals = bool(skill_cfg.get("save_visuals", True))
    visual_episodes = int(skill_cfg.get("visual_episodes", 1 if save_visuals else 0))
    visual_dir = str(skill_cfg.get("visual_dir", "reports/skill_eval"))

    console.print(Panel.fit(f"[bold blue]Skill Evaluation: {spec.name}[/bold blue]"))
    policy = MAPPOPolicy.load(spec.model_path, device=str(eval_cfg.get("device", "cpu")))
    env = build_skill_eval_env(spec, config)
    validate_policy_env(policy, env)

    result_path = _skill_result_path(config["output"]["results_path"], spec.name)
    writer = ResultsWriter(result_path, {**config, "skill_eval_target": spec.__dict__})
    history = build_empty_history()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Evaluating {spec.name} skill...", total=num_missions)
            for mission_index in range(num_missions):
                env.reset()
                visual_path = (
                    _visual_path(visual_dir, spec.name, mission_index)
                    if save_visuals and mission_index < visual_episodes
                    else None
                )
                record = run_policy_episode(
                    policy=policy,
                    env=env,
                    config=config,
                    policy_path=spec.model_path,
                    label=f"{spec.name}_skill",
                    deterministic=deterministic,
                    record_visual=bool(visual_path),
                    visual_path=visual_path,
                    show_progress=False,
                )
                record["skill"] = spec.name
                record["role"] = spec.role
                writer.append_mission(record)
                update_history(history, record)
                progress.update(task, advance=1)
    finally:
        env.close()

    benchmark = build_benchmark(history)
    benchmark["skill"] = spec.name
    benchmark["role"] = spec.role
    benchmark["model_path"] = spec.model_path
    writer.finalize(benchmark)
    print_benchmark(benchmark, title=f"{spec.name.title()} Skill Evaluation Summary")
    return benchmark


def eval_skills(config: dict) -> dict:
    specs = build_skill_eval_specs(config)
    summaries = {}
    for spec in specs:
        summaries[spec.name] = eval_single_skill(config, spec)
    aggregate_path = Path(config["output"]["results_path"])
    aggregate = {
        "skills": summaries,
        "result_files": {
            name: os.path.abspath(_skill_result_path(str(aggregate_path), name))
            for name in summaries
        },
    }
    return aggregate
