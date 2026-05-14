from __future__ import annotations

from rich.console import Console

from .manager_eval import eval_manager
from .skill_eval import eval_skills

console = Console()


def eval(config: dict):
    eval_cfg = config["eval"]
    skill_cfg = eval_cfg.get("skill_eval", {}) or {}
    run_manager = bool(eval_cfg.get("run_manager_eval", True))
    run_skills = bool(skill_cfg.get("enabled", False))

    results = {}
    if run_manager:
        results["manager"] = eval_manager(config)
    if run_skills:
        results["skills"] = eval_skills(config)
    if not run_manager and not run_skills:
        console.print("[yellow]Nothing to evaluate: both manager eval and skill_eval are disabled.[/yellow]")
    return results
