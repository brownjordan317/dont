from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from mappo.runtime import validate_policy_env

from .utils.envs import build_manager_eval_env, load_eval_policy
from .utils.results import (
    ResultsWriter,
    build_benchmark,
    build_empty_history,
    print_benchmark,
    update_history,
)
from .utils.runners import run_policy_episode

console = Console()


def eval_manager(config: dict) -> dict:
    eval_cfg = config["eval"]
    console.print(Panel.fit("[bold blue]MAPPO Manager Evaluation[/bold blue]"))
    policy = load_eval_policy(config, section="eval")
    env = build_manager_eval_env(config, policy)
    validate_policy_env(policy, env)

    writer = ResultsWriter(config["output"]["results_path"], config)
    history = build_empty_history()
    num_missions = int(eval_cfg["num_missions"])
    deterministic = bool(eval_cfg.get("deterministic", True))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Evaluating manager...", total=num_missions)
            for mission_index in range(num_missions):
                env.reset(options={"num_agents": int(config["drone_settings"]["num_drones"])})
                record = run_policy_episode(
                    policy=policy,
                    env=env,
                    config=config,
                    policy_path=eval_cfg["model_path"],
                    label="manager",
                    deterministic=deterministic,
                    record_visual=False,
                    show_progress=False,
                )
                writer.append_mission(record)
                update_history(history, record)
                progress.update(task, advance=1)
    finally:
        env.close()

    benchmark = build_benchmark(history)
    writer.finalize(benchmark)
    print_benchmark(benchmark, title="MAPPO Manager Evaluation Summary")
    return benchmark
