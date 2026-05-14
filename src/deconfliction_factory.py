import os
from rich.console import Console
from rich.panel import Panel
import argparse

from config_utils import load_mode_config

console = Console()

class DeconflictionAutoPilotFactory:
    def __init__(self, mode):
        config_mode = "eval" if mode == "eval_skills" else mode
        self.config = self.read_config(config_mode)

        if mode == 'train':
            console.print(Panel.fit("[bold green]Starting Training Mode[/bold green]"))
            self.run = self.run_training_pipeline
        elif mode in {'train_route_skill', 'train_avoid_skill', 'train_manager'}:
            console.print(Panel.fit("[bold green]Starting Training Mode[/bold green]"))
            from train import train
            self.run = train
        elif mode == 'test':
            console.print(Panel.fit("[bold blue]Starting Testing Mode[/bold blue]"))
            from test import test
            self.run = test
        elif mode == 'eval':
            console.print(Panel.fit("[bold red]Starting Evaluation Mode[/bold red]"))
            from eval import eval
            self.run = eval
        elif mode == 'eval_skills':
            console.print(Panel.fit("[bold red]Starting Per-Skill Evaluation Mode[/bold red]"))
            from eval import eval
            self.config.setdefault("eval", {}).setdefault("skill_eval", {})["enabled"] = True
            self.config.setdefault("eval", {})["run_manager_eval"] = False
            self.run = eval

    
    def read_config(self, mode):
        config = load_mode_config(mode)
        console.print(
            f"[green]✓[/green] Config loaded successfully from "
            f"config/{mode}.yaml + config/tuning.yaml"
        )
        return config

    def _manager_skill_paths(self, config):
        train_cfg = config["train"]
        manager_cfg = train_cfg.get("manager", {})
        return (
            manager_cfg.get("route_skill_path"),
            manager_cfg.get("avoid_skill_path"),
        )

    def run_training_pipeline(self, config=None):
        from train import (
            avoid_checkpoint_contract_mismatches,
            route_checkpoint_contract_mismatches,
            train,
        )

        manager_config = self.config if config is None else config
        train_cfg = manager_config["train"]
        role = str(train_cfg.get("role", "manager")).strip().lower()
        if role != "manager":
            return train(config=manager_config)

        route_skill_path, avoid_skill_path = self._manager_skill_paths(manager_config)
        route_config = None
        route_checkpoint_reasons = []
        route_checkpoint_exists = bool(
            route_skill_path and os.path.exists(route_skill_path)
        )
        if route_checkpoint_exists:
            route_config = self.read_config("train_route_skill")
            route_checkpoint_reasons = route_checkpoint_contract_mismatches(
                str(route_skill_path),
                route_config["train"],
                expected_tuning_cfg=route_config.get("tuning", {}),
                device=str(train_cfg.get("device", "cpu")),
            )
        missing_route_skill = (
            not route_skill_path
            or not route_checkpoint_exists
            or bool(route_checkpoint_reasons)
        )
        avoid_config = None
        avoid_checkpoint_reasons = []
        avoid_checkpoint_exists = bool(
            avoid_skill_path and os.path.exists(avoid_skill_path)
        )
        if avoid_checkpoint_exists:
            avoid_config = self.read_config("train_avoid_skill")
            avoid_checkpoint_reasons = avoid_checkpoint_contract_mismatches(
                str(avoid_skill_path),
                avoid_config["train"],
                expected_tuning_cfg=avoid_config.get("tuning", {}),
                device=str(train_cfg.get("device", "cpu")),
            )
        missing_avoid_skill = (
            not avoid_skill_path
            or not avoid_checkpoint_exists
            or bool(avoid_checkpoint_reasons)
        )

        if missing_route_skill or missing_avoid_skill:
            missing_labels = []
            if missing_route_skill:
                missing_labels.append(
                    "route_follow (outdated)" if route_checkpoint_reasons else "route_follow"
                )
            if missing_avoid_skill:
                missing_labels.append(
                    "avoid (outdated)" if avoid_checkpoint_reasons else "avoid"
                )
            detail_lines = [
                "Manager training needs pretrained low-level skills.",
                f"Bootstrapping missing checkpoints: {', '.join(missing_labels)}",
            ]
            if avoid_checkpoint_reasons:
                detail_lines.append(
                    "Avoid checkpoint contract mismatch: "
                    + "; ".join(avoid_checkpoint_reasons)
                )
            if route_checkpoint_reasons:
                detail_lines.append(
                    "Route checkpoint contract mismatch: "
                    + "; ".join(route_checkpoint_reasons)
                )
            console.print(
                Panel.fit(
                    "\n".join(detail_lines),
                    title="[bold yellow]HRL Bootstrap[/bold yellow]",
                )
            )

            if missing_route_skill:
                if route_config is None:
                    route_config = self.read_config("train_route_skill")
                train(config=route_config)
            if missing_avoid_skill:
                if avoid_config is None:
                    avoid_config = self.read_config("train_avoid_skill")
                train(config=avoid_config)
        console.print(
            Panel.fit(
                "Launching manager training with the available skill checkpoints.",
                title="[bold cyan]HRL Manager[/bold cyan]",
            )
        )
        return train(config=manager_config)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "train_route_skill",
            "train_avoid_skill",
            "train_manager",
            "test",
            "eval",
            "eval_skills",
        ],
        default="train",
        help=(
            "Mode to run the factory in: "
            "'train', 'train_route_skill', 'train_avoid_skill', "
            "'train_manager', 'test', 'eval', "
            "or 'eval_skills'."
        )
    )
    parser.add_argument(
        "--single-drone",
        action="store_true",
        help=(
            "In test mode, set test.gen_mission.num_drones=1 so the route "
            "skill is tested directly."
        ),
    )
    args = parser.parse_args()

    factory = DeconflictionAutoPilotFactory(args.mode)
    if args.single_drone:
        if "test" not in factory.config:
            parser.error("--single-drone can only be used with --mode test.")
        factory.config["test"]["mission_mode"] = "gen_mission"
        factory.config["test"].setdefault("gen_mission", {})["num_drones"] = 1
        factory.config["test"]["deconfliction"] = False
    factory.run(config=factory.config)

    
