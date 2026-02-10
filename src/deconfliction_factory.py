import yaml
from rich.console import Console
from rich.panel import Panel
import argparse
import os

console = Console()

class DeconflictionAutoPilotFactory:
    def __init__(self, mode):
        config_path = os.path.join("config", f"{mode}_config.yaml")
        self.config = self.read_config(config_path)

        if mode == 'train':
            console.print(Panel.fit("[bold green]Starting Training Mode[/bold green]"))
            from train import train
            self.run = train
        elif mode == 'test':
            console.print(Panel.fit("[bold blue]Starting Testing Mode[/bold blue]"))
            from test import test
            self.run = test

    
    def read_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Config file not found at {config_path}")
            raise
        console.print(f"[green]✓[/green] Config loaded successfully from {config_path}")
        return config
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Mode to run the factory in: 'train' or 'test'."
    )
    args = parser.parse_args()

    factory = DeconflictionAutoPilotFactory(args.mode)
    factory.run(config=factory.config)

    