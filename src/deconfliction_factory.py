from rich.console import Console
from rich.panel import Panel
import argparse

from config_utils import load_mode_config

console = Console()

class DeconflictionAutoPilotFactory:
    def __init__(self, mode):
        config_mode = "test" if mode == "test_dubins" else mode
        self.config = self.read_config(config_mode)

        if mode == 'train':
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
        elif mode == 'test_dubins':
            console.print(Panel.fit("[bold magenta]Starting Dubins Visualization Mode[/bold magenta]"))
            from test import create_dubins_sample_visualization
            self.run = create_dubins_sample_visualization

    
    def read_config(self, mode):
        config = load_mode_config(mode)
        console.print(
            f"[green]✓[/green] Config loaded successfully from "
            f"config/{mode}_config.yaml + config/tuning_config.yaml"
        )
        return config
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test", "eval", "test_dubins"],
        default="train",
        help="Mode to run the factory in: 'train', 'test', 'eval', or 'test_dubins'."
    )
    args = parser.parse_args()

    factory = DeconflictionAutoPilotFactory(args.mode)
    factory.run(config=factory.config)

    
