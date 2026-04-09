from src.runtime_builder import load_default_config
from src.visualizer import run_visualiser


def main():
    return run_visualiser(load_default_config())


if __name__ == "__main__":
    main()
