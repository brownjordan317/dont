import numpy as np
import copy
from geopy.distance import distance as geopy_distance
import yaml

from test import run_light_episode
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position
from gym_env import MultiUAVEnv
from vec_env_utils import make_eval_env, unwrap_env

from stable_baselines3 import A2C, PPO, SAC

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class Evaluator:

    def __init__(self, config):

        if "seed" in config["env"]:
            np.random.seed(config["env"]["seed"])

        self.metrics = {}
        self.history = {}

        self.mission = {
            "config": config,
            "uavs": None
        }

    # -------------------------------------------------------
    # Mission Generation
    # -------------------------------------------------------

    def create_mission(self):

        def generate_origin():
            return (
                np.random.uniform(-70.0, 70.0),
                np.random.uniform(-170.0, 170.0),
            )

        def pick_random_locs(tl, br, num_locs, min_dist=30):

            lat_buffer = abs(tl[0] - br[0]) * 0.1
            lon_buffer = abs(tl[1] - br[1]) * 0.1

            safe_tl = (tl[0] - lat_buffer, tl[1] + lon_buffer)
            safe_br = (br[0] + lat_buffer, br[1] - lon_buffer)

            coords = []

            for _ in range(1000):

                lat = np.random.uniform(safe_br[0], safe_tl[0])
                lon = np.random.uniform(safe_tl[1], safe_br[1])

                if all(
                    geopy_distance((lat, lon), (c[0], c[1])).meters >= min_dist
                    for c in coords
                ):
                    coords.append((lat, lon))

                if len(coords) >= num_locs:
                    return coords

            console.print("[yellow]WARNING: relaxing min_dist[/yellow]")
            return pick_random_locs(tl, br, num_locs, min_dist=10)

        def generate_new_box(min_m, max_m, origin):

            w_m, h_m = np.random.uniform(min_m, max_m, size=2)

            lat_off = (h_m / 2) / 111_320.0
            lon_off = (w_m / 2) / (111_320.0 * np.cos(np.radians(origin[0])))

            tl = (origin[0] + lat_off, origin[1] - lon_off)
            br = (origin[0] - lat_off, origin[1] + lon_off)

            return tl, br

        config = copy.deepcopy(self.mission["config"])

        origin = generate_origin()

        tl, br = generate_new_box(
            config["env"]["box_min_m"],
            config["env"]["box_max_m"],
            origin
        )

        num_drones = config["drone_settings"]["num_drones"]

        init_poses = pick_random_locs(tl, br, num_drones, min_dist=100)

        uavs = []

        for i in range(num_drones):

            uavs.append(
                FixedWingAircraft(
                    id_tag=f"UAV-{i}",
                    initial_position=Position(
                        init_poses[i][0],
                        init_poses[i][1]
                    ),
                    initial_heading=np.random.uniform(0, 360),
                    cruise_speed=config["drone_settings"]["max_cruise_speed"],
                    turning_radius=np.random.uniform(
                        config["drone_settings"]["min_turning_radius"],
                        config["drone_settings"]["max_turning_radius"]
                    ),
                    mission=pick_random_locs(
                        tl,
                        br,
                        num_locs=config["drone_settings"]["num_wps_per_drone"],
                        min_dist=50
                    )
                )
            )

        config["env"]["origin"] = origin
        config["env"]["top_left"] = tl
        config["env"]["bottom_right"] = br

        self.mission["config"] = config
        self.mission["uavs"] = uavs

    # -------------------------------------------------------
    # Environment Creation
    # -------------------------------------------------------

    def create_test_environment(self):

        uavs = copy.deepcopy(self.mission["uavs"])

        return MultiUAVEnv(
            uavs,
            tl=self.mission["config"]["env"]["top_left"],
            br=self.mission["config"]["env"]["bottom_right"],
            dt=self.mission["config"]["env"]["dt"],
            mode="manual_mission",
            caution_dist=self.mission["config"]["env"].get("caution_dist", 30.0),
            critical_dist=self.mission["config"]["env"].get("critical_dist", 3.0),
            inference_mode=False
        )

    # -------------------------------------------------------
    # Load Models
    # -------------------------------------------------------

    def load_models(self):

        algo_map = {"SAC": SAC, "A2C": A2C, "PPO": PPO}

        self.models = {}

        for alg_name, alg_cfg in self.mission["config"]["eval_models"].items():

            if not alg_cfg["model_path"]:
                continue

            console.print(f"[cyan]Loading {alg_name}[/cyan]")

            model_cls = algo_map.get(alg_name)

            model = model_cls.load(
                alg_cfg["model_path"],
                device=alg_cfg["device"]
            )

            self.models[alg_name] = {
                "model": model,
                "model_path": alg_cfg["model_path"],
                "vecnormalize_path": alg_cfg.get("vecnormalize_path"),
            }

            self.history.setdefault(alg_name, {
                "rewards": [],
                "steps": [],
                "wps": [],
                "distances": [],
                "crashes": 0
            })

    # -------------------------------------------------------
    # Run Evaluation
    # -------------------------------------------------------

    def run_tests_on_env(self):

        self.metrics = {}

        for alg_name, model_info in self.models.items():

            raw_env = self.create_test_environment()
            env, stats_path = make_eval_env(
                lambda raw_env=raw_env: raw_env,
                model_path=model_info["model_path"],
                vecnormalize_path=model_info["vecnormalize_path"],
                norm_reward=False,
            )
            base_env = unwrap_env(env)
            base_env.clear_missions = False

            model = model_info["model"]
            model.set_env(env)

            if stats_path:
                console.print(f"[green]{alg_name} VecNormalize:[/green] {stats_path}")
            else:
                console.print(f"[yellow]{alg_name} running without VecNormalize stats.[/yellow]")

            step_count, reward, _, done, truncated, sim = run_light_episode(
                model,
                env,
                self.mission["config"]["env"]["max_steps"],
                alg_name
            )

            self.metrics[alg_name] = {
                "reward": reward,
                "steps": step_count,
                "simulation": sim
            }

            # ---- accumulate stats

            self.history[alg_name]["rewards"].append(reward)
            self.history[alg_name]["steps"].append(step_count)

            total_wp = 0
            reached_wp = 0
            total_dist = 0

            for d in sim["mission_stats"]:

                reached_wp += d["waypoints_reached"]
                total_wp += self.mission["config"]["drone_settings"]["num_wps_per_drone"]

                total_dist += d["dist_navigating"]

            self.history[alg_name]["wps"].append(reached_wp / total_wp)

            avg_dist = total_dist / len(sim["mission_stats"])
            self.history[alg_name]["distances"].append(avg_dist)

            crashes = sim["safety_violations"]["critical"]["total_count"]

            if crashes > 0:
                self.history[alg_name]["crashes"] += 1

            env.close()

    # -------------------------------------------------------
    # Mission Table
    # -------------------------------------------------------

    def show_metrics(self):

        console.print(Panel.fit("[bold blue]Mission Results[/bold blue]"))

        drone_count = len(next(iter(self.metrics.values()))
                          ["simulation"]["mission_stats"])

        table = Table()

        table.add_column("Algo")
        table.add_column("Reward")
        table.add_column("Steps")

        for i in range(drone_count):
            table.add_column(f"WP{i}")
            table.add_column(f"Dist{i}")

        table.add_column("Geofence")
        table.add_column("Caution")
        table.add_column("Critical")

        for alg_name, metrics in self.metrics.items():

            row = [
                alg_name,
                f"{metrics['reward']:.0f}",
                str(metrics["steps"])
            ]

            for drone in metrics["simulation"]["mission_stats"]:

                row.append(
                    f"{drone['waypoints_reached']}/"
                    f"{self.mission['config']['drone_settings']['num_wps_per_drone']}"
                )

                row.append(
                    f"{drone['dist_navigating']:.0f}"
                )

            safety = metrics["simulation"]["safety_violations"]

            row.extend([
                str(safety["geofence"]["total_count"]),
                str(safety["caution"]["total_count"]),
                str(safety["critical"]["total_count"])
            ])

            table.add_row(*row)

        console.print(table)

    # -------------------------------------------------------
    # Benchmark Summary
    # -------------------------------------------------------

    def show_benchmark(self):

        console.print(Panel.fit("[bold green]Benchmark Summary[/bold green]"))

        table = Table()

        table.add_column("Algo")
        table.add_column("Mean Reward")
        table.add_column("Std Reward")
        table.add_column("Avg Steps")
        table.add_column("Avg Dist")
        table.add_column("WP Completion")
        table.add_column("Crash Rate")

        ranking = []

        for alg, stats in self.history.items():

            rewards = np.array(stats["rewards"])
            steps = np.array(stats["steps"])
            dists = np.array(stats["distances"])
            wps = np.array(stats["wps"])

            mean_reward = rewards.mean()
            std_reward = rewards.std()

            avg_steps = steps.mean()
            avg_dist = dists.mean()

            crash_rate = stats["crashes"] / len(rewards)
            wp_rate = np.mean(wps)

            ranking.append((alg, mean_reward))

            table.add_row(
                alg,
                f"{mean_reward:.0f}",
                f"{std_reward:.0f}",
                f"{avg_steps:.0f}",
                f"{avg_dist:.0f} m",
                f"{wp_rate*100:.1f}%",
                f"{crash_rate*100:.1f}%"
            )

        console.print(table)

        ranking.sort(key=lambda x: x[1], reverse=True)

        console.print("\n[bold yellow]Algorithm Ranking[/bold yellow]")

        for i, (alg, score) in enumerate(ranking, 1):
            console.print(f"{i}. {alg} (mean reward {score:.0f})")

    # -------------------------------------------------------

    def run_missions(self, num_missions):

        for i in range(num_missions):

            console.print(
                Panel.fit(
                    f"[bold magenta]Starting Mission {i+1}/{num_missions}[/bold magenta]"
                )
            )

            self.create_mission()
            self.run_tests_on_env()
            self.show_metrics()

        self.show_benchmark()


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------

def eval(config):

    evaluator = Evaluator(config)

    evaluator.load_models()

    evaluator.run_missions(config["eval"]["num_missions"])


if __name__ == "__main__":

    with open("config/eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    eval(config)
