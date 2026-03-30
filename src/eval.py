import json
import os
import numpy as np
import copy
from datetime import datetime, timezone
from geopy.distance import distance as geopy_distance

from test import run_light_episode
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position
from gym_env import MultiUAVEnv
from config_utils import get_tuning_section
from vec_env_utils import make_eval_env, unwrap_env

from stable_baselines3 import A2C, PPO, SAC

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# -------------------------------------------------------
# Universal Saliency (works with A2C / PPO / SAC)
# -------------------------------------------------------

class SaliencyAnalyzer:
    """
    Model-agnostic saliency using observation perturbation.
    Works with any SB3 policy without accessing logits.
    """

    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon

    def compute_saliency(self, obs):
        obs = obs.copy()
        base_action, _ = self.model.predict(obs, deterministic=True)

        saliency = np.zeros_like(obs, dtype=np.float32)

        for i in range(obs.shape[-1]):
            obs_perturbed = obs.copy()
            obs_perturbed[0, i] += self.epsilon

            action2, _ = self.model.predict(obs_perturbed, deterministic=True)

            saliency[0, i] = np.linalg.norm(action2 - base_action)

        return saliency


# -------------------------------------------------------
# Results Writer
# -------------------------------------------------------

class ResultsWriter:

    def __init__(self, path: str, config: dict):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        self._data = {
            "meta": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
            },
            "missions": [],
            "benchmark": None,
        }
        self._flush()

    def append_mission(self, mission_index: int, mission_config: dict, metrics: dict):

        record = {
            "mission_index": mission_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": mission_config["env"].get("origin"),
            "top_left": mission_config["env"].get("top_left"),
            "bottom_right": mission_config["env"].get("bottom_right"),
            "algorithms": {},
        }

        for alg_name, m in metrics.items():
            sim = m["simulation"]

            record["algorithms"][alg_name] = {
                "reward": m["reward"],
                "steps": m["steps"],
                "saliency_mean": m.get("saliency_mean", 0),
                "saliency_max": m.get("saliency_max", 0),
                "mission_stats": sim["mission_stats"],
                "safety_violations": sim["safety_violations"],
            }

        self._data["missions"].append(record)
        self._flush()

    def write_benchmark(self, history: dict):

        summary = {}

        for alg, stats in history.items():

            rewards = np.array(stats["rewards"])
            steps = np.array(stats["steps"])
            dists = np.array(stats["distances"])
            wps = np.array(stats["wps"])

            summary[alg] = {
                "mean_reward": float(rewards.mean()),
                "std_reward": float(rewards.std()),
                "min_reward": float(rewards.min()),
                "max_reward": float(rewards.max()),
                "avg_steps": float(steps.mean()),
                "avg_dist_m": float(dists.mean()),
                "wp_completion_rate": float(wps.mean()),
                "crash_rate": stats["crashes"] / len(rewards),
                "num_missions": len(rewards),
            }

        self._data["benchmark"] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "algorithms": summary,
            "ranking": sorted(summary.keys(), key=lambda a: summary[a]["mean_reward"], reverse=True),
        }

        self._flush()
        console.print(f"[green]Results saved → {self.path}[/green]")

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)


# -------------------------------------------------------
# Evaluator
# -------------------------------------------------------

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

        results_path = config.get("output", {}).get("results_path", "results/eval_results.json")
        self.writer = ResultsWriter(results_path, config)

    # -------------------------------------------------------
    # Mission Generation
    # -------------------------------------------------------

    def create_mission(self):
        eval_env_cfg = get_tuning_section(self.mission["config"], "eval")["env"]
        eval_flight_cfg = get_tuning_section(self.mission["config"], "eval")["flight"]

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
            eval_env_cfg["box_min_m"],
            eval_env_cfg["box_max_m"],
            origin
        )

        num_drones = config["drone_settings"]["num_drones"]

        init_poses = pick_random_locs(tl, br, num_drones, min_dist=100)

        uavs = []

        for i in range(num_drones):

            uavs.append(
                FixedWingAircraft(
                    id_tag=f"UAV-{i}",
                    initial_position=Position(init_poses[i][0], init_poses[i][1]),
                    initial_heading=np.random.uniform(-np.pi, np.pi),
                    cruise_speed=np.random.uniform(
                        eval_flight_cfg["cruise_speed_min_mps"],
                        eval_flight_cfg["cruise_speed_max_mps"]
                    ),
                    turning_radius=np.random.uniform(
                        eval_flight_cfg["turning_radius_min_m"],
                        eval_flight_cfg["turning_radius_max_m"]
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

    def create_test_environment(self):
        tuning = get_tuning_section(self.mission["config"], "eval")
        env_cfg = tuning["env"]
        reward_cfg = tuning["rewards"]
        hard_safety_cfg = tuning["hard_safety"]
        anti_circling_cfg = tuning["anti_circling"]

        uavs = copy.deepcopy(self.mission["uavs"])

        return MultiUAVEnv(
            uavs,
            tl=self.mission["config"]["env"]["top_left"],
            br=self.mission["config"]["env"]["bottom_right"],
            dt=env_cfg["dt"],
            max_steps=env_cfg["max_steps"],
            mode="manual_mission",
            caution_dist=env_cfg["caution_dist"],
            critical_dist=env_cfg["critical_dist"],
            inference_mode=False,
            reward_config=reward_cfg,
            hard_safety_config=hard_safety_cfg,
            anti_circling_config=anti_circling_cfg,
        )

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
    # Evaluation + Saliency
    # -------------------------------------------------------

    def run_tests_on_env(self):

        self.metrics = {}

        for alg_name, model_info in self.models.items():
            env_cfg = get_tuning_section(self.mission["config"], "eval")["env"]

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

            # -------------------------------
            # MAIN EVALUATION (unchanged)
            # -------------------------------
            step_count, reward, obs, done, truncated, sim = run_light_episode(
                model,
                env,
                env_cfg["max_steps"],
                alg_name
            )

            # -------------------------------
            # SALIENCY ROLLOUT
            # -------------------------------
            saliency_values = []

            try:
                obs = env.reset()
                saliency_analyzer = SaliencyAnalyzer(model)

                for _ in range(200):

                    action, _ = model.predict(obs, deterministic=True)

                    # Normalize observation
                    obs_array = None

                    if isinstance(obs, np.ndarray):
                        obs_array = obs
                    elif isinstance(obs, dict):
                        obs_array = np.concatenate(
                            [np.asarray(v).flatten() for v in obs.values()]
                        )[None, :]
                    elif isinstance(obs, tuple):
                        obs_array = np.asarray(obs[0])

                    if obs_array is not None:
                        try:
                            s = saliency_analyzer.compute_saliency(obs_array)
                            saliency_values.append(np.mean(s))
                        except Exception:
                            pass

                    step_result = env.step(action)

                    # ---- Handle 4 or 5 return values
                    if len(step_result) == 5:
                        obs, _, done, truncated, _ = step_result
                    else:
                        obs, _, done, _ = step_result
                        truncated = False

                    if done or truncated:
                        break

            except Exception as e:
                console.print(f"[yellow]Saliency skipped: {e}[/yellow]")

            # -------------------------------
            # STORE METRICS
            # -------------------------------
            self.metrics[alg_name] = {
                "reward": reward,
                "steps": step_count,
                "simulation": sim,
                "saliency_mean": float(np.mean(saliency_values)) if saliency_values else 0,
                "saliency_max": float(np.max(saliency_values)) if saliency_values else 0
            }

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

    def show_metrics(self):

        console.print(Panel.fit("[bold blue]Mission Results[/bold blue]"))

        drone_count = len(next(iter(self.metrics.values()))["simulation"]["mission_stats"])

        table = Table()
        table.add_column("Algo")
        table.add_column("Reward")
        table.add_column("Steps")
        table.add_column("Saliency")

        for i in range(drone_count):
            table.add_column(f"WP{i}")
            table.add_column(f"Dist{i}")

        table.add_column("Geofence")
        table.add_column("Caution")
        table.add_column("Critical")
        table.add_column("HardSafe")
        table.add_column("AntiCircle")

        for alg_name, metrics in self.metrics.items():

            row = [
                alg_name,
                f"{metrics['reward']:.0f}",
                str(metrics["steps"]),
                f"{metrics['saliency_mean']:.4f}"
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
                str(safety["critical"]["total_count"]),
                str(safety["hard_safety"]["total_count"]),
                str(safety["anti_circling"]["total_count"]),
            ])

            table.add_row(*row)

        console.print(table)

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
            self.writer.append_mission(i + 1, self.mission["config"], self.metrics)

        self.writer.write_benchmark(self.history)


# -------------------------------------------------------

def eval(config):
    evaluator = Evaluator(config)
    evaluator.load_models()
    evaluator.run_missions(config["eval"]["num_missions"])


if __name__ == "__main__":
    from config_utils import load_mode_config

    config = load_mode_config("eval")

    eval(config)
