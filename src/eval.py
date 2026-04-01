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


def route_distance_m(start, waypoints):
    points = [start, *waypoints]
    if len(points) < 2:
        return 0.0

    total = 0.0
    for a, b in zip(points, points[1:]):
        total += geopy_distance(a, b).meters
    return float(total)


def box_dimensions_m(top_left, bottom_right):
    if not top_left or not bottom_right:
        return 0.0, 0.0

    mid_lat = (top_left[0] + bottom_right[0]) / 2.0
    mid_lon = (top_left[1] + bottom_right[1]) / 2.0

    width_m = geopy_distance(
        (mid_lat, top_left[1]),
        (mid_lat, bottom_right[1]),
    ).meters
    height_m = geopy_distance(
        (top_left[0], mid_lon),
        (bottom_right[0], mid_lon),
    ).meters

    return float(width_m), float(height_m)


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
                "schema_version": 2,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
            },
            "missions": [],
            "benchmark": None,
        }
        self._flush()

    def append_mission(
        self,
        mission_index: int,
        mission_definition: dict,
        metrics: dict,
        history: dict,
    ):

        record = {
            "mission_index": mission_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": mission_definition.get("origin"),
            "top_left": mission_definition.get("top_left"),
            "bottom_right": mission_definition.get("bottom_right"),
            "mission": mission_definition,
            "algorithms": {},
        }

        for alg_name, m in metrics.items():
            record["algorithms"][alg_name] = {
                "policy": m.get("policy", {}),
                "reward": m["reward"],
                "steps": m["steps"],
                "duration_s": m.get("duration_s", 0),
                "reward_per_step": m.get("reward_per_step", 0),
                "done": m.get("done", False),
                "truncated": m.get("truncated", False),
                "termination_reason": m.get("termination_reason"),
                "completed_mission": m.get("completed_mission", False),
                "saliency_mean": m.get("saliency_mean", 0),
                "saliency_max": m.get("saliency_max", 0),
                "saliency": m.get("saliency", {}),
                "summary": m.get("summary", {}),
                "telemetry": m.get("telemetry", []),
                "mission_stats": m.get("mission_stats", []),
                "episode_summary": m.get("episode_summary", {}),
                "reward_breakdown": m.get("reward_breakdown", {}),
                "safety_violations": m.get("safety_violations", {}),
            }

        record["ranking"] = sorted(
            record["algorithms"].keys(),
            key=lambda name: record["algorithms"][name]["reward"],
            reverse=True,
        )
        record["winner"] = record["ranking"][0] if record["ranking"] else None
        self._data["missions"].append(record)
        self._update_benchmark(history, completed=False)
        self._flush()

    def write_benchmark(self, history: dict):
        self._update_benchmark(history, completed=True)
        self._flush()
        console.print(f"[green]Results saved → {self.path}[/green]")

    def _update_benchmark(self, history: dict, completed: bool):
        summary = {}
        now = datetime.now(timezone.utc).isoformat()

        for alg, stats in history.items():
            if not stats["rewards"]:
                continue

            rewards = np.asarray(stats["rewards"], dtype=float)
            steps = np.asarray(stats["steps"], dtype=float)
            durations = np.asarray(stats["durations_s"], dtype=float)
            reward_per_step = np.asarray(stats["reward_per_step"], dtype=float)
            dists = np.asarray(stats["distances"], dtype=float)
            wps = np.asarray(stats["wps"], dtype=float)
            caution = np.asarray(stats["caution_counts"], dtype=float)
            critical = np.asarray(stats["critical_counts"], dtype=float)
            geofence = np.asarray(stats["geofence_counts"], dtype=float)
            outside_steps = np.asarray(stats["outside_steps"], dtype=float)
            hard_safety = np.asarray(stats["hard_safety_counts"], dtype=float)
            anti_circling = np.asarray(stats["anti_circling_counts"], dtype=float)
            min_sep = np.asarray(stats["min_pairwise_distances"], dtype=float)
            saliency = np.asarray(stats["saliency_means"], dtype=float)
            finite_min_sep = min_sep[np.isfinite(min_sep)]

            summary[alg] = {
                "mean_reward": float(rewards.mean()),
                "std_reward": float(rewards.std()),
                "median_reward": float(np.median(rewards)),
                "p10_reward": float(np.percentile(rewards, 10)),
                "p90_reward": float(np.percentile(rewards, 90)),
                "min_reward": float(rewards.min()),
                "max_reward": float(rewards.max()),
                "avg_steps": float(steps.mean()),
                "avg_duration_s": float(durations.mean()),
                "avg_dist_m": float(dists.mean()),
                "avg_reward_per_step": float(reward_per_step.mean()),
                "wp_completion_rate": float(wps.mean()),
                "mission_completion_rate": stats["completed"] / len(rewards),
                "crash_rate": stats["crashes"] / len(rewards),
                "avg_caution_events": float(caution.mean()),
                "avg_critical_events": float(critical.mean()),
                "avg_geofence_exits": float(geofence.mean()),
                "avg_geofence_outside_steps": float(outside_steps.mean()),
                "avg_hard_safety_uses": float(hard_safety.mean()),
                "avg_anti_circling_uses": float(anti_circling.mean()),
                "avg_min_pairwise_distance_m": (
                    float(finite_min_sep.mean()) if finite_min_sep.size else None
                ),
                "avg_saliency_mean": float(saliency.mean()),
                "status_counts": {
                    "completed": int(stats["completed"]),
                    "critical_violation": int(stats["crashes"]),
                    "max_steps": int(stats["max_steps"]),
                },
                "num_missions": len(rewards),
            }

        self._data["benchmark"] = {
            "status": "completed" if completed else "running",
            "last_updated_at": now,
            "completed_at": now if completed else None,
            "algorithms": summary,
            "ranking": sorted(summary.keys(), key=lambda a: summary[a]["mean_reward"], reverse=True),
        }

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
            "uavs": None,
            "definition": None,
        }

        results_path = config.get("output", {}).get("results_path", "results/eval_results.json")
        self.writer = ResultsWriter(results_path, config)

    def _ordered_waypoints(self, aircraft):
        waypoints = []
        current_wp = aircraft.waypoint_manager.current_waypoint
        if current_wp is not None:
            waypoints.append(current_wp.to_tuple())
        waypoints.extend(wp.to_tuple() for wp in aircraft.waypoint_manager.waypoint_queue)
        return waypoints

    def _build_mission_definition(self):
        origin = self.mission["config"]["env"].get("origin")
        top_left = self.mission["config"]["env"].get("top_left")
        bottom_right = self.mission["config"]["env"].get("bottom_right")
        width_m, height_m = box_dimensions_m(top_left, bottom_right)

        uav_records = []
        total_planned_distance = 0.0

        for aircraft in self.mission["uavs"]:
            mission_waypoints = self._ordered_waypoints(aircraft)
            planned_distance = route_distance_m(
                aircraft.initial_pos.to_tuple(),
                mission_waypoints,
            )
            total_planned_distance += planned_distance

            uav_records.append({
                "id": aircraft.id_tag,
                "initial_position": aircraft.initial_pos.to_tuple(),
                "initial_heading_rad": float(aircraft.initial_heading),
                "initial_heading_deg": float(np.degrees(aircraft.initial_heading)),
                "cruise_speed_mps": float(aircraft.base_cruise_speed),
                "turning_radius_m": float(aircraft.base_turning_radius),
                "waypoints": mission_waypoints,
                "planned_route_distance_m": float(planned_distance),
            })

        return {
            "origin": origin,
            "top_left": top_left,
            "bottom_right": bottom_right,
            "box_width_m": float(width_m),
            "box_height_m": float(height_m),
            "box_area_km2": float((width_m * height_m) / 1_000_000.0),
            "num_drones": len(self.mission["uavs"]),
            "waypoints_per_drone": self.mission["config"]["drone_settings"]["num_wps_per_drone"],
            "total_waypoints": int(sum(len(record["waypoints"]) for record in uav_records)),
            "planned_route_distance_m_total": float(total_planned_distance),
            "planned_route_distance_m_avg": (
                float(total_planned_distance / len(uav_records))
                if uav_records
                else 0.0
            ),
            "uavs": uav_records,
        }

    def _summarize_saliency(self, saliency_values):
        values = np.asarray(saliency_values, dtype=float)
        if values.size == 0:
            return {
                "num_samples": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "max": 0.0,
                "samples": [],
            }

        return {
            "num_samples": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
            "samples": [float(v) for v in values.tolist()],
        }

    def _build_algorithm_metrics(
        self,
        alg_name,
        model_info,
        resolved_vecnormalize_path,
        reward,
        steps,
        done,
        truncated,
        sim,
        saliency_values,
    ):
        env_cfg = get_tuning_section(self.mission["config"], "eval")["env"]
        mission_stats = sim.get("mission_stats", [])
        safety = sim.get("safety_violations", {})
        episode_summary = sim.get("episode_summary", {})
        reward_breakdown = sim.get("reward_breakdown", {})
        telemetry = sim.get("telemetry", [])

        reached_waypoints = sum(
            drone.get("waypoints_reached", 0)
            for drone in mission_stats
        )
        total_waypoints = sum(
            drone.get(
                "assigned_waypoints",
                self.mission["config"]["drone_settings"]["num_wps_per_drone"],
            )
            for drone in mission_stats
        )
        total_distance = sum(
            drone.get("dist_navigating", 0.0)
            for drone in mission_stats
        )
        avg_distance = (
            float(total_distance / len(mission_stats))
            if mission_stats
            else 0.0
        )
        planned_total_distance = self.mission["definition"]["planned_route_distance_m_total"]
        saliency = self._summarize_saliency(saliency_values)

        termination_reason = episode_summary.get("termination_reason")
        if not termination_reason or termination_reason == "in_progress":
            if safety.get("critical", {}).get("total_count", 0) > 0:
                termination_reason = "critical_violation"
            elif total_waypoints and reached_waypoints >= total_waypoints:
                termination_reason = "completed"
            elif steps >= env_cfg["max_steps"]:
                termination_reason = "max_steps"
            else:
                termination_reason = "ended"

        completion_rate = (
            float(reached_waypoints / total_waypoints)
            if total_waypoints
            else 0.0
        )
        duration_s = float(steps * env_cfg["dt"])
        reward_per_step = float(reward / steps) if steps else 0.0
        min_pairwise_distance = episode_summary.get("min_pairwise_distance_m")

        summary = {
            "waypoints_reached_total": int(reached_waypoints),
            "waypoints_total": int(total_waypoints),
            "waypoint_completion_rate": completion_rate,
            "avg_distance_per_uav_m": avg_distance,
            "total_distance_m": float(total_distance),
            "planned_route_distance_m_total": float(planned_total_distance),
            "distance_vs_planned_ratio": (
                float(total_distance / planned_total_distance)
                if planned_total_distance
                else None
            ),
            "caution_events": int(safety.get("caution", {}).get("total_count", 0)),
            "critical_events": int(safety.get("critical", {}).get("total_count", 0)),
            "geofence_exits": int(safety.get("geofence", {}).get("total_count", 0)),
            "geofence_outside_steps": int(safety.get("geofence", {}).get("outside_step_total", 0)),
            "hard_safety_uses": int(safety.get("hard_safety", {}).get("total_count", 0)),
            "anti_circling_uses": int(safety.get("anti_circling", {}).get("total_count", 0)),
            "min_pairwise_distance_m": (
                float(min_pairwise_distance)
                if min_pairwise_distance is not None
                else None
            ),
            "min_pairwise_pair": episode_summary.get("min_pairwise_pair"),
            "min_pairwise_time_s": episode_summary.get("min_pairwise_time_s"),
            "uavs_completed": int(episode_summary.get("uavs_completed", 0)),
            "duration_s": duration_s,
            "reward_per_step": reward_per_step,
            "completed_mission": termination_reason == "completed",
            "crashed": termination_reason == "critical_violation",
            "truncated": bool(truncated or termination_reason == "max_steps"),
        }

        return {
            "algorithm": alg_name,
            "policy": {
                "model_path": model_info["model_path"],
                "configured_vecnormalize_path": model_info.get("vecnormalize_path"),
                "resolved_vecnormalize_path": resolved_vecnormalize_path,
            },
            "reward": float(reward),
            "steps": int(steps),
            "duration_s": duration_s,
            "reward_per_step": reward_per_step,
            "done": bool(done),
            "truncated": summary["truncated"],
            "termination_reason": termination_reason,
            "completed_mission": summary["completed_mission"],
            "saliency_mean": saliency["mean"],
            "saliency_max": saliency["max"],
            "saliency": saliency,
            "summary": summary,
            "telemetry": telemetry,
            "mission_stats": mission_stats,
            "episode_summary": episode_summary,
            "reward_breakdown": reward_breakdown,
            "safety_violations": safety,
            "simulation": sim,
        }

    def _update_history(self, alg_name, algorithm_metrics):
        summary = algorithm_metrics["summary"]

        self.history[alg_name]["rewards"].append(algorithm_metrics["reward"])
        self.history[alg_name]["steps"].append(algorithm_metrics["steps"])
        self.history[alg_name]["durations_s"].append(algorithm_metrics["duration_s"])
        self.history[alg_name]["reward_per_step"].append(algorithm_metrics["reward_per_step"])
        self.history[alg_name]["wps"].append(summary["waypoint_completion_rate"])
        self.history[alg_name]["distances"].append(summary["avg_distance_per_uav_m"])
        self.history[alg_name]["caution_counts"].append(summary["caution_events"])
        self.history[alg_name]["critical_counts"].append(summary["critical_events"])
        self.history[alg_name]["geofence_counts"].append(summary["geofence_exits"])
        self.history[alg_name]["outside_steps"].append(summary["geofence_outside_steps"])
        self.history[alg_name]["hard_safety_counts"].append(summary["hard_safety_uses"])
        self.history[alg_name]["anti_circling_counts"].append(summary["anti_circling_uses"])
        self.history[alg_name]["min_pairwise_distances"].append(
            float(summary["min_pairwise_distance_m"])
            if summary["min_pairwise_distance_m"] is not None
            else np.nan
        )
        self.history[alg_name]["saliency_means"].append(algorithm_metrics["saliency_mean"])

        if summary["crashed"]:
            self.history[alg_name]["crashes"] += 1
        if summary["completed_mission"]:
            self.history[alg_name]["completed"] += 1
        if algorithm_metrics["termination_reason"] == "max_steps":
            self.history[alg_name]["max_steps"] += 1

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
        self.mission["definition"] = self._build_mission_definition()

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
                "durations_s": [],
                "reward_per_step": [],
                "wps": [],
                "distances": [],
                "caution_counts": [],
                "critical_counts": [],
                "geofence_counts": [],
                "outside_steps": [],
                "hard_safety_counts": [],
                "anti_circling_counts": [],
                "min_pairwise_distances": [],
                "saliency_means": [],
                "crashes": 0,
                "completed": 0,
                "max_steps": 0,
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
            step_count, reward, _, done, truncated, sim = run_light_episode(
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
            algorithm_metrics = self._build_algorithm_metrics(
                alg_name,
                model_info,
                stats_path,
                reward,
                step_count,
                done,
                truncated,
                sim,
                saliency_values,
            )
            self.metrics[alg_name] = algorithm_metrics
            self._update_history(alg_name, algorithm_metrics)

            env.close()

    # -------------------------------------------------------

    def show_metrics(self):

        console.print(Panel.fit("[bold blue]Mission Results[/bold blue]"))

        drone_count = len(next(iter(self.metrics.values()))["simulation"]["mission_stats"])

        table = Table()
        table.add_column("Algo")
        table.add_column("Reward")
        table.add_column("Steps")
        table.add_column("Outcome")
        table.add_column("Comp%")
        table.add_column("MinSep")
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
                metrics["termination_reason"],
                f"{metrics['summary']['waypoint_completion_rate']:.0%}",
                (
                    f"{metrics['summary']['min_pairwise_distance_m']:.1f}"
                    if metrics["summary"]["min_pairwise_distance_m"] is not None
                    else "-"
                ),
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
            self.writer.append_mission(
                i + 1,
                self.mission["definition"],
                self.metrics,
                self.history,
            )

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
