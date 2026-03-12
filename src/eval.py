import numpy as np
import copy
from geopy.distance import distance as geopy_distance
import yaml

from test import run_light_episode
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position
from gym_env import MultiUAVEnv

from stable_baselines3 import A2C, PPO, SAC


class Evaluator:

    def __init__(self, config):

        # Optional reproducibility
        if "seed" in config["env"]:
            np.random.seed(config["env"]["seed"])

        self.metrics = {}

        self.mission = {
            "config": config,
            "uavs": None
        }

        self.create_mission()
        self.load_models()

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

            print(f"[WARNING] Could not meet min_dist {min_dist}m, retrying with 10m")

            return pick_random_locs(tl, br, num_locs, min_dist=10)

        def generate_new_box(min_m, max_m, origin):

            w_m, h_m = np.random.uniform(min_m, max_m, size=2)

            lat_off = (h_m / 2.0) / 111_320.0
            lon_off = (w_m / 2.0) / (
                111_320.0 * np.cos(np.radians(origin[0]))
            )

            tl = (origin[0] + lat_off, origin[1] - lon_off)
            br = (origin[0] - lat_off, origin[1] + lon_off)

            return tl, br

        def update_mission_config(config):

            config = copy.deepcopy(config)

            origin = generate_origin()

            tl, br = generate_new_box(
                int(config["env"]["box_min_m"]),
                int(config["env"]["box_max_m"]),
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
                        cruise_speed=np.random.uniform(
                            config["drone_settings"]["max_cruise_speed"],
                            config["drone_settings"]["max_cruise_speed"],
                        ),
                        turning_radius=np.random.uniform(
                            config["drone_settings"]["min_turning_radius"],
                            config["drone_settings"]["max_turning_radius"],
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

            return config, uavs

        self.mission["config"], self.mission["uavs"] = \
            update_mission_config(self.mission["config"])

    # -------------------------------------------------------
    # Environment Creation
    # -------------------------------------------------------

    def create_test_environment(self):

        # Deep copy UAVs so every model starts from identical state
        uavs = copy.deepcopy(self.mission["uavs"])

        return MultiUAVEnv(
            uavs,
            tl=self.mission["config"]["env"]["top_left"],
            br=self.mission["config"]["env"]["bottom_right"],
            dt=self.mission["config"]["env"]["dt"],
            mode="manual_mission",
            inference_mode=False
        )

    # -------------------------------------------------------
    # Load RL Models
    # -------------------------------------------------------

    def load_models(self):

        algo_map = {
            "SAC": SAC,
            "A2C": A2C,
            "PPO": PPO
        }

        self.models = {}

        for alg_name, alg_cfg in self.mission["config"]["eval_models"].items():

            if not alg_cfg["model_path"]:
                continue

            model_cls = algo_map.get(alg_name)

            model = model_cls.load(
                alg_cfg["model_path"],
                device=alg_cfg["device"]
            )

            self.models[alg_name] = model

    # -------------------------------------------------------
    # Run Evaluation
    # -------------------------------------------------------

    def run_tests_on_env(self):

        for alg_name, model in self.models.items():

            print(f"\nRunning {alg_name}")

            env = self.create_test_environment()

            env.clear_missions = False

            step_count, reward, _, done, truncated = run_light_episode(
                model,
                env,
                self.mission["config"]["env"]["max_steps"]
            )

            self.metrics[alg_name] = {
                "reward": reward,
                "steps": step_count,
                "simulation": env.get_uav_metrics()
            }

            env.close()

    # -------------------------------------------------------
    # Display Results
    # -------------------------------------------------------

    def show_metrics(self):

        for alg_name, metrics in self.metrics.items():

            print(f"\nMetrics for {alg_name}:")

            for drone in metrics["simulation"]["mission_stats"]:

                print(f"\tDrone {drone['id']}:")
                print(
                    f"\t\tReached {drone['waypoints_reached']}/"
                    f"{self.mission['config']['drone_settings']['num_wps_per_drone']} waypoints"
                )
                print(f"\t\tTraveled {drone['dist_navigating']} m")

            print(f"\tReward: {metrics['reward']}")
            print(f"\tSteps: {metrics['steps']}")

            print(
                f"\tGeofence Violations: "
                f"{metrics['simulation']['safety_violations']['geofence']['total_count']}"
            )

            print(
                f"\tCaution Dist Violations: "
                f"{metrics['simulation']['safety_violations']['caution']['total_count']}"
            )

            print(
                f"\tCritical Dist Violations (Crash): "
                f"{metrics['simulation']['safety_violations']['critical']['total_count']}"
            )


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------

if __name__ == "__main__":

    with open("config/eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    evaluator = Evaluator(config)

    evaluator.run_tests_on_env()

    evaluator.show_metrics()