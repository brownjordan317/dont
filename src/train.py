import os
import numpy as np
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from rich.console import Console
from rich.panel import Panel

from gym_env import MultiUAVEnv
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position
from config_utils import get_tuning_section
from vec_env_utils import make_training_env, save_vecnormalize_stats, unwrap_env

console = Console()

# ============================================================
# ALGORITHM REGISTRY
# Hyperparameter keys each algorithm reads from config.
# Only SAC has a replay buffer, so buffer_clear only runs for it.
# ============================================================
ALGO_REGISTRY = {
    "SAC": {
        "cls": SAC,
        "has_replay_buffer": True,
        "kwargs": lambda c: dict(
            learning_rate=float(c["learning_rate"]),
            buffer_size=int(c["buffer_size"]),
            learning_starts=int(c["learning_starts"]),
            batch_size=int(c["batch_size"]),
            tau=float(c["tau"]),
            gamma=float(c["gamma"]),
            ent_coef=c["ent_coef"],
            target_update_interval=int(c["target_update_interval"]),
            train_freq=(int(c["train_freq"]), "step"),
            gradient_steps=int(c["gradient_steps"]),
            policy_kwargs=c["policy_kwargs"],
            tensorboard_log=c["tensorboard_log"],
            verbose=c["verbose"],
            device=c["device"],
        ),
    },
    "PPO": {
        "cls": PPO,
        "has_replay_buffer": False,
        "kwargs": lambda c: dict(
            learning_rate=float(c["learning_rate"]),
            n_steps=int(c["n_steps"]),
            batch_size=int(c["batch_size"]),
            n_epochs=int(c["n_epochs"]),
            gamma=float(c["gamma"]),
            gae_lambda=float(c["gae_lambda"]),
            clip_range=float(c["clip_range"]),
            ent_coef=float(c["ent_coef"]),
            vf_coef=float(c["vf_coef"]),
            max_grad_norm=float(c["max_grad_norm"]),
            policy_kwargs=c["policy_kwargs"],
            tensorboard_log=c["tensorboard_log"],
            verbose=c["verbose"],
            device=c["device"],
        ),
    },
    "A2C": {
        "cls": A2C,
        "has_replay_buffer": False,
        "kwargs": lambda c: dict(
            learning_rate=float(c["learning_rate"]),
            n_steps=int(c["n_steps"]),
            gamma=float(c["gamma"]),
            gae_lambda=float(c["gae_lambda"]),
            ent_coef=float(c["ent_coef"]),
            vf_coef=float(c["vf_coef"]),
            max_grad_norm=float(c["max_grad_norm"]),
            policy_kwargs=c["policy_kwargs"],
            tensorboard_log=c["tensorboard_log"],
            verbose=c["verbose"],
            device=c["device"],
        ),
    },
}


def save_model_artifacts(model, save_path: str):
    """Save the model and any VecNormalize statistics alongside it."""
    model.save(save_path)
    return save_vecnormalize_stats(model, save_path)

class RobustCurriculumCallback(BaseCallback):
    def __init__(
            self, 
            origin,
            config, 
        ):
        super().__init__()
        self.origin = origin
        self.config = config    
        self.change_freq = config["train"]["change_frequency"]
        self.total_timesteps = config["train"]["total_timesteps"]
        self.save_dir = config["train"]["save_dir"]
        self.curriculum = self.set_curriculum(config)
        self.flight_cfg = get_tuning_section(config, "train")["flight"]

        self.current_phase_idx = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_base_env(self):
        """Access the underlying single-env instance through VecEnv wrappers."""
        return unwrap_env(self.training_env)

    def set_curriculum(self, config):
        """
        Sets up the curriculum for training.

        Args:
            config (dict): The configuration dictionary

        Returns:
            list: A sorted list of tuples containing the phase, minimum box 
            size, maximum box size, and number of drones for each phase.
        """
        c = []

        for phase, args in config["train"]["curriculum"].items():
            c.append((
                float(phase),
                float(args["min_box_size"]),
                float(args["max_box_size"]),
                int(args["num_drones"]),
            ))

        c.sort(key=lambda x: x[0])
        return c


    def _get_curriculum_phase_idx(self):
        """
        Calculates the current phase index based on the progress of the 
        training.

        Args:
            None

        Returns:
            int: The current phase index
        """
        progress = self.num_timesteps / self.total_timesteps
        idx = 0
        for i, phase in enumerate(self.curriculum):
            if progress >= phase[0]:
                idx = i
        return idx

    def _get_curriculum_phase(self):
        """
        Returns the current phase of the curriculum based on the progress of 
        the training.

        Returns:
            tuple: The current phase containing the phase, minimum box size, 
            maximum box size, and number of drones.
        """
        return self.curriculum[self._get_curriculum_phase_idx()]

    def _generate_new_box(self):
        """
        Generates a new bounding box for the environment based on the current 
        phase.

        The bounding box is a rectangle with its center at the origin and its 
        size randomly sampled from the minimum and maximum box sizes for the 
        current phase.

        Returns:
            tuple: A tuple containing the top-left and bottom-right 
                coordinates of the bounding box in latitude-longitude format, 
                and the width and height of the box in meters.
        """
        _, min_m, max_m, _ = self._get_curriculum_phase()

        w_m, h_m = np.random.uniform(min_m, max_m, size=2)

        lat_off = (h_m / 2.0) / 111_320.0
        lon_off = (w_m / 2.0) / (
            111_320.0 * np.cos(
            np.radians(
                self.origin[0]
                )
            )
        )

        tl = (
            self.origin[0] + lat_off, 
            self.origin[1] - lon_off
        )
        br = (
            self.origin[0] - lat_off, 
            self.origin[1] + lon_off
        )

        return tl, br, w_m, h_m

    def _add_drone(self):
        """
        Adds a new drone to the environment at the origin with a random 
        heading.
        """
        env = self._get_base_env()
        new_id = f"UAV-{len(env.aircraft_list) + 1}"

        heading = np.random.uniform(-np.pi, np.pi)

        new_uav = FixedWingAircraft(
            new_id,
            Position(self.origin[0], self.origin[1]),
            heading,
            self.flight_cfg["cruise_speed_mps"],
            self.flight_cfg["turning_radius_m"],
            speed_variance=self.flight_cfg["cruise_speed_variation_mps"],
            turning_variance=self.flight_cfg["turning_radius_variation_m"],
        )

        env.aircraft_list.append(new_uav)

    def _update_drone_count(self):
        """
        Updates the number of drones in the environment to match the current 
        phase of the curriculum.
        """
        _, _, _, target_uavs = self._get_curriculum_phase()
        env = self._get_base_env()

        while len(env.aircraft_list) < target_uavs:
            self._add_drone()
            console.print(
                Panel(
                    f"[bold cyan]UAV ADDED[/bold cyan]\n"
                    f"Total UAVs: {len(env.aircraft_list)}",
                    expand=False,
                )
            )

    def _on_step(self) -> bool:
        """
        Called every step to update the environment based on the curriculum.

        Checks if the current phase index is greater than the saved phase 
        index. If so, updates the saved phase index and saves the model at 
        the current step.

        Also updates the environment by generating a new bounding box and 
        adding/removing drones based on the current phase of the curriculum.

        Returns:
            bool: Always returns True.
        """
        phase_idx = self._get_curriculum_phase_idx()

        if phase_idx > self.current_phase_idx:
            self.current_phase_idx = phase_idx
            phase = self.curriculum[phase_idx]

            save_path = os.path.join(
                self.save_dir, 
                f"phase_{phase_idx}_step_{self.num_timesteps}"
            )
            stats_path = save_model_artifacts(self.model, save_path)

            console.print(
                Panel(
                    f"[bold magenta]MODEL SAVED[/bold magenta]\n"
                    f"Phase: {phase_idx}\n"
                    f"Progress ≥ {phase[0]*100:.0f}%\n"
                    f"Saved to:\n{save_path}"
                    + (
                        f"\nNormalization stats:\n{stats_path}"
                        if stats_path else ""
                    ),
                    expand=False,
                )
            )

        if self.num_timesteps % self.change_freq == 0:
            tl, br, w, h = self._generate_new_box()
            self.training_env.env_method("update_bounds", tl, br)

            _, _, _, uavs = self._get_curriculum_phase()

            console.print(
                Panel(
                    f"[bold green]Curriculum Update[/bold green]\n"
                    f"Step: {self.num_timesteps}\n"
                    f"Area: {w:.0f}m × {h:.0f}m\n"
                    f"UAVs: {uavs}",
                    expand=False,
                )
            )

        self._update_drone_count()
        return True

def train(config):
    tuning = get_tuning_section(config, "train")
    env_cfg = tuning["env"]
    flight_cfg = tuning["flight"]
    reward_cfg = tuning["rewards"]
    hard_safety_cfg = tuning["hard_safety"]
    anti_circling_cfg = tuning["anti_circling"]
    algo_name = config["train"].get("algorithm", "SAC").upper()

    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Choose from: {list(ALGO_REGISTRY.keys())}")

    algo_info = ALGO_REGISTRY[algo_name]

    console.print(
        Panel.fit(
            f"[bold white]Multi-UAV {config['train']['algorithm']} Trainer[/bold white]",
            subtitle="Percent-Based Curriculum Learning",
        )
    )

    # Random global origin
    origin = (
        np.random.uniform(-70.0, 70.0),
        np.random.uniform(-170.0, 170.0),
    )

    # Initial UAV
    initial_heading = np.random.uniform(-np.pi, np.pi)
    initial_uavs = [
        FixedWingAircraft(
            "UAV-1",
            Position(origin[0], origin[1]),
            initial_heading,
            flight_cfg["cruise_speed_mps"],
            flight_cfg["turning_radius_m"],
            speed_variance=flight_cfg["cruise_speed_variation_mps"],
            turning_variance=flight_cfg["turning_radius_variation_m"],
        )
    ]

    # Initial small training box
    w_m = h_m = config["train"]["curriculum"][0.0]["min_box_size"]
    
    lat_off = (h_m / 2.0) / 111_320.0
    lon_off = (w_m / 2.0) / (111_320.0 * np.cos(np.radians(origin[0])))

    tl = (origin[0] + lat_off, origin[1] - lon_off)
    br = (origin[0] - lat_off, origin[1] + lon_off)

    # dt: Time step size for physics simulation 
    # (smaller values yield more realistic simulation)
    def make_env():
        return MultiUAVEnv(
            initial_uavs,
            tl=tl,
            br=br,
            dt=env_cfg["dt"],
            max_steps=env_cfg["max_steps"],
            boundary_margin=env_cfg["boundary_margin"],
            mission_waypoint_count=env_cfg["mission_waypoint_count"],
            mode=env_cfg["mode"],
            caution_dist=env_cfg["caution_dist"],
            critical_dist=env_cfg["critical_dist"],
            reward_config=reward_cfg,
            hard_safety_config=hard_safety_cfg,
            anti_circling_config=anti_circling_cfg,
        )

    env = make_training_env(make_env, config["train"])

    model = algo_info["cls"](
        "MlpPolicy",
        env,
        **algo_info["kwargs"](config["train"])
    )

    callback = RobustCurriculumCallback(
        origin=origin,
        config=config,
    )

    try:
        model.learn(
            total_timesteps=config["train"]["total_timesteps"],
            callback=callback,
            progress_bar=True,
            tb_log_name=config["train"]["model_name"],
            log_interval=100
        )
        save_model_artifacts(
            model,
            os.path.join(
                config["train"]["save_dir"], 
                config["train"]["model_name"]
            )
        )
    except KeyboardInterrupt:
        save_model_artifacts(
            model,
            os.path.join(
                config["train"]["save_dir"], 
                f"{config['train']['model_name']}_interrupted"
            )
        )

if __name__ == "__main__":
    print()
