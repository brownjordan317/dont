import os
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from rich.console import Console
from rich.panel import Panel

from gym_env import MultiUAVEnv
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position

console = Console()

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

        self.current_phase_idx = 0
        os.makedirs(self.save_dir, exist_ok=True)

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
        env = self.training_env.envs[0].unwrapped
        new_id = f"UAV-{len(env.aircraft_list) + 1}"

        heading = np.random.uniform(-np.pi, np.pi)

        new_uav = FixedWingAircraft(
            new_id,
            Position(self.origin[0], self.origin[1]),
            heading,
            self.config["train"]["drone_speed"],
            self.config["train"]["drone_turn_rate"],
            speed_variance = self.config["train"]["speed_var"], 
            turning_variance = self.config["train"]["speed_var"]
        )

        env.aircraft_list.append(new_uav)

    def _update_drone_count(self):
        """
        Updates the number of drones in the environment to match the current 
        phase of the curriculum.
        """
        _, _, _, target_uavs = self._get_curriculum_phase()
        env = self.training_env.envs[0].unwrapped

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
                f"a2c_phase_{phase_idx}_step_{self.num_timesteps}"
            )
            self.model.save(save_path)

            console.print(
                Panel(
                    f"[bold magenta]MODEL SAVED[/bold magenta]\n"
                    f"Phase: {phase_idx}\n"
                    f"Progress ≥ {phase[0]*100:.0f}%\n"
                    f"Saved to:\n{save_path}",
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
    console.print(
        Panel.fit(
            "[bold white]Multi-UAV A2C Trainer[/bold white]",
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
            config["train"]["drone_speed"],
            config["train"]["drone_turn_rate"],
            speed_variance = config["train"]["speed_var"], 
            turning_variance = config["train"]["speed_var"]
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
    env = MultiUAVEnv(
        initial_uavs, 
        tl=tl, 
        br=br, 
        dt=config["train"]["dt"],
        max_steps=config["train"]["max_steps"],
        boundary_margin=config["train"]["boundary_margin"],
        mission_waypoint_count=config["train"]["mission_waypoint_count"],
        mode='gen_mission', #gen_mission or manual_mission
        caution_dist=config["train"]["caution_dist"],
        critical_dist=config["train"]["critical_dist"]
    ) 

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=float(config["train"]["learning_rate"]),
        max_grad_norm=config["train"]["max_grad_norm"],
        ent_coef=config["train"]["ent_coeff"],
        vf_coef=config["train"]["vf_coeff"],
        policy_kwargs=config["train"]["policy_kwargs"],
        tensorboard_log=config["train"]["tensorboard_log"],
        verbose=config["train"]["verbose"],
        device=config["train"]["device"],
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
        )
        model.save(
            os.path.join(
                config["train"]["save_dir"], 
                config["train"]["model_name"]
            )
        )
    except KeyboardInterrupt:
        model.save(
            os.path.join(
                config["train"]["save_dir"], 
                f"{config['train']['model_name']}_interrupted"
            )
        )

if __name__ == "__main__":
    print()
