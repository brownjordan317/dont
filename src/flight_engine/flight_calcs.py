from typing import Tuple
import numpy as np

class FlightDynamics:
    def __init__(
        self,
        turning_radius: float,
        cruise_speed: float,
        turn_response_time_s: float = 0.0,
    ):
        self.turning_radius = turning_radius
        self.cruise_speed = cruise_speed
        self.max_turn_rate = cruise_speed / turning_radius
        self.turn_response_time_s = max(float(turn_response_time_s), 0.0)

    def resolve_turn_rate(
        self,
        current_turn_rate: float,
        commanded_turn_rate: float,
        dt: float,
    ) -> float:
        commanded_turn_rate = float(
            np.clip(commanded_turn_rate, -self.max_turn_rate, self.max_turn_rate)
        )
        if self.turn_response_time_s <= 1e-6:
            return commanded_turn_rate

        alpha = 1.0 - np.exp(-max(float(dt), 0.0) / self.turn_response_time_s)
        resolved_turn_rate = current_turn_rate + (
            alpha * (commanded_turn_rate - current_turn_rate)
        )
        return float(
            np.clip(resolved_turn_rate, -self.max_turn_rate, self.max_turn_rate)
        )

    def compute_arc_motion(
        self,
        x: float,
        y: float,
        heading: float,
        turn_amount: float,
    ) -> Tuple[float, float]:
        turn_radius = self.turning_radius

        if turn_amount > 0:
            center_x = x - turn_radius * np.sin(heading)
            center_y = y + turn_radius * np.cos(heading)
        else:
            center_x = x + turn_radius * np.sin(heading)
            center_y = y - turn_radius * np.cos(heading)

        current_angle = np.arctan2(y - center_y, x - center_x)
        new_angle = current_angle + turn_amount

        new_x = center_x + turn_radius * np.cos(new_angle)
        new_y = center_y + turn_radius * np.sin(new_angle)

        return new_x, new_y
