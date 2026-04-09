from dataclasses import dataclass

from src.drone_controller.motion import advance_toward_target


@dataclass(frozen=True)
class DroneState:
    e: float
    n: float
    heading: float
    agl: float

    def as_dict(self):
        return {
            "e": self.e,
            "n": self.n,
            "heading": self.heading,
            "agl": self.agl,
        }


class SimulatedFlightController:
    def __init__(
        self,
        initial_state: DroneState,
        drone_speed: float,
        step_seconds: float,
        max_turn_rate_deg: float,
        map_bounds=None,
    ):
        self._state = initial_state
        self._target = None
        self._drone_speed = float(drone_speed)
        self._step_seconds = float(step_seconds)
        self._max_turn_rate_deg = float(max_turn_rate_deg)
        self._map_bounds = map_bounds

    def get_state(self) -> DroneState:
        return self._state

    def set_state(self, state: DroneState) -> None:
        self._state = state

    def get_target(self):
        return self._target

    def set_target(self, target) -> None:
        self._target = target

    def clear_target(self) -> None:
        self._target = None

    def update(self) -> DroneState:
        if self._target is None:
            return self._state

        next_state = advance_toward_target(
            self._state.as_dict(),
            (self._target["target_e"], self._target["target_n"]),
            self._drone_speed,
            self._max_turn_rate_deg,
            map_bounds=self._map_bounds,
            step_seconds=self._step_seconds,
        )
        self._state = DroneState(
            e=next_state["e"],
            n=next_state["n"],
            heading=next_state["heading"],
            agl=next_state.get("agl", self._state.agl),
        )
        return self._state
