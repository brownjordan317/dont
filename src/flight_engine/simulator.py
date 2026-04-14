import numpy as np
from typing import Iterable, List, Optional

from flight_engine.helpers import wrap_angle, Position, FlightMode
from flight_engine.trans_coorders import CoordinateTransformer
from flight_engine.wp_manager import WaypointManager
from flight_engine.flight_calcs import FlightDynamics

class FixedWingAircraft:
    """Fixed-wing aircraft with Dubins-like path following"""

    def __init__(
        self,
        id_tag: str,
        initial_position: Position,
        initial_heading: float,
        cruise_speed: float,
        turning_radius: float,
        mission=None,
        turn_response_time_s: float = 0.0,
    ):
        self.id_tag = id_tag
        self.position = Position(initial_position.latitude, initial_position.longitude)
        self.initial_pos = self.position
        self.heading = initial_heading
        self.initial_heading = initial_heading
        self.base_turning_radius = turning_radius
        self.base_cruise_speed = cruise_speed

        self.dynamics = FlightDynamics(
            turning_radius,
            cruise_speed,
            turn_response_time_s=turn_response_time_s,
        )
        self.waypoint_manager = WaypointManager()
        self.flight_mode = FlightMode.IDLE
        self.loiter_center: Optional[Position] = None
        self.path_history: List[Position] = [
            Position(
                initial_position.latitude,
                initial_position.longitude
            )
        ]
        self.distance_traveled = 0.0
        self.actual_turn_rate = 0.0
        self.desired_turn_rate = 0.0

        if mission:
            self.add_waypoints(mission)

    def add_wp(self, waypoint: Position):
        self.waypoint_manager.add_waypoint(waypoint)
        if self.flight_mode in (FlightMode.IDLE, FlightMode.LOITERING):
            self.flight_mode = FlightMode.NAVIGATING
            if self.waypoint_manager.current_waypoint is None:
                self.waypoint_manager.advance()

    def add_waypoints(self, waypoints: List[Optional[Position]]):
        for wp in waypoints:
            if not isinstance(wp, Position):
                wp = Position(*wp)
            self.add_wp(wp)

    def append_waypoints(self, waypoints: Iterable[Position]):
        for waypoint in waypoints:
            self.add_wp(waypoint)

    def replace_waypoint_queue(
        self,
        waypoints: Iterable[Position],
        *,
        replace_current: bool = False,
    ):
        self.waypoint_manager.replace_queue(
            waypoints,
            replace_current=replace_current,
        )
        if self.waypoint_manager.has_waypoints():
            self.flight_mode = FlightMode.NAVIGATING
            self.loiter_center = None
        else:
            self._enter_loiter()

    def update_simple(
        self,
        turn_rate: float,
        dt: float,
        transformer: CoordinateTransformer,
    ):
        resolved_turn_rate = self.dynamics.resolve_turn_rate(
            current_turn_rate=self.actual_turn_rate,
            commanded_turn_rate=turn_rate,
            dt=dt,
        )

        arc_length = self.dynamics.cruise_speed * dt
        self.distance_traveled += arc_length

        if abs(resolved_turn_rate) < 1e-6:
            dist_straight = arc_length
            dx = dist_straight * np.sin(self.heading)
            dy = dist_straight * np.cos(self.heading)
        else:
            dist_straight = (
                2
                * self.dynamics.cruise_speed
                / abs(resolved_turn_rate)
                * np.sin(abs(resolved_turn_rate) * dt / 2.0)
            )
            avg_heading = self.heading + (resolved_turn_rate * dt / 2.0)
            dx = dist_straight * np.sin(avg_heading)
            dy = dist_straight * np.cos(avg_heading)

        self.actual_turn_rate = resolved_turn_rate
        self.heading = wrap_angle(self.heading + (resolved_turn_rate * dt))

        curr_x, curr_y = transformer.geo_to_local(
            self.position.latitude,
            self.position.longitude
        )
        new_lat, new_lon = transformer.local_to_geo(curr_x + dx, curr_y + dy)
        self.position = Position(new_lat, new_lon)

        return dx, dy

    def _update_loiter(
        self,
        x: float,
        y: float,
        dt: float,
        transformer: CoordinateTransformer,
    ):
        turn_amount = self.dynamics.max_turn_rate * dt
        new_heading = wrap_angle(self.heading + turn_amount)
        x, y = self.dynamics.compute_arc_motion(
            x, y, self.heading, turn_amount
        )

        self.heading = new_heading
        lat, lon = transformer.local_to_geo(x, y)
        new_pos = Position(lat, lon)

        self.position = new_pos
        self.path_history.append(Position(lat, lon))

    def _enter_loiter(self):
        self.flight_mode = FlightMode.LOITERING
        self.loiter_center = Position(
            self.position.latitude,
            self.position.longitude
        )
