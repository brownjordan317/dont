import numpy as np
from typing import List, Optional, Tuple

from flight_engine.helpers import wrap_angle, Position, FlightMode
from flight_engine.trans_coorders import CoordinateTransformer
from flight_engine.wp_manager import WaypointManager
from flight_engine.flight_calcs import FlightDynamics

# ============================================================================
# FIXED WING AIRCRAFT
# ============================================================================

class FixedWingAircraft:
    """Fixed-wing aircraft with Dubins-like path following"""
    
    def __init__(self, id_tag: str, initial_position: Position, 
                 initial_heading: float, cruise_speed: float, 
                 turning_radius: float, color: str = 'blue', mission = None,
                 speed_variance = 0, turning_variance = 0):
        self.id_tag = id_tag
        self.position = Position(initial_position.latitude, initial_position.longitude)
        self.initial_pos = self.position
        self.heading = initial_heading
        self.color = color
        self.speed_variance = speed_variance
        self.turning_variance = turning_variance
        self.turning_radius = turning_radius
        self.cruise_speed = cruise_speed

        # Components
        self.dynamics = FlightDynamics(turning_radius, cruise_speed)
        self.waypoint_manager = WaypointManager()
        
        # State tracking
        self.flight_mode = FlightMode.IDLE
        self.loiter_center: Optional[Position] = None
        self.path_history: List[Position] = [
            Position(
                initial_position.latitude, 
                initial_position.longitude
            )
        ]
        self.distance_traveled = 0.0

        # Add initial waypoints
        if mission:
            self.add_waypoints(mission)
    
    def add_wp(self, waypoint: Position):
        self.waypoint_manager.add_waypoint(waypoint)
        if self.flight_mode in (FlightMode.IDLE, FlightMode.LOITERING):
            self.flight_mode = FlightMode.NAVIGATING
            if self.waypoint_manager.current_waypoint is None:
                self.waypoint_manager.advance()
    
    def add_waypoints(self, waypoints: List[Optional[Position]]):
        """Add multiple waypoints to the queue"""
        for wp in waypoints:
            if not isinstance(wp, Position):
                wp = Position(*wp)
            self.add_wp(wp)
    
    def update_simple(self, turn_rate: float, dt: float, transformer: CoordinateTransformer):
        # 1. Exact Arc Length (The "Odometer" distance)
        arc_length = self.dynamics.cruise_speed * dt
        self.distance_traveled += arc_length

        # 2. Calculate Displacement
        # If turn_rate is near zero, use the straight-line distance to avoid div by zero
        if abs(turn_rate) < 1e-6:
            dist_straight = arc_length
            # Displacement uses current heading
            dx = dist_straight * np.sin(self.heading)
            dy = dist_straight * np.cos(self.heading)
        else:
            # The chord length is the straight line between start and end of the arc
            dist_straight = (2 * self.dynamics.cruise_speed / abs(turn_rate)) * \
                            np.sin(abs(turn_rate) * dt / 2.0)
            
            # The effective heading for the displacement is the average 
            # of the start heading and the end heading
            avg_heading = self.heading + (turn_rate * dt / 2.0)
            dx = dist_straight * np.sin(avg_heading)
            dy = dist_straight * np.cos(avg_heading)

        # 3. Update the state
        self.heading = wrap_angle(self.heading + (turn_rate * dt))
        
        curr_x, curr_y = transformer.geo_to_local(
            self.position.latitude, 
            self.position.longitude
        )
        new_lat, new_lon = transformer.local_to_geo(curr_x + dx, curr_y + dy)
        self.position = Position(new_lat, new_lon)
        
        return dx, dy
    
    def _update_loiter(self, x: float, y: float, dt: float, 
                      transformer: CoordinateTransformer):
        """Updates the aircraft state while in loiter mode"""
        turn_amount = self.dynamics.max_turn_rate * dt
        new_heading = wrap_angle(self.heading + turn_amount)
        x, y = self.dynamics.compute_arc_motion(
            x, y, self.heading, turn_amount)
        
        self.heading = new_heading
        lat, lon = transformer.local_to_geo(x, y)
        new_pos = Position(lat, lon)
        
        self.position = new_pos
        self.path_history.append(Position(lat, lon))
    
    def _enter_loiter(self):
        """Enters loiter mode"""
        self.flight_mode = FlightMode.LOITERING
        self.loiter_center = Position(
            self.position.latitude, 
            self.position.longitude
        )