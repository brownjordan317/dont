"""
Many of the features of this class are only used in the just fly a path example
that is meant for testing the physics and visualizer. However, portions of this
class are still used in the main training and testing pipelines, so it is not deleted.

"""


import numpy as np
from typing import List, Tuple, Optional

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
                 turning_radius: float, color: str = 'blue', mission = None):
        self.id_tag = id_tag
        self.initial_position = Position(initial_position.latitude, initial_position.longitude)
        self.initial_heading =  initial_heading
        self.position = Position(initial_position.latitude, initial_position.longitude)
        self.heading = initial_heading
        self.color = color
        
        # Components
        self.dynamics = FlightDynamics(turning_radius, cruise_speed)
        self.waypoint_manager = WaypointManager()
        
        # State tracking
        self.flight_mode = FlightMode.IDLE
        self.loiter_center: Optional[Position] = None
        self.path_history: List[Position] = [Position(initial_position.latitude, initial_position.longitude)]
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
    
    def update(self, dt: float):
        """Update the aircraft state based on the current flight mode"""
        # Create coordinate transformer centered on current position
        transformer = CoordinateTransformer(
            self.position.latitude, 
            self.position.longitude
        )
        
        # Current position in local coordinates (origin)
        x, y = 0.0, 0.0
        
        # Check arrival if navigating
        if self.flight_mode == FlightMode.NAVIGATING and self.waypoint_manager.current_waypoint:
            dist_to_wp = self._calculate_dist(self.position, self.waypoint_manager.current_waypoint)
            if dist_to_wp < self.waypoint_manager.arrival_threshold:
                # print(f"[{self.id_tag}] Reached waypoint at {self.waypoint_manager.current_waypoint}")
                # if no more waypoints, enter loiter
                if not self.waypoint_manager.advance():
                    print(f"[{self.id_tag}] No more waypoints - entering loiter")
                    self.flight_mode = FlightMode.LOITERING
                    self.loiter_center = Position(self.position.latitude, self.position.longitude)
                self.waypoint_manager.current_waypoint = None
        
        # Update based on mode
        if self.flight_mode == FlightMode.LOITERING:
            self._update_loiter(x, y, dt, transformer)
        elif self.waypoint_manager.current_waypoint is not None:
            self._update_navigation(x, y, dt, transformer)
        else:
            # No waypoints - enter loiter
            self._enter_loiter()
    
    def _calculate_dist(self, pos1: Position, pos2: Position) -> float:
        """Calculate Euclidean distance in meters between two geographic positions"""
        transformer = CoordinateTransformer(pos1.latitude, pos1.longitude)
        lx, ly = transformer.geo_to_local(pos2.latitude, pos2.longitude)
        return np.hypot(lx, ly)
    
    def _update_navigation(self, x: float, y: float, dt: float, 
                          transformer: CoordinateTransformer):
        """Update aircraft state when navigating to waypoints"""
        # Get target in local coordinates
        wp = self.waypoint_manager.current_waypoint
        xt, yt = transformer.geo_to_local(wp.latitude, wp.longitude)
        
        # Check arrival
        distance = np.hypot(xt - x, yt - y)
        if self.waypoint_manager.check_arrival(distance):
            if not self.waypoint_manager.advance():
                self._enter_loiter()
                return
            # Recalculate with new waypoint
            wp = self.waypoint_manager.current_waypoint
            xt, yt = transformer.geo_to_local(wp.latitude, wp.longitude)
        
        # Compute navigation
        target_bearing = np.arctan2(yt - y, xt - x)
        new_heading, turn_amount = self.dynamics.compute_turn(
            self.heading, target_bearing, dt
        )
        
        # Update position based on turn or straight
        if abs(turn_amount) > 0.001:  # Turning
            x, y = self.dynamics.compute_arc_motion(
                x, y, self.heading, turn_amount)
        else:  # Straight
            travel_dist = min(self.dynamics.cruise_speed * dt, distance)
            x, y = self.dynamics.compute_straight_motion(
                x, y, new_heading, travel_dist)
        
        self.heading = new_heading
        self._update_position(x, y, transformer)
    
    def _update_loiter(self, x: float, y: float, dt: float, 
                      transformer: CoordinateTransformer):
        """Updates the aircraft state while in loiter mode"""
        turn_amount = self.dynamics.max_turn_rate * dt
        new_heading = wrap_angle(self.heading + turn_amount)
        x, y = self.dynamics.compute_arc_motion(
            x, y, self.heading, turn_amount)
        
        self.heading = new_heading
        self._update_position(x, y, transformer)
    
    def _enter_loiter(self):
        """Enters loiter mode"""
        self.flight_mode = FlightMode.LOITERING
        self.loiter_center = Position(
            self.position.latitude, 
            self.position.longitude
        )
    
    def _update_position(self, x: float, y: float, 
                        transformer: CoordinateTransformer):
        """Update the aircraft position and path history based on local coordinates"""
        lat, lon = transformer.local_to_geo(x, y)
        new_pos = Position(lat, lon)

        if self.flight_mode == FlightMode.NAVIGATING:
            # Convert previous geo position to local
            prev_x, prev_y = transformer.geo_to_local(
                self.position.latitude,
                self.position.longitude
            )
            # Euclidean distance in meters
            distance = np.hypot(x - prev_x, y - prev_y)
            self.distance_traveled += distance
        
        self.position = new_pos
        self.path_history.append(Position(lat, lon))
    
    def get_state(self) -> dict:
        """Returns the current state of the aircraft as a dictionary"""
        return {
            'id': self.id_tag,
            'position': self.position.to_tuple(),
            'heading': np.rad2deg(self.heading),
            'mode': self.flight_mode.value,
            'current_waypoint': self.waypoint_manager.current_waypoint.to_tuple() 
                if self.waypoint_manager.current_waypoint else None,
            'queue_size': self.waypoint_manager.queue_size(),
            'color': self.color,
            'distance_traveled': self.distance_traveled
        }