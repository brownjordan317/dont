from typing import Tuple
import numpy as np

from flight_engine.helpers import wrap_angle

# ============================================================================
# FLIGHT DYNAMICS
# ============================================================================

class FlightDynamics:  
    def __init__(self, turning_radius: float, cruise_speed: float):
        self.turning_radius = turning_radius
        self.cruise_speed = cruise_speed
        self.max_turn_rate = cruise_speed / turning_radius
    
    def compute_turn(self, current_heading: float, target_bearing: float, 
                    dt: float) -> Tuple[float, float]:
        """
        Compute turn amount and new heading
        Returns: (new_heading, turn_amount)
        """
        heading_error = wrap_angle(target_bearing - current_heading)
        max_turn = self.max_turn_rate * dt
        
        if abs(heading_error) < max_turn * 1.5:
            # Small error, align directly
            return target_bearing, heading_error
        else:
            # Gradual turn
            turn_direction = np.sign(heading_error)
            turn_amount = min(abs(heading_error), max_turn)
            new_heading = wrap_angle(current_heading + turn_direction * turn_amount)
            return new_heading, turn_direction * turn_amount
    
    def compute_arc_motion(self, x: float, y: float, heading: float, 
                          turn_amount: float) -> Tuple[float, float]:
        """
        Compute new position after turning
        Returns: (new_x, new_y)
        """
        R = self.turning_radius
        
        # Determine turn center
        if turn_amount > 0:  # Left turn
            center_x = x - R * np.sin(heading)
            center_y = y + R * np.cos(heading)
        else:  # Right turn
            center_x = x + R * np.sin(heading)
            center_y = y - R * np.cos(heading)
        
        # Calculate new position on arc
        current_angle = np.arctan2(y - center_y, x - center_x)
        new_angle = current_angle + turn_amount
        
        new_x = center_x + R * np.cos(new_angle)
        new_y = center_y + R * np.sin(new_angle)
        
        return new_x, new_y
    
    def compute_straight_motion(self, x: float, y: float, heading: float, 
                               distance: float) -> Tuple[float, float]:
        """
        Compute new position after straight flight
        Returns: (new_x, new_y)
        """
        new_x = x + distance * np.cos(heading)
        new_y = y + distance * np.sin(heading)
        return new_x, new_y