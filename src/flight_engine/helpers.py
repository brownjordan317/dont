from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Tuple

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    """Geographic position"""
    latitude: float
    longitude: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)
    
    @classmethod
    def from_tuple(cls, pos: Tuple[float, float]) -> 'Position':
        return cls(latitude=pos[0], longitude=pos[1])


@dataclass
class FlightState:
    """Current flight state"""
    position: Position
    heading: float  # radians
    speed: float  # m/s
    
    
class FlightMode(Enum):
    """Flight mode enumeration"""
    NAVIGATING = "NAVIGATING"
    LOITERING = "LOITERING"
    IDLE = "IDLE"