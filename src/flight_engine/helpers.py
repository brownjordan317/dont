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


def clip_scalar(value: float, lower: float, upper: float) -> float:
    """Fast scalar clamp without numpy dispatch overhead."""
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


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


class FlightMode(Enum):
    """Flight mode enumeration"""
    NAVIGATING = "NAVIGATING"
    LOITERING = "LOITERING"
    IDLE = "IDLE"
