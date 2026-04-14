from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Tuple

def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def clip_scalar(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value

@dataclass
class Position:
    latitude: float
    longitude: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)


class FlightMode(Enum):
    NAVIGATING = "NAVIGATING"
    LOITERING = "LOITERING"
    IDLE = "IDLE"
