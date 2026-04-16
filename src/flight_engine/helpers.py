from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Any, Optional, Tuple

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
    waypoint_id: Optional[str] = None

    def to_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)

    def to_waypoint_payload(self) -> dict[str, Any]:
        return {
            "id": self.waypoint_id,
            "lat": float(self.latitude),
            "lon": float(self.longitude),
        }


class FlightMode(Enum):
    NAVIGATING = "NAVIGATING"
    LOITERING = "LOITERING"
    IDLE = "IDLE"
