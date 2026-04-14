from collections import deque
from typing import List, Optional, Deque

from flight_engine.helpers import Position

class WaypointManager:
    def __init__(self, arrival_threshold: float = 30.0):
        self.waypoint_queue: Deque[Position] = deque()
        self.current_waypoint: Optional[Position] = None
        self.arrival_threshold = arrival_threshold
        self.hit_waypoints: List[Position] = []

    def add_waypoint(self, waypoint: Position):
        self.waypoint_queue.append(waypoint)
        if self.current_waypoint is None:
            self.advance()

    def advance(self) -> bool:
        if self.waypoint_queue:
            self.current_waypoint = self.waypoint_queue.popleft()
            return True
        self.current_waypoint = None
        return False

    def has_waypoints(self) -> bool:
        return self.current_waypoint is not None or len(self.waypoint_queue) > 0

    def queue_size(self) -> int:
        return len(self.waypoint_queue)
