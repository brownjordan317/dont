from collections import deque
from typing import List, Optional, Deque

from flight_engine.helpers import Position

# ============================================================================
# WAYPOINT MANAGER
# ============================================================================

class WaypointManager:
    """Manages waypoint queue and navigation logic"""
    
    def __init__(self, arrival_threshold: float = 30.0):
        self.waypoint_queue: Deque[Position] = deque()
        self.current_waypoint: Optional[Position] = None
        self.arrival_threshold = arrival_threshold
        self.hit_waypoints: List[Position] = []
    
    def add_waypoint(self, waypoint: Position):
        """Add a waypoint to the queue"""
        self.waypoint_queue.append(waypoint)
        if self.current_waypoint is None:
            self.advance()
    
    def advance(self) -> bool:
        """
        Advance to next waypoint
        Returns: True if waypoint available, False if queue empty
        """
        if self.waypoint_queue:
            self.current_waypoint = self.waypoint_queue.popleft()
            return True
        else:
            self.current_waypoint = None
            return False
    
    def check_arrival(self, distance: float) -> bool:
        """Check if arrived at current waypoint"""
        return distance < self.arrival_threshold
    
    def has_waypoints(self) -> bool:
        """Check if any waypoints remain"""
        return self.current_waypoint is not None or len(self.waypoint_queue) > 0
    
    def queue_size(self) -> int:
        """Get number of queued waypoints (excluding current)"""
        return len(self.waypoint_queue)