from collections import deque
from typing import Deque, Iterable, List, Optional

from flight_engine.helpers import Position


class WaypointManager:
    def __init__(
        self,
        arrival_threshold: float = 30.0,
        waypoint_id_prefix: str = "wp",
    ):
        self.waypoint_queue: Deque[Position] = deque()
        self.current_waypoint: Optional[Position] = None
        self.arrival_threshold = arrival_threshold
        self.hit_waypoints: List[Position] = []
        self._waypoint_id_prefix = waypoint_id_prefix
        self._next_waypoint_seq = 1
        self._issued_waypoint_ids: set[str] = set()

    def add_waypoint(self, waypoint: Position):
        self.waypoint_queue.append(self._normalize_waypoint(waypoint))
        if self.current_waypoint is None:
            self.advance()

    def append_waypoints(self, waypoints: Iterable[Position]) -> int:
        normalized_waypoints = [
            self._normalize_waypoint(waypoint)
            for waypoint in waypoints
        ]
        appended = 0
        for waypoint in normalized_waypoints:
            self.waypoint_queue.append(waypoint)
            appended += 1
        if self.current_waypoint is None:
            self.advance()
        return appended

    def replace_queue(
        self,
        waypoints: Iterable[Position],
        *,
        replace_current: bool = False,
    ) -> int:
        self.waypoint_queue = deque(
            self._normalize_waypoint(waypoint)
            for waypoint in waypoints
        )
        if replace_current:
            self.current_waypoint = None
        if self.current_waypoint is None:
            self.advance()
        return self.queue_size() + (1 if self.current_waypoint is not None else 0)

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

    def remaining_waypoints(self) -> List[Position]:
        remaining: List[Position] = []
        if self.current_waypoint is not None:
            remaining.append(self.current_waypoint)
        remaining.extend(self.waypoint_queue)
        return remaining

    def _normalize_waypoint(self, waypoint: Position) -> Position:
        waypoint_id = waypoint.waypoint_id
        if waypoint_id is None or not str(waypoint_id).strip():
            waypoint_id = self._next_generated_waypoint_id()
        else:
            waypoint_id = str(waypoint_id).strip()
            if waypoint_id in self._issued_waypoint_ids:
                raise ValueError(
                    f"Duplicate waypoint id {waypoint_id!r} is not allowed "
                    "for the same drone."
                )
            self._issued_waypoint_ids.add(waypoint_id)

        return Position(
            float(waypoint.latitude),
            float(waypoint.longitude),
            waypoint_id=waypoint_id,
        )

    def _next_generated_waypoint_id(self) -> str:
        while True:
            waypoint_id = (
                f"{self._waypoint_id_prefix}-{self._next_waypoint_seq}"
            )
            self._next_waypoint_seq += 1
            if waypoint_id not in self._issued_waypoint_ids:
                self._issued_waypoint_ids.add(waypoint_id)
                return waypoint_id
