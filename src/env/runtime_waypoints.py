from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np

from flight_engine.helpers import FlightMode, Position
from flight_engine.simulator import FixedWingAircraft


class RuntimeWaypointMixin:
    def runtime_agent_snapshot(self, agent: str) -> dict:
        aircraft = self._runtime_snapshot_aircraft(agent)
        idx = self.agent_name_to_index[agent]
        latitude = float(aircraft.position.latitude)
        longitude = float(aircraft.position.longitude)
        local_x, local_y = self.transformer.geo_to_local(latitude, longitude)
        current_waypoint = aircraft.waypoint_manager.current_waypoint
        current_waypoint_payload = (
            self._runtime_waypoint_payload(current_waypoint)
            if current_waypoint is not None
            else None
        )
        queued_waypoints = [
            self._runtime_waypoint_payload(waypoint)
            for waypoint in aircraft.waypoint_manager.waypoint_queue
        ]
        hit_waypoints = [
            self._runtime_waypoint_payload(waypoint)
            for waypoint in aircraft.waypoint_manager.hit_waypoints
        ]
        reached_waypoints, remaining_waypoints, _ = self._mission_waypoint_counts(
            aircraft
        )
        targets_by_id = self._runtime_target_payloads_by_id(
            current_waypoint=current_waypoint,
            queued_waypoints=aircraft.waypoint_manager.waypoint_queue,
            hit_waypoints=aircraft.waypoint_manager.hit_waypoints,
        )
        heading_rad = float(aircraft.heading)
        bearing_deg = float(np.degrees(heading_rad) % 360.0)
        return {
            "agent": agent,
            "active": bool(agent in self.agents),
            "mode": aircraft.flight_mode.value,
            "flight_mode": aircraft.flight_mode.value,
            "position": {
                "lat": latitude,
                "lon": longitude,
            },
            "position_latlon": (latitude, longitude),
            "local_position_m": {
                "x": float(local_x),
                "y": float(local_y),
            },
            "heading_rad": heading_rad,
            "heading_deg": float(np.degrees(heading_rad)),
            "bearing_deg": bearing_deg,
            "current_waypoint": current_waypoint_payload,
            "current_waypoint_latlon": (
                current_waypoint.to_tuple()
                if current_waypoint is not None
                else None
            ),
            "current_target": current_waypoint_payload,
            "current_target_id": (
                current_waypoint.waypoint_id
                if current_waypoint is not None
                else None
            ),
            "queued_waypoints": queued_waypoints,
            "queued_waypoint_latlons": [
                waypoint.to_tuple()
                for waypoint in aircraft.waypoint_manager.waypoint_queue
            ],
            "hit_waypoints": hit_waypoints,
            "completed_waypoints": int(reached_waypoints),
            "remaining_waypoints": int(remaining_waypoints),
            "targets": {
                "current": current_waypoint_payload,
                "queued": list(queued_waypoints),
                "hit": list(hit_waypoints),
                "remaining": (
                    [current_waypoint_payload, *queued_waypoints]
                    if current_waypoint_payload is not None
                    else list(queued_waypoints)
                ),
            },
            "waypoints_by_id": targets_by_id,
            "targets_by_id": dict(targets_by_id),
            "distance_traveled_m": float(aircraft.distance_traveled),
            "actual_turn_rate_rad_s": float(aircraft.actual_turn_rate),
            "desired_turn_rate_rad_s": float(aircraft.desired_turn_rate),
            "skill_actions": {
                "route_follow": float(self._last_reference_action_vector[idx]),
                "avoid": float(self._last_avoidance_action_vector[idx]),
            },
            "current_step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "sim_time_s": float(self.current_step * self.dt),
        }

    def runtime_agent_snapshots(self) -> dict[str, dict]:
        return {
            agent: self.runtime_agent_snapshot(agent)
            for agent in sorted(self.aircraft_by_agent)
        }

    def runtime_target_snapshot(self, agent: str, target_id: str) -> dict:
        targets_by_id = self.runtime_target_snapshots(agent)
        if target_id not in targets_by_id:
            raise KeyError(
                f"Target {target_id!r} is not available for agent {agent!r}."
            )
        return dict(targets_by_id[target_id])

    def runtime_target_snapshots(self, agent: str) -> dict[str, dict]:
        snapshot = self.runtime_agent_snapshot(agent)
        return {
            target_id: dict(target_state)
            for target_id, target_state in snapshot["targets_by_id"].items()
        }

    def append_runtime_waypoints(
        self,
        agent: str,
        waypoints: Any,
    ) -> dict:
        aircraft = self._runtime_waypoint_aircraft(agent)
        positions = self._coerce_runtime_waypoint_list(waypoints)
        if positions:
            aircraft.append_waypoints(positions)
            self._refresh_after_runtime_waypoint_update(agent)
        return self.runtime_agent_snapshot(agent)

    def replace_runtime_waypoint_queue(
        self,
        agent: str,
        waypoints: Any,
        *,
        replace_current: bool = False,
    ) -> dict:
        aircraft = self._runtime_waypoint_aircraft(agent)
        positions = self._coerce_runtime_waypoint_list(waypoints)
        aircraft.replace_waypoint_queue(
            positions,
            replace_current=replace_current,
        )
        self._refresh_after_runtime_waypoint_update(agent)
        return self.runtime_agent_snapshot(agent)

    def _runtime_waypoint_aircraft(self, agent: str) -> FixedWingAircraft:
        if not self.allow_live_waypoint_updates:
            raise RuntimeError(
                "Live waypoint updates are disabled for this environment. "
                "Use the trained-policy runtime path to enable them."
            )
        if agent not in self.aircraft_by_agent or agent not in self.agents:
            raise KeyError(
                f"Agent {agent!r} is not active in the current episode."
            )
        return self.aircraft_by_agent[agent]

    def _runtime_snapshot_aircraft(self, agent: str) -> FixedWingAircraft:
        if not self.allow_live_waypoint_updates:
            raise RuntimeError(
                "Live state snapshots are disabled for this environment. "
                "Use the trained-policy runtime path to enable them."
            )
        if agent not in self.aircraft_by_agent:
            raise KeyError(
                f"Agent {agent!r} is not available in the current runtime session."
            )
        return self.aircraft_by_agent[agent]

    def _coerce_runtime_waypoint_list(self, waypoints: Any) -> List[Position]:
        if waypoints is None:
            raise TypeError("waypoints cannot be None")
        if (
            isinstance(waypoints, Position)
            or isinstance(waypoints, dict)
            or self._is_scalar_waypoint_pair(waypoints)
        ):
            raw_waypoints = [waypoints]
        else:
            raw_waypoints = list(waypoints)
        return [
            self._coerce_runtime_waypoint(waypoint)
            for waypoint in raw_waypoints
        ]

    def _coerce_runtime_waypoint(self, waypoint: Any) -> Position:
        if isinstance(waypoint, Position):
            return Position(
                float(waypoint.latitude),
                float(waypoint.longitude),
                waypoint_id=waypoint.waypoint_id,
            )
        if isinstance(waypoint, dict):
            waypoint_id = waypoint.get("id", waypoint.get("waypoint_id"))
            if "latitude" in waypoint and "longitude" in waypoint:
                return Position(
                    float(waypoint["latitude"]),
                    float(waypoint["longitude"]),
                    waypoint_id=(
                        str(waypoint_id).strip()
                        if waypoint_id is not None and str(waypoint_id).strip()
                        else None
                    ),
                )
            if "lat" in waypoint and "lon" in waypoint:
                return Position(
                    float(waypoint["lat"]),
                    float(waypoint["lon"]),
                    waypoint_id=(
                        str(waypoint_id).strip()
                        if waypoint_id is not None and str(waypoint_id).strip()
                        else None
                    ),
                )
            raise TypeError(
                "Waypoint dicts must provide latitude/longitude or lat/lon "
                "keys. Use id or waypoint_id to preserve a caller-supplied "
                "waypoint id."
            )
        if self._is_scalar_waypoint_pair(waypoint):
            latitude, longitude = waypoint
            return Position(float(latitude), float(longitude))
        raise TypeError(
            "Waypoints must be Position instances, (lat, lon) pairs, or "
            "dicts with latitude/longitude keys."
        )

    def _is_scalar_waypoint_pair(self, waypoint: Any) -> bool:
        if not isinstance(waypoint, (list, tuple, np.ndarray)):
            return False
        if len(waypoint) != 2:
            return False
        return bool(np.isscalar(waypoint[0]) and np.isscalar(waypoint[1]))

    def _runtime_waypoint_payload(self, waypoint: Position) -> dict:
        if waypoint.waypoint_id is None:
            raise ValueError("Runtime waypoint payloads require waypoint ids.")
        return waypoint.to_waypoint_payload()

    def _runtime_target_payloads_by_id(
        self,
        *,
        current_waypoint: Optional[Position],
        queued_waypoints: Iterable[Position],
        hit_waypoints: Iterable[Position],
    ) -> dict[str, dict]:
        targets_by_id: dict[str, dict] = {}

        def register(status: str, waypoint: Position):
            payload = self._runtime_waypoint_payload(waypoint)
            payload["status"] = status
            targets_by_id[payload["id"]] = payload

        if current_waypoint is not None:
            register("current", current_waypoint)
        for waypoint in queued_waypoints:
            register("queued", waypoint)
        for waypoint in hit_waypoints:
            register("hit", waypoint)
        return targets_by_id

    def _reset_runtime_waypoint_tracking(
        self,
        *,
        idx: int,
        aircraft: FixedWingAircraft,
    ) -> None:
        self._clear_waypoint_tracking(idx)
        self._reset_reference_route_progress(idx=idx, aircraft=aircraft)

    def _refresh_after_runtime_waypoint_update(self, agent: str) -> None:
        aircraft = self.aircraft_by_agent[agent]
        idx = self.agent_name_to_index[agent]
        self._reset_runtime_waypoint_tracking(idx=idx, aircraft=aircraft)

        if aircraft.waypoint_manager.has_waypoints():
            self._completion_steps[idx] = None
            aircraft.flight_mode = FlightMode.NAVIGATING
            aircraft.loiter_center = None
        else:
            if self._completion_steps[idx] is None:
                self._completion_steps[idx] = self.current_step
            aircraft._enter_loiter()

        self._sync_local_caches()
        self._refresh_skill_guidance_cache()
        refreshed_max_steps = self._resolve_episode_max_steps({})
        self.max_steps = max(
            int(self.max_steps),
            int(refreshed_max_steps),
            int(self.current_step) + 1,
        )
        self._update_obs_cache(fill_history=True)
