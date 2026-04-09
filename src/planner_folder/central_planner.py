import numpy as np

from src.drone_controller.flight_controller import DroneState
from src.drone_controller.motion import (
    advance_toward_target,
    apply_state_transition_sweep,
    boundary_turn_heading,
    normalize_heading_deg,
    speed_to_step_distance,
    turn_rate_to_step_delta,
)
from src.planner_folder.geometry import build_camera, projection_roi_mask, projection_span_meters
from src.planner_folder.planner import (
    build_empty_reservation_heatmap,
    build_target_fields,
    choose_best_cluster_target,
    choose_camera_projection_target,
    choose_greedy_route_subtarget,
    cluster_has_remaining_heat,
    collect_cluster_target_candidates,
    combine_claim_heatmaps,
    estimate_projection_hit_value,
    merge_reservation_heatmap,
    point_is_in_bounds,
    resolve_navigation_bounds,
    sample_heat_value_at_target,
    select_joint_target_assignment,
    simulate_motion_prefixes,
)


DEFAULT_NAVIGATION_POSITION_TOLERANCE_M = 1e-4


def estimate_target_timeout_frames(
    start_state: DroneState,
    target,
    drone_speed: float,
    step_seconds: float,
):
    goal_e = float(target.get("main_target_e", target["target_e"]))
    goal_n = float(target.get("main_target_n", target["target_n"]))
    dx = goal_e - start_state.e
    dy = goal_n - start_state.n
    if "main_target_e" in target or "main_target_n" in target:
        travel_distance = float(np.hypot(dx, dy))
    else:
        travel_distance = float(target.get("effective_distance", np.hypot(dx, dy)))
    expected_frames = travel_distance / max(
        speed_to_step_distance(drone_speed, step_seconds),
        1e-6,
    )
    return max(18, int(np.ceil(expected_frames * 3.0)) + 6)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class CentralPlanner:
    def __init__(self, config, hmu, controllers, height, origin, *, drone_ids=None):
        self.config = config
        self.hmu = hmu
        self.controllers = list(controllers)
        self.height = height
        self.origin = origin
        self.width_m = float(hmu.data.shape[1]) * config.heatmap.resolution
        self.height_m = float(height) * config.heatmap.resolution
        self.map_bounds = (0.0, self.width_m, 0.0, self.height_m)
        self.navigation_bounds = resolve_navigation_bounds(
            self.width_m,
            self.height_m,
            config.planner,
        )
        self.finished = False
        self.searchers = []
        self.drone_ids = list(
            drone_ids
            if drone_ids is not None
            else (f"drone_{idx}" for idx in range(len(self.controllers)))
        )

        camera = config.camera

        for drone_id, controller in zip(self.drone_ids, self.controllers):
            state = self._clamp_state_to_map(controller.get_state())
            controller.set_state(state)
            projection = build_camera(camera, (state.e, state.n), state.heading, agl=state.agl).project()

            self.searchers.append(
                {
                    "drone_id": drone_id,
                    "controller": controller,
                    "current_target": None,
                    "camera_target": None,
                    "camera_pitch_deg": clamp(camera.pitch, camera.min_pitch, camera.max_pitch),
                    "camera_yaw_deg": clamp(camera.yaw, camera.min_yaw, camera.max_yaw),
                    "reserved_path": None,
                    "reserved_claim_heatmap": None,
                    "last_state": state,
                    "last_projection": projection,
                    "last_target_distance": None,
                    "best_target_distance": None,
                    "stalled_frames": 0,
                    "frames_on_target": 0,
                    "target_timeout_frames": 0,
                    "cooldown_cluster": None,
                    "cooldown_frames": 0,
                }
            )

        self._replan_targets()
        self._refresh_finished_state()

    def _clamp_point_to_map(self, easting, northing):
        return (
            clamp(float(easting), 0.0, self.width_m),
            clamp(float(northing), 0.0, self.height_m),
        )

    def _clamp_state_to_map(self, state: DroneState):
        easting, northing = self._clamp_point_to_map(state.e, state.n)
        if np.isclose(easting, state.e) and np.isclose(northing, state.n):
            return state
        return DroneState(easting, northing, state.heading, state.agl)

    def _refresh_finished_state(self):
        self.finished = self.hmu.total_heat <= 1e-12

    def _point_is_in_navigation_bounds(self, easting, northing):
        min_e, max_e, min_n, max_n = self.navigation_bounds
        tolerance_m = float(
            getattr(
                self.config.planner,
                "navigation_tolerance_m",
                DEFAULT_NAVIGATION_POSITION_TOLERANCE_M,
            )
        )
        return (
            float(min_e) - tolerance_m <= float(easting) <= float(max_e) + tolerance_m
            and float(min_n) - tolerance_m <= float(northing) <= float(max_n) + tolerance_m
        )

    def _snap_state_to_navigation_bounds(self, state: DroneState):
        min_e, max_e, min_n, max_n = self.navigation_bounds
        tolerance_m = float(
            getattr(
                self.config.planner,
                "navigation_tolerance_m",
                DEFAULT_NAVIGATION_POSITION_TOLERANCE_M,
            )
        )
        easting = float(state.e)
        northing = float(state.n)
        if (
            easting < float(min_e) - tolerance_m
            or easting > float(max_e) + tolerance_m
            or northing < float(min_n) - tolerance_m
            or northing > float(max_n) + tolerance_m
        ):
            return state

        snapped_e = clamp(easting, float(min_e), float(max_e))
        snapped_n = clamp(northing, float(min_n), float(max_n))
        if abs(snapped_e - easting) <= 1e-12 and abs(snapped_n - northing) <= 1e-12:
            return state

        return DroneState(snapped_e, snapped_n, state.heading, state.agl)

    def _normalize_controller_state(self, controller):
        state = self._clamp_state_to_map(controller.get_state())
        state = self._snap_state_to_navigation_bounds(state)
        controller.set_state(state)
        return state

    def _build_recovery_target(self, state: DroneState):
        if self._point_is_in_navigation_bounds(state.e, state.n):
            return None

        min_e, max_e, min_n, max_n = self.navigation_bounds
        recovery_e = clamp(float(state.e), float(min_e), float(max_e))
        recovery_n = clamp(float(state.n), float(min_n), float(max_n))

        return {
            **build_target_fields(
                recovery_e,
                recovery_n,
                self.config.heatmap.resolution,
                self.height,
            ),
            "cluster_bounds": (-1, -1, -1, -1),
            "recovery_target": True,
            "main_target_e": recovery_e,
            "main_target_n": recovery_n,
            "main_target_c": recovery_e / self.config.heatmap.resolution,
            "main_target_r": float(self.height) - (recovery_n / self.config.heatmap.resolution),
            "greedy_subtarget": False,
        }

    def _assign_recovery_target(self, searcher, state: DroneState):
        recovery_target = self._build_recovery_target(state)
        if recovery_target is None:
            return False

        controller = searcher["controller"]
        controller.set_target(recovery_target)
        searcher["current_target"] = recovery_target
        searcher["camera_target"] = None
        searcher["reserved_path"] = None
        searcher["reserved_claim_heatmap"] = None
        searcher["last_target_distance"] = None
        searcher["best_target_distance"] = None
        searcher["stalled_frames"] = 0
        searcher["frames_on_target"] = 0
        searcher["target_timeout_frames"] = estimate_target_timeout_frames(
            state,
            recovery_target,
            self.config.planner.drone_speed,
            self.config.planner.step_seconds,
        )
        return True

    def _clear_target(self, searcher, *, cooldown=False):
        current_target = searcher["current_target"]

        if cooldown and current_target is not None:
            searcher["cooldown_cluster"] = current_target["cluster_bounds"]
            searcher["cooldown_frames"] = 10

        controller = searcher["controller"]
        controller.clear_target()

        searcher["current_target"] = None
        searcher["camera_target"] = None
        searcher["reserved_path"] = None
        searcher["reserved_claim_heatmap"] = None
        searcher["last_target_distance"] = None
        searcher["best_target_distance"] = None
        searcher["stalled_frames"] = 0
        searcher["frames_on_target"] = 0
        searcher["target_timeout_frames"] = 0

    def _move_camera_axis(self, current_value, target_value, min_value, max_value, turn_rate):
        if turn_rate <= 0:
            return clamp(current_value, min_value, max_value)

        delta = clamp(target_value - current_value, -turn_rate, turn_rate)
        return clamp(current_value + delta, min_value, max_value)

    def _point_is_in_map(self, easting, northing):
        return (
            0.0 <= float(easting) <= self.width_m
            and 0.0 <= float(northing) <= self.height_m
        )

    def _evaluate_camera_pose(self, state, pitch_deg, yaw_deg):
        try:
            camera_model = build_camera(
                self.config.camera,
                (state.e, state.n),
                state.heading,
                pitch=pitch_deg,
                yaw=yaw_deg,
                agl=state.agl,
            )
            projection = camera_model.project()
            projection_center = camera_model.project_center()
        except ValueError:
            return None

        if projection_roi_mask(
            projection,
            self.hmu.origin,
            self.hmu.resolution,
            self.hmu.data.shape,
        ) is None:
            return None

        footprint_value, footprint_mean = estimate_projection_hit_value(
            self.hmu,
            projection,
        )
        target_c = float(np.clip(projection_center[0] / self.hmu.resolution, 0.0, self.hmu.data.shape[1] - 1e-6))
        target_r = float(np.clip(self.height - (projection_center[1] / self.hmu.resolution), 0.0, self.hmu.data.shape[0] - 1e-6))
        center_value = sample_heat_value_at_target(self.hmu, target_c, target_r)
        return projection_center, footprint_value, footprint_mean, center_value

    @staticmethod
    def _target_has_subgoal(target):
        return (
            "main_target_e" in target
            and "main_target_n" in target
            and (
                abs(float(target["target_e"]) - float(target["main_target_e"])) > 1e-6
                or abs(float(target["target_n"]) - float(target["main_target_n"])) > 1e-6
            )
        )

    def _promote_target_to_main(self, searcher, state):
        target = searcher["current_target"]
        if target is None or not self._target_has_subgoal(target):
            return

        target["target_e"] = float(target["main_target_e"])
        target["target_n"] = float(target["main_target_n"])
        target["target_c"] = float(target["main_target_c"])
        target["target_r"] = float(target["main_target_r"])
        target["greedy_subtarget"] = False
        target.pop("greedy_prefix_steps", None)
        target.pop("greedy_progress_ratio", None)
        target.pop("boundary_turn_target", None)
        target.pop("edge_approach_target", None)
        target.pop("edge_margin_m", None)
        target.pop("main_target_edge_distance", None)

        searcher["last_target_distance"] = None
        searcher["best_target_distance"] = None
        searcher["stalled_frames"] = 0
        searcher["frames_on_target"] = 0
        searcher["target_timeout_frames"] = estimate_target_timeout_frames(
            state,
            target,
            self.config.planner.drone_speed,
            self.config.planner.step_seconds,
        )
        searcher["controller"].set_target(target)

    def _prepare_targets_for_motion(self):
        for searcher in self.searchers:
            controller = searcher["controller"]
            state = self._normalize_controller_state(controller)
            current_target = searcher["current_target"]
            if current_target is None:
                continue

            if self._assign_recovery_target(searcher, state):
                continue

            current_target = searcher["current_target"]
            if current_target is None:
                continue

            if current_target.get("recovery_target"):
                self._clear_target(searcher)
                continue

            self._apply_boundary_turn_target(searcher, state)

    def _route_needs_boundary_turn(self, state, target):
        _, predicted_states = simulate_motion_prefixes(
            state.as_dict(),
            target,
            self.config.planner,
            path_steps=8,
            predicted_steps=8,
            map_bounds=self.navigation_bounds,
        )
        if not predicted_states:
            return False

        nominal_step_distance = speed_to_step_distance(
            self.config.planner.drone_speed,
            self.config.planner.step_seconds,
        )
        first_state = predicted_states[0]
        first_step_distance = float(
            np.hypot(
                float(first_state["e"]) - float(state.e),
                float(first_state["n"]) - float(state.n),
            )
        )
        if first_step_distance < 0.75 * nominal_step_distance:
            return True

        if len(predicted_states) < 8:
            return True

        min_e, max_e, min_n, max_n = self.navigation_bounds
        return any(
            min(
                float(item["e"]) - float(min_e),
                float(max_e) - float(item["e"]),
                float(item["n"]) - float(min_n),
                float(max_n) - float(item["n"]),
            ) <= nominal_step_distance
            for item in predicted_states
        )

    def _build_boundary_turn_target(self, state, target):
        if target.get("recovery_target") or target.get("boundary_turn_target"):
            return None

        if not self._route_needs_boundary_turn(state, target):
            return None

        goal_e = float(target.get("main_target_e", target["target_e"]))
        goal_n = float(target.get("main_target_n", target["target_n"]))
        desired_heading = float(np.degrees(np.arctan2(goal_e - state.e, goal_n - state.n)))
        turn_heading = boundary_turn_heading(
            state.as_dict(),
            desired_heading,
            self.navigation_bounds,
            trigger_distance=max(
                speed_to_step_distance(
                    self.config.planner.drone_speed,
                    self.config.planner.step_seconds,
                ),
                self.navigation_bounds[0] if self.navigation_bounds[0] > 0.0 else 1.0,
            ),
        )
        if turn_heading is None:
            return None

        turn_step_distance = max(
            speed_to_step_distance(
                self.config.planner.drone_speed,
                self.config.planner.step_seconds,
            ),
            1e-6,
        )
        waypoint_distance = min(
            self.navigation_bounds[1] - self.navigation_bounds[0],
            self.navigation_bounds[3] - self.navigation_bounds[2],
        ) * 0.15
        waypoint_distance = max(4.0 * turn_step_distance, waypoint_distance)
        delta_e = np.sin(np.radians(turn_heading)) * waypoint_distance
        delta_n = np.cos(np.radians(turn_heading)) * waypoint_distance
        min_e, max_e, min_n, max_n = self.navigation_bounds
        target_e = clamp(float(state.e) + float(delta_e), float(min_e), float(max_e))
        target_n = clamp(float(state.n) + float(delta_n), float(min_n), float(max_n))
        if (
            abs(target_e - float(target["target_e"])) <= 1e-6
            and abs(target_n - float(target["target_n"])) <= 1e-6
        ):
            return None

        return {
            **build_target_fields(
                target_e,
                target_n,
                self.config.heatmap.resolution,
                self.height,
            ),
            "boundary_turn_target": True,
        }

    def _apply_boundary_turn_target(self, searcher, state):
        target = searcher["current_target"]
        if target is None:
            return False

        boundary_turn_target = self._build_boundary_turn_target(state, target)
        if boundary_turn_target is None:
            return False

        target.update(boundary_turn_target)
        searcher["last_target_distance"] = None
        searcher["best_target_distance"] = None
        searcher["stalled_frames"] = 0
        searcher["frames_on_target"] = 0
        searcher["target_timeout_frames"] = estimate_target_timeout_frames(
            state,
            target,
            self.config.planner.drone_speed,
            self.config.planner.step_seconds,
        )
        searcher["controller"].set_target(target)
        return True

    def _is_turning_to_escape_edge(self, state, target):
        probe_state = state.as_dict()
        max_probe_steps = max(
            4,
            int(
                np.ceil(
                    30.0
                    / max(
                        turn_rate_to_step_delta(
                            self.config.planner.max_turn_rate_deg,
                            self.config.planner.step_seconds,
                        ),
                        1e-6,
                    )
                )
            ),
        )

        for _ in range(max_probe_steps):
            next_state = advance_toward_target(
                probe_state,
                (target["target_e"], target["target_n"]),
                self.config.planner.drone_speed,
                self.config.planner.max_turn_rate_deg,
                map_bounds=self.map_bounds,
                step_seconds=self.config.planner.step_seconds,
            )
            moved = (
                abs(float(next_state["e"]) - float(probe_state["e"])) > 1e-9
                or abs(float(next_state["n"]) - float(probe_state["n"])) > 1e-9
            )
            turned = abs(float(next_state["heading"]) - float(probe_state["heading"])) > 1e-9
            if moved:
                return turned
            if not turned:
                return False
            probe_state = next_state

        return False

    def _refresh_greedy_subtarget(self, searcher, state, reservation_heatmap):
        current_target = searcher["current_target"]
        if current_target is None or self._target_has_subgoal(current_target):
            return False
        if not self.config.planner.greedy_paths_enabled:
            return False
        if "main_target_e" not in current_target or "main_target_n" not in current_target:
            return False

        _, predicted_states = simulate_motion_prefixes(
            state.as_dict(),
            current_target,
            self.config.planner,
            path_steps=15,
            predicted_steps=8,
            map_bounds=self.navigation_bounds,
        )
        greedy_choice, greedy_path, greedy_blocks = choose_greedy_route_subtarget(
            self.hmu,
            self.config,
            state.as_dict(),
            self.height,
            current_target,
            predicted_states,
            reservation_heatmap,
        )
        if greedy_choice is None:
            return False

        current_target.update(greedy_choice)
        searcher["reserved_path"] = greedy_path
        searcher["reserved_claim_heatmap"] = combine_claim_heatmaps(
            searcher["reserved_claim_heatmap"],
            greedy_blocks,
        )
        searcher["last_target_distance"] = None
        searcher["best_target_distance"] = None
        searcher["stalled_frames"] = 0
        searcher["controller"].set_target(current_target)
        return True

    def _update_camera_pose(self, searcher, state):
        camera_target = searcher["camera_target"]
        camera_cfg = self.config.camera
        step_seconds = self.config.planner.step_seconds
        current_pitch = searcher["camera_pitch_deg"]
        current_yaw = searcher["camera_yaw_deg"]
        target_pitch = current_pitch if camera_target is None else camera_target.get("pitch_deg", current_pitch)
        target_yaw = current_yaw if camera_target is None else camera_target.get("yaw_deg", current_yaw)
        pitch_step_deg = turn_rate_to_step_delta(camera_cfg.pitch_turn_rate_deg, step_seconds)
        yaw_step_deg = turn_rate_to_step_delta(camera_cfg.yaw_turn_rate_deg, step_seconds)
        direct_pose = (
            self._move_camera_axis(
                current_pitch,
                target_pitch,
                camera_cfg.min_pitch,
                camera_cfg.max_pitch,
                pitch_step_deg,
            ),
            self._move_camera_axis(
                current_yaw,
                target_yaw,
                camera_cfg.min_yaw,
                camera_cfg.max_yaw,
                yaw_step_deg,
            ),
        )

        if pitch_step_deg <= 0:
            pitch_candidates = [clamp(current_pitch, camera_cfg.min_pitch, camera_cfg.max_pitch)]
        else:
            pitch_candidates = sorted(
                {
                    clamp(
                        current_pitch + step * pitch_step_deg,
                        camera_cfg.min_pitch,
                        camera_cfg.max_pitch,
                    )
                    for step in (-1, 0, 1)
                }
            )

        if yaw_step_deg <= 0:
            yaw_candidates = [clamp(current_yaw, camera_cfg.min_yaw, camera_cfg.max_yaw)]
        else:
            yaw_candidates = sorted(
                {
                    clamp(
                        current_yaw + step * yaw_step_deg,
                        camera_cfg.min_yaw,
                        camera_cfg.max_yaw,
                    )
                    for step in (-1, 0, 1)
                }
            )

        best_positive_pose = None
        best_positive_key = None
        best_tracking_pose = None
        best_tracking_key = None
        best_safe_pose = None
        best_safe_key = None
        current_target_error = (
            abs(current_pitch - target_pitch) + abs(current_yaw - target_yaw)
        )

        # Choose the *next* pitch/yaw step from the locally reachable 3x3
        # neighborhood, not by blindly stepping both axes toward the desired
        # target. This prevents intermediate camera poses from walking the
        # projection center off the map while still preferring higher-value hits.
        for pitch_deg in pitch_candidates:
            for yaw_deg in yaw_candidates:
                evaluation = self._evaluate_camera_pose(state, pitch_deg, yaw_deg)
                if evaluation is None:
                    continue

                _, footprint_value, footprint_mean, center_value = evaluation
                target_error = abs(pitch_deg - target_pitch) + abs(yaw_deg - target_yaw)
                target_progress = current_target_error - target_error
                move_amount = abs(pitch_deg - current_pitch) + abs(yaw_deg - current_yaw)
                safe_key = (
                    -target_error,
                    -move_amount,
                )
                if best_safe_key is None or safe_key > best_safe_key:
                    best_safe_key = safe_key
                    best_safe_pose = (pitch_deg, yaw_deg)

                if footprint_value <= 0.0:
                    continue

                positive_key = (
                    center_value,
                    footprint_mean,
                    footprint_value,
                    -target_error,
                    -move_amount,
                )
                if best_positive_key is None or positive_key > best_positive_key:
                    best_positive_key = positive_key
                    best_positive_pose = (pitch_deg, yaw_deg)

                tracking_key = (
                    target_progress,
                    center_value,
                    footprint_mean,
                    footprint_value,
                    -target_error,
                    -move_amount,
                )
                if best_tracking_key is None or tracking_key > best_tracking_key:
                    best_tracking_key = tracking_key
                    best_tracking_pose = (pitch_deg, yaw_deg)

        if (
            camera_target is not None
            and best_tracking_pose is not None
            and best_tracking_key is not None
            and best_tracking_key[0] > 1e-9
        ):
            return best_tracking_pose
        if best_positive_pose is not None:
            return best_positive_pose
        if camera_target is not None and (
            abs(direct_pose[0] - current_pitch) > 1e-9
            or abs(direct_pose[1] - current_yaw) > 1e-9
        ):
            return direct_pose
        if best_safe_pose is not None:
            return best_safe_pose
        return current_pitch, current_yaw

    @staticmethod
    def _shift_projection(proj, delta_e, delta_n):
        return {
            key: (point[0] + delta_e, point[1] + delta_n)
            for key, point in proj.items()
        }

    def _projection_from_observation(
        self,
        state,
        camera_pitch_deg,
        camera_yaw_deg,
        projection_point=None,
    ):
        try:
            camera = build_camera(
                self.config.camera,
                (state.e, state.n),
                state.heading,
                pitch=camera_pitch_deg,
                yaw=camera_yaw_deg,
                agl=state.agl,
            )
            projection = camera.project()
            if projection_point is None:
                return projection

            predicted_center = camera.project_center()
            # External callers provide the observed ground projection center. We
            # keep the configured footprint shape/orientation, but shift it so
            # the searched map is updated around the observed point rather than
            # a purely simulated one.
            delta_e = float(projection_point[0]) - float(predicted_center[0])
            delta_n = float(projection_point[1]) - float(predicted_center[1])
            return self._shift_projection(projection, delta_e, delta_n)
        except ValueError:
            return None

    def _evaluate_active_targets(self):
        config = self.config
        drone_step_distance = speed_to_step_distance(
            config.planner.drone_speed,
            config.planner.step_seconds,
        )
        cluster_size = config.planner.cluster_size
        camera = config.camera
        heatmap = self.hmu.data
        cluster_view = self.hmu.get_cluster_view(cluster_size)

        assigned_clusters = set()
        assigned_targets = []
        assigned_camera_targets = []
        assigned_camera_rois = []
        reserved_paths = []
        reservation_heatmap = (
            build_empty_reservation_heatmap(cluster_view.shape)
            if cluster_view is not None
            else None
        )

        for searcher in self.searchers:
            cooldown_frames = searcher["cooldown_frames"]
            if cooldown_frames > 0:
                cooldown_frames -= 1
                searcher["cooldown_frames"] = cooldown_frames
                if cooldown_frames == 0:
                    searcher["cooldown_cluster"] = None

            current_target = searcher["current_target"]
            if current_target is None:
                continue

            controller = searcher["controller"]
            state = self._normalize_controller_state(controller)
            if self._assign_recovery_target(searcher, state):
                continue
            if current_target.get("recovery_target"):
                self._clear_target(searcher)
                continue

            if self._apply_boundary_turn_target(searcher, state):
                current_target = searcher["current_target"]

            nav_dx = current_target["target_e"] - state.e
            nav_dy = current_target["target_n"] - state.n
            navigation_distance = float(np.hypot(nav_dx, nav_dy))
            target_has_subgoal = self._target_has_subgoal(current_target)

            if target_has_subgoal:
                navigation_arrival_radius = max(
                    drone_step_distance,
                    0.25 * cluster_size,
                )
                if navigation_distance <= navigation_arrival_radius:
                    self._promote_target_to_main(searcher, state)
                    current_target = searcher["current_target"]
                    nav_dx = current_target["target_e"] - state.e
                    nav_dy = current_target["target_n"] - state.n
                    navigation_distance = float(np.hypot(nav_dx, nav_dy))
                    target_has_subgoal = False
                    if self._refresh_greedy_subtarget(searcher, state, reservation_heatmap):
                        current_target = searcher["current_target"]
                        nav_dx = current_target["target_e"] - state.e
                        nav_dy = current_target["target_n"] - state.n
                        navigation_distance = float(np.hypot(nav_dx, nav_dy))
                        target_has_subgoal = True

            dx = current_target["target_e"] - state.e
            dy = current_target["target_n"] - state.n
            target_distance = navigation_distance

            target_heading = np.degrees(np.arctan2(dx, dy))
            heading_error = abs(normalize_heading_deg(target_heading - state.heading))

            footprint_span = projection_span_meters(
                camera,
                (current_target["target_e"], current_target["target_n"]),
                target_heading,
                agl=state.agl,
            )

            arrival_radius = max(
                drone_step_distance,
                0.5 * cluster_size,
                0.5 * footprint_span,
            )

            target_reached = (not target_has_subgoal) and (target_distance <= arrival_radius)
            target_has_heat = cluster_has_remaining_heat(
                heatmap, current_target["cluster_bounds"]
            )

            progress_epsilon = max(1.0, 0.2 * drone_step_distance)

            searcher["frames_on_target"] += 1

            best_dist = searcher["best_target_distance"]
            if best_dist is None or target_distance < best_dist - progress_epsilon:
                searcher["best_target_distance"] = target_distance
                searcher["stalled_frames"] = 0
            elif self._is_turning_to_escape_edge(state, current_target):
                searcher["stalled_frames"] = 0
            else:
                searcher["stalled_frames"] += 1

            searcher["last_target_distance"] = target_distance

            target_expired = searcher["frames_on_target"] >= searcher["target_timeout_frames"]
            heading_turn_step = max(
                turn_rate_to_step_delta(
                    config.planner.max_turn_rate_deg,
                    config.planner.step_seconds,
                ),
                1e-6,
            )
            turn_allowance_frames = int(np.ceil(heading_error / heading_turn_step))
            stall_limit_frames = max(6, turn_allowance_frames + 6)
            target_stalled = searcher["stalled_frames"] >= stall_limit_frames

            if target_reached or not target_has_heat or target_stalled or target_expired:
                self._clear_target(searcher, cooldown=(target_stalled or target_expired))
            else:
                bounds = current_target["cluster_bounds"]
                assigned_clusters.add(bounds)
                assigned_targets.append(current_target)
                searcher["camera_target"] = choose_camera_projection_target(
                    self.hmu,
                    config,
                    state.as_dict(),
                    self.height,
                    searcher["camera_pitch_deg"],
                    searcher["camera_yaw_deg"],
                    reserved_targets=assigned_camera_targets,
                    reserved_rois=assigned_camera_rois,
                    preferred_target=searcher["camera_target"],
                    flight_target=current_target,
                )
                if searcher["camera_target"] is not None:
                    assigned_camera_targets.append(searcher["camera_target"])
                    assigned_camera_rois.append(
                        searcher["camera_target"].get("projection_roi")
                    )
                if searcher["reserved_path"]:
                    reserved_paths.append(searcher["reserved_path"])
                merge_reservation_heatmap(
                    reservation_heatmap,
                    searcher["reserved_claim_heatmap"],
                )

        return (
            assigned_clusters,
            assigned_targets,
            assigned_camera_targets,
            assigned_camera_rois,
            reserved_paths,
            reservation_heatmap,
        )

    def _replan_targets(self):
        config = self.config
        heatmap = self.hmu.data

        if self.hmu.total_heat <= 1e-12:
            for searcher in self.searchers:
                self._clear_target(searcher)
            return

        (
            assigned_clusters,
            assigned_targets,
            assigned_camera_targets,
            assigned_camera_rois,
            reserved_paths,
            reservation_heatmap,
        ) = self._evaluate_active_targets()

        planner_cfg = config.planner
        height = self.height
        hmu = self.hmu
        cluster_view = hmu.get_cluster_view(planner_cfg.cluster_size)
        idle_searchers = []
        candidate_lists = {}
        idle_states = {}

        for idx, searcher in enumerate(self.searchers):
            if searcher["current_target"] is not None:
                continue

            blocked_clusters = set(assigned_clusters)
            cooldown_cluster = searcher["cooldown_cluster"]
            if cooldown_cluster is not None:
                blocked_clusters.add(cooldown_cluster)

            controller = searcher["controller"]
            state = self._normalize_controller_state(controller)
            if self._assign_recovery_target(searcher, state):
                continue

            idle_searchers.append((idx, searcher))
            idle_states[idx] = state
            candidate_lists[idx] = collect_cluster_target_candidates(
                hmu,
                config,
                state.as_dict(),
                height,
                reserved_clusters=blocked_clusters,
                reserved_targets=assigned_targets,
                reserved_paths=reserved_paths,
            )

        joint_assignment = {}
        if idle_searchers and cluster_view is not None:
            joint_assignment = select_joint_target_assignment(
                candidate_lists,
                cluster_view,
                planner_cfg,
                reservation_heatmap,
                reserved_targets=assigned_targets,
                reserved_paths=reserved_paths,
            )

        for idx, searcher in idle_searchers:
            controller = searcher["controller"]
            state = idle_states[idx]

            target = None
            assignment_entry = joint_assignment.get(idx)
            if assignment_entry is not None:
                candidate, ranking_key, unique_gain, overlap_gain, route_gain = assignment_entry
                target = dict(candidate["target"])
                target["unique_route_gain"] = float(unique_gain)
                target["overlap_gain"] = float(overlap_gain)
                target["route_gain"] = float(route_gain)
                target["distance_efficiency"] = float(ranking_key[0])
                reserved_path = candidate["path"]
                reserved_claim_heatmap = candidate["claim_heatmap"]
                if planner_cfg.greedy_paths_enabled:
                    greedy_choice, greedy_path, greedy_claim_heatmap = choose_greedy_route_subtarget(
                        hmu,
                        config,
                        state.as_dict(),
                        height,
                        target,
                        candidate["predicted_states"],
                        reservation_heatmap,
                    )
                    if greedy_choice is not None:
                        target.update(greedy_choice)
                        reserved_path = greedy_path
                        reserved_claim_heatmap = combine_claim_heatmaps(
                            reserved_claim_heatmap,
                            greedy_claim_heatmap,
                        )
                target["reserved_path"] = reserved_path
                target["reserved_claim_heatmap"] = reserved_claim_heatmap
                searcher["reserved_path"] = reserved_path
                searcher["reserved_claim_heatmap"] = reserved_claim_heatmap
            else:
                target = choose_best_cluster_target(
                    hmu,
                    config,
                    state.as_dict(),
                    height,
                    reserved_clusters=assigned_clusters,
                    reserved_targets=assigned_targets,
                    reserved_paths=reserved_paths,
                    reserved_blocks=reservation_heatmap,
                )
                searcher["reserved_path"] = None if target is None else target.get("reserved_path")
                searcher["reserved_claim_heatmap"] = (
                    None if target is None else target.get("reserved_claim_heatmap")
                )

            searcher["current_target"] = target
            searcher["last_target_distance"] = None
            searcher["best_target_distance"] = None
            searcher["stalled_frames"] = 0
            searcher["frames_on_target"] = 0
            searcher["target_timeout_frames"] = 0

            if target is not None:
                searcher["current_target"] = target
                self._apply_boundary_turn_target(searcher, state)
                target = searcher["current_target"]
                controller.set_target(target)
                searcher["camera_target"] = choose_camera_projection_target(
                    hmu,
                    config,
                    state.as_dict(),
                    height,
                    searcher["camera_pitch_deg"],
                    searcher["camera_yaw_deg"],
                    reserved_targets=assigned_camera_targets,
                    reserved_rois=assigned_camera_rois,
                    preferred_target=None,
                    flight_target=target,
                )
                if searcher["camera_target"] is not None:
                    assigned_camera_targets.append(searcher["camera_target"])
                    assigned_camera_rois.append(
                        searcher["camera_target"].get("projection_roi")
                    )

                timeout = estimate_target_timeout_frames(
                    state,
                    target,
                    planner_cfg.drone_speed,
                    planner_cfg.step_seconds,
                )
                searcher["target_timeout_frames"] = timeout

                assigned_clusters.add(target["cluster_bounds"])
                assigned_targets.append(target)
                if searcher["reserved_path"]:
                    reserved_paths.append(searcher["reserved_path"])
                merge_reservation_heatmap(
                    reservation_heatmap,
                    searcher["reserved_claim_heatmap"],
                )
            else:
                controller.clear_target()

    def step(self):
        config = self.config
        camera = config.camera
        drone_step_distance = speed_to_step_distance(
            config.planner.drone_speed,
            config.planner.step_seconds,
        )
        step_distance = max(1.0, 0.5 * drone_step_distance)
        dt_seconds = float(config.planner.step_seconds)

        hmu = self.hmu

        self._prepare_targets_for_motion()

        for searcher in self.searchers:
            controller = searcher["controller"]

            start_state = self._normalize_controller_state(controller)
            start_pitch_deg = searcher["camera_pitch_deg"]
            start_yaw_deg = searcher["camera_yaw_deg"]
            controller.update()
            end_state = self._normalize_controller_state(controller)
            end_pitch_deg, end_yaw_deg = self._update_camera_pose(searcher, end_state)

            projection = apply_state_transition_sweep(
                hmu,
                camera,
                start_state.as_dict(),
                end_state.as_dict(),
                step_distance=step_distance,
                start_pitch_deg=start_pitch_deg,
                end_pitch_deg=end_pitch_deg,
                start_yaw_deg=start_yaw_deg,
                end_yaw_deg=end_yaw_deg,
                dt_seconds=dt_seconds,
            )

            searcher["camera_pitch_deg"] = end_pitch_deg
            searcher["camera_yaw_deg"] = end_yaw_deg
            searcher["last_projection"] = projection
            searcher["last_state"] = end_state

        self._replan_targets()
        self._refresh_finished_state()

        return self.get_render_state()

    def observe_external(self, observations, dt_seconds=None):
        if len(observations) != len(self.searchers):
            raise ValueError(
                f"Expected {len(self.searchers)} observations, got {len(observations)}."
            )
        if dt_seconds is None:
            dt_seconds = float(self.config.planner.step_seconds)

        camera_cfg = self.config.camera

        for searcher, observation in zip(self.searchers, observations):
            controller = searcher["controller"]
            state = self._clamp_state_to_map(observation["state"])
            controller.set_state(state)

            if observation.get("camera_pitch_deg") is not None:
                searcher["camera_pitch_deg"] = clamp(
                    float(observation["camera_pitch_deg"]),
                    camera_cfg.min_pitch,
                    camera_cfg.max_pitch,
                )
            if observation.get("camera_yaw_deg") is not None or observation.get("camera_roll_deg") is not None:
                searcher["camera_yaw_deg"] = clamp(
                    float(observation.get("camera_yaw_deg", observation.get("camera_roll_deg"))),
                    camera_cfg.min_yaw,
                    camera_cfg.max_yaw,
                )

            projection = self._projection_from_observation(
                state,
                searcher["camera_pitch_deg"],
                searcher["camera_yaw_deg"],
                projection_point=observation.get("projection_point"),
            )
            if projection is not None:
                self.hmu.change_to_zeroes(projection, dt_seconds=dt_seconds)

            searcher["last_projection"] = projection
            searcher["last_state"] = state

        self._replan_targets()
        self._refresh_finished_state()

        return self.get_render_state()

    def get_render_state(self):
        render_state = []

        for searcher in self.searchers:
            controller = searcher["controller"]
            state = self._clamp_state_to_map(controller.get_state())
            controller.set_state(state)
            camera_pitch_deg = searcher["camera_pitch_deg"]
            camera_yaw_deg = searcher["camera_yaw_deg"]
            try:
                camera_projection_point = build_camera(
                    self.config.camera,
                    (state.e, state.n),
                    state.heading,
                    pitch=camera_pitch_deg,
                    yaw=camera_yaw_deg,
                    agl=state.agl,
                ).project_center()
            except ValueError:
                camera_projection_point = None

            render_state.append(
                {
                    "drone_id": searcher["drone_id"],
                    "state": state,
                    "target": searcher["current_target"],
                    "camera_target": searcher["camera_target"],
                    "camera_pitch_deg": camera_pitch_deg,
                    "camera_yaw_deg": camera_yaw_deg,
                    "camera_projection_point": camera_projection_point,
                    "projection": searcher["last_projection"],
                }
            )

        return render_state
