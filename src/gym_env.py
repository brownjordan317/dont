import gymnasium as gym
from gymnasium import spaces
import numpy as np
from flight_engine.helpers import Position, FlightMode, wrap_angle
from flight_engine.trans_coorders import CoordinateTransformer

class MultiUAVEnv(gym.Env):
    def __init__(
        self, 
        aircraft_list, 
        tl, 
        br, 
        dt=0.3, 
        max_steps=10_000, 
        boundary_margin=0.15, 
        mission_waypoint_count=3,
        mode='gen_mission', 
        caution_dist=30.0, 
        critical_dist=3.0,
        inference_mode=False,
        reward_config=None,
        hard_safety_config=None,
        anti_circling_config=None,
        ):

        super().__init__()
        self.aircraft_list = aircraft_list
        # Maximum number of UAVs this environment supports 
        # (observation/action space is fixed to this)
        self.max_uavs = 5
        self.dt, self.max_steps = dt, max_steps
        # Fraction of the geo-bounds to exclude from waypoint generation 
        # (keeps waypoints away from edges)
        self.boundary_margin = boundary_margin
        self.mission_waypoint_count = mission_waypoint_count
        # 'gen_mission': randomly generate waypoints each episode;
        # 'manual_mission': use preset positions
        self.mode = mode
        self.caution_dist, self.critical_dist = caution_dist, critical_dist
        self.caution_dist_breakers = []
        self.crit_dist_breakers = []
        # When True, reward calculations skipped for faster inference
        self.inference_mode = inference_mode
        reward_config = reward_config or {}
        hard_safety_config = hard_safety_config or {}
        anti_circling_config = anti_circling_config or {}

        # ------------------------------------------------------------------ #
        # Reward / penalty coefficients                                       #
        # ------------------------------------------------------------------ #

        # --- Navigation rewards ---
        # Scale factor for distance-to-waypoint progress reward (delta-dist based)
        self.reward_progress_scale = float(
            reward_config.get("progress_scale", 6.0)
        )
        # Scale factor for best-distance improvement bonus
        # Rewards closing in on the personal best distance to waypoint,
        # preventing circular flight patterns from generating reward.
        self.reward_best_dist_scale = float(
            reward_config.get("best_dist_scale", 3.0)
        )
        # Penalty coefficient on squared action magnitude (discourages large turns)
        self.penalty_control_effort = float(
            reward_config.get("control_effort_penalty", 0.3)
        )
        # Flat per-step penalty to encourage mission efficiency
        self.penalty_timestep = float(
            reward_config.get("timestep_penalty", 0.1)
        )

        # --- Waypoint arrival bonus ---
        # Large sparse reward awarded on waypoint arrival
        self.reward_waypoint_hit = float(
            reward_config.get("waypoint_hit_reward", 3000.0)
        )

        # --- Geofence / boundary penalty ---
        # Base coefficient for the exponential geofence penalty.
        # Penalty = penalty_geofence_base * (e^(k * normalised_overshoot) - 1)
        # so it is near-zero just inside the fence but grows steeply outside.
        self.penalty_geofence_base = float(
            reward_config.get("geofence_base_penalty", 20.0)
        )
        # Steepness of the exponential growth (higher → more aggressive)
        self.penalty_geofence_k = float(
            reward_config.get("geofence_exp_k", 1.0)
        )

        # --- Deconfliction penalties ---
        # Linear penalty for pairs inside the caution radius.
        # Scaled so that touching the caution boundary gives ~0 penalty and
        # closing to zero separation gives ~penalty_caution_max.
        self.penalty_caution_scale = float(
            reward_config.get("caution_scale_penalty", 8.0)
        )
        # One-shot terminal penalty applied when a critical violation occurs.
        # This is much larger than a per-step value so crashes are catastrophic.
        self.penalty_crash = float(
            reward_config.get("crash_penalty", 5000.0)
        )
        self.penalty_safety_override = float(
            reward_config.get("safety_override_penalty", 25.0)
        )

        # --- Hard safety fallback ---
        self.hard_safety_enabled = bool(
            hard_safety_config.get("enabled", False)
        )
        self.hard_safety_activation_dist = float(
            hard_safety_config.get(
                "activation_dist",
                max(self.critical_dist * 2.0, self.critical_dist + 1.0),
            )
        )
        self.hard_safety_release_dist = float(
            hard_safety_config.get(
                "release_dist",
                self.hard_safety_activation_dist,
            )
        )
        self.hard_safety_hold_steps = int(
            hard_safety_config.get("hold_steps", 0)
        )
        self.hard_safety_turn_strength = float(
            hard_safety_config.get("turn_strength", 1.0)
        )
        self.hard_safety_require_closing = bool(
            hard_safety_config.get("require_closing", True)
        )

        # --- Anti-circling breakout ---
        self.anti_circling_enabled = bool(
            anti_circling_config.get("enabled", False)
        )
        self.anti_circling_min_progress_m = float(
            anti_circling_config.get("min_progress_m", 0.5)
        )
        self.anti_circling_min_distance_from_wp_m = float(
            anti_circling_config.get("min_distance_from_wp_m", 25.0)
        )
        self.anti_circling_min_turn_input = float(
            anti_circling_config.get("min_turn_input", 0.35)
        )
        self.anti_circling_trigger_steps = int(
            anti_circling_config.get("trigger_steps", 30)
        )
        self.anti_circling_breakout_steps = int(
            anti_circling_config.get("breakout_steps", 12)
        )
        self.anti_circling_turn_strength = float(
            anti_circling_config.get("turn_strength", 1.0)
        )

        # --- Geofence violation tracking ---
        # counts distinct exit events per UAV, not sustained frames
        self.geofence_exit_counts = [0 for _ in range(self.max_uavs)]
        self._was_outside = [False for _ in range(self.max_uavs)]

        # --- Per-drone personal-best distance to current waypoint ---
        # Reset when a new waypoint is assigned.  Prevents circle-farming.
        self._best_dist_to_wp = np.full(self.max_uavs, np.inf)

        # --- Crash flag (set when any critical violation occurs this step) ---
        self._crashed = False
        self.safety_override_counts = [0 for _ in range(self.max_uavs)]
        self.anti_circling_counts = [0 for _ in range(self.max_uavs)]
        self._stalled_turn_steps = np.zeros(self.max_uavs, dtype=np.int32)
        self._breakout_steps_remaining = np.zeros(self.max_uavs, dtype=np.int32)
        self._steps_since_best_progress = np.zeros(self.max_uavs, dtype=np.int32)
        self._safety_turn_memory = np.zeros(self.max_uavs, dtype=np.float32)
        self._safety_hold_remaining = np.zeros(self.max_uavs, dtype=np.int32)

        self.update_bounds(tl, br)
        # Number of observation features per UAV:
        #   self-features  : 14  (was 10 – added boundary proximity × 4)
        #   2 neighbours × 7 features each = 14  (added closing-urgency flag)
        # Total per UAV    : 28
        self.obs_per_uav = 28

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.max_uavs * self.obs_per_uav,),
            dtype=np.float32
        )
        # One continuous action per UAV:
        #   desired turn rate normalised to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.max_uavs,), dtype=np.float32
        )
        self.current_step = 0

        # Pre-allocated numpy buffers to avoid per-step heap allocations
        self._obs_buffer = np.zeros(
            self.max_uavs * self.obs_per_uav,
            dtype=np.float32
        )
        # Post-step local (x, y) positions
        self._local_pos_cache = np.zeros((self.max_uavs, 2))
        # Pre-step local (x, y) positions
        self._prev_local_cache = np.zeros((self.max_uavs, 2))
        # Current waypoint local (x, y) positions
        self._wp_local_cache = np.zeros((self.max_uavs, 2))
        # Pairwise separation matrices (reused each step)
        self._dx_matrix = np.zeros((self.max_uavs, self.max_uavs))
        self._dy_matrix = np.zeros((self.max_uavs, self.max_uavs))
        self._dist_matrix = np.zeros((self.max_uavs, self.max_uavs))

        self.clear_missions = True

    # ======================================================================= #
    # STEP                                                                    #
    # ======================================================================= #

    def step(self, actions):
        self.current_step += 1
        n = len(self.aircraft_list)
        actions = np.asarray(actions, dtype=np.float32)[:n]
        applied_actions = np.clip(actions.copy(), -1.0, 1.0)
        policy_actions = applied_actions.copy()

        # Skip reward calculations in inference mode
        if self.inference_mode:
            rewards = 0.0
        else:
            rewards = np.zeros(n)

        self._crashed = False

        # --- 1. Batch Transformation: Pre-move state & Waypoints ---
        lats = [ac.position.latitude for ac in self.aircraft_list]
        lons = [ac.position.longitude for ac in self.aircraft_list]

        x_vals, y_vals = self.transformer.geo_to_local(lats, lons)
        self._prev_local_cache[:n, 0] = x_vals
        self._prev_local_cache[:n, 1] = y_vals
        self._update_waypoint_local_cache(self._prev_local_cache)
        self._update_pairwise_distance_cache(self._prev_local_cache)

        # Distance-to-waypoint BEFORE physics (for progress reward)
        dist_before = np.linalg.norm(
            self._wp_local_cache[:n] - self._prev_local_cache[:n],
            axis=1
        )
        best_dist_before = self._best_dist_to_wp.copy()

        anti_circling_activated = self._apply_anti_circling_breakout(
            applied_actions,
            dist_before,
        )
        safety_activated = self._apply_hard_safety_turns(applied_actions)

        # --- 2. Physics update ---
        for i, ac in enumerate(self.aircraft_list):
            if ac.flight_mode == FlightMode.LOITERING:
                ac._update_loiter(
                    *self._prev_local_cache[i],
                    self.dt, self.transformer
                )
            else:
                ac.update_simple(
                    applied_actions[i] * ac.dynamics.max_turn_rate,
                    self.dt,
                    self.transformer
                )

        # --- 3. Batch Transformation: Post-move state ---
        lats_post = [ac.position.latitude for ac in self.aircraft_list]
        lons_post = [ac.position.longitude for ac in self.aircraft_list]
        px, py = self.transformer.geo_to_local(lats_post, lons_post)
        self._local_pos_cache[:n, 0] = px
        self._local_pos_cache[:n, 1] = py

        # --- 4. Collision Detection ---
        self._update_pairwise_distance_cache(self._local_pos_cache)
        if n > 1:
            # Track violations for metrics (always)
            self._track_collision_violations()

            # Apply penalties only in training mode
            if not self.inference_mode:
                self._calculate_collision_rewards(rewards)

        # --- 5. Nav Rewards & Waypoint Management ---
        dist_after = np.linalg.norm(
            self._wp_local_cache[:n] - self._local_pos_cache[:n],
            axis=1
        )

        for i, ac in enumerate(self.aircraft_list):
            if not self.inference_mode:
                # ----------------------------------------------------------
                # 5a. Progress reward: reward closing on the waypoint THIS step
                # ----------------------------------------------------------
                delta_dist = dist_before[i] - dist_after[i]
                rewards[i] += delta_dist * self.reward_progress_scale

                # ----------------------------------------------------------
                # 5b. Best-distance bonus:
                #   Reward the drone for reaching a NEW closest-ever distance
                #   to its current waypoint.  Once the best has been beaten,
                #   a circle will never earn this bonus again for this waypoint.
                # ----------------------------------------------------------
                best_dist = self._best_dist_to_wp[i]
                if not np.isfinite(best_dist):
                    self._best_dist_to_wp[i] = dist_after[i]
                elif dist_after[i] < best_dist:
                    improvement = best_dist - dist_after[i]
                    rewards[i] += improvement * self.reward_best_dist_scale
                    self._best_dist_to_wp[i] = dist_after[i]

                # ----------------------------------------------------------
                # 5c. Control effort & timestep penalties
                # ----------------------------------------------------------
                rewards[i] -= self.penalty_control_effort * (applied_actions[i] ** 2)
                rewards[i] -= self.penalty_timestep
                if safety_activated[i]:
                    rewards[i] -= self.penalty_safety_override

            # --- Waypoint arrival ---
            wm = ac.waypoint_manager
            if wm.current_waypoint:
                if self._check_line_segment_arrival_local(
                    self._prev_local_cache[i], self._local_pos_cache[i],
                    self._wp_local_cache[i], wm.arrival_threshold
                ):
                    reached_waypoint = wm.current_waypoint
                    if not self.inference_mode:
                        rewards[i] += self.reward_waypoint_hit
                    wm.advance()
                    if self.mode == 'gen_mission':
                        self._refill_mission(ac)
                    wm.hit_waypoints.append(reached_waypoint)
                    # Reset personal best for the NEW waypoint
                    self._best_dist_to_wp[i] = np.inf
                    self._stalled_turn_steps[i] = 0
                    self._breakout_steps_remaining[i] = 0
                    self._steps_since_best_progress[i] = 0

                if not wm.has_waypoints():
                    ac._enter_loiter()

            # --- Geofence ---
            inside = (
                self.min_lat < ac.position.latitude < self.max_lat and
                self.min_lon < ac.position.longitude < self.max_lon
            )

            # Count geofence exit as a discrete event
            if not inside and not self._was_outside[i]:
                self.geofence_exit_counts[i] += 1
                self._was_outside[i] = True

            if inside:
                self._was_outside[i] = False

            # ----------------------------------------------------------
            # 5d. Exponential geofence penalty:
            #   Normalised overshoot = how far outside relative to box size
            #   penalty = base * (exp(k * overshoot) - 1)
            #   → ≈0 at the boundary, grows steeply with distance outside.
            # ----------------------------------------------------------
            if not self.inference_mode and not inside:
                box_lat_span = self.max_lat - self.min_lat
                box_lon_span = self.max_lon - self.min_lon

                lat_over = max(
                    0.0,
                    self.min_lat - ac.position.latitude,
                    ac.position.latitude - self.max_lat
                ) / (box_lat_span + 1e-9)

                lon_over = max(
                    0.0,
                    self.min_lon - ac.position.longitude,
                    ac.position.longitude - self.max_lon
                ) / (box_lon_span + 1e-9)

                normalised_overshoot = np.sqrt(lat_over ** 2 + lon_over ** 2)

                # In gym_env.py, inside the reward logic:
                overshoot_term = np.exp(
                    self.penalty_geofence_k * normalised_overshoot
                    ) - 1.0
                geo_penalty = (
                      self.penalty_geofence_base 
                    * np.clip(overshoot_term, 0, 500)) # Clip the exponential growth
                rewards[i] -= geo_penalty

        self._update_anti_circling_state(
            policy_actions,
            best_dist_before,
            dist_after,
            anti_circling_activated,
        )

        self._update_waypoint_local_cache(self._local_pos_cache)
        self._prime_best_distance_to_waypoint(self._local_pos_cache)
        self._update_obs_buffer()

        # ------------------------------------------------------------------
        # Crash termination: any critical violation ends the episode and
        # already-applied per-pair penalty stands; the crash flag triggers
        # the `terminated` signal so the agent learns the state is absorbing.
        # ------------------------------------------------------------------
        terminated = self._crashed
        truncated  = self.current_step >= self.max_steps

        # Also end if all drones have no more waypoints
        all_done = all(
            not ac.waypoint_manager.has_waypoints()
            for ac in self.aircraft_list
        )

        done = terminated or truncated or all_done

        total_reward = float(rewards.sum()) if not self.inference_mode else 0.0

        info = {
            "waypoints_hit": sum(
                len(ac.waypoint_manager.hit_waypoints)
                for ac in self.aircraft_list
            ),
            "crashed": terminated,
            "safety_overrides": int(np.sum(safety_activated)),
            "anti_circling_breakouts": int(np.sum(anti_circling_activated)),
        }

        if done:
            info["episode_metrics"] = self.get_uav_metrics()

        return (
            self._obs_buffer.copy(),
            total_reward,
            done,
            False,
            info
        )

    # ======================================================================= #
    # OBSERVATION BUFFER                                                      #
    # ======================================================================= #

    def _update_obs_buffer(self):
        n = len(self.aircraft_list)
        headings = np.array(
            [ac.heading for ac in self.aircraft_list],
            dtype=np.float32
        )
        speeds = np.array(
            [ac.dynamics.cruise_speed for ac in self.aircraft_list],
            dtype=np.float32
        )
        has_wp = np.array(
            [1.0 if ac.waypoint_manager.has_waypoints() else 0.0
             for ac in self.aircraft_list],
            dtype=np.float32
        )

        # Vector and scalar distance/bearing to current waypoint for each UAV
        d_wp = self._wp_local_cache[:n] - self._local_pos_cache[:n]
        dist_wp = np.linalg.norm(d_wp, axis=1)
        brg_wp = np.arctan2(d_wp[:, 0], d_wp[:, 1])
        # Relative bearing: 0 when waypoint is dead ahead, ±π when behind
        rel_brg_wp = ((brg_wp - headings + np.pi) % (2 * np.pi)) - np.pi

        # How far inside each boundary wall (normalised, clipped to [0, 1])
        # Gives the policy early warning when approaching the geofence.
        box_lat = max(self.max_lat - self.min_lat, 1e-6)
        box_lon = max(self.max_lon - self.min_lon, 1e-6)

        # Mask self-distances on the diagonal
        masked_dist = self._dist_matrix[:n, :n].copy()
        np.fill_diagonal(masked_dist, np.inf)

        obs_temp = np.zeros(
            (self.max_uavs, self.obs_per_uav),
            dtype=np.float32
        )

        for i, ac in enumerate(self.aircraft_list):
            # Normalised distances to the four geofence walls [0 = at wall, 1 = far away]
            d_south = np.clip((ac.position.latitude  - self.min_lat) / box_lat, 0.0, 1.0)
            d_north = np.clip((self.max_lat - ac.position.latitude)  / box_lat, 0.0, 1.0)
            d_west  = np.clip((ac.position.longitude - self.min_lon) / box_lon, 0.0, 1.0)
            d_east  = np.clip((self.max_lon - ac.position.longitude)  / box_lon, 0.0, 1.0)

            # Normalised personal-best distance to current waypoint
            # 0 = already at best distance, 1 = never been closer than 1 km
            best_norm = np.clip(self._best_dist_to_wp[i] / 1000.0, 0.0, 1.0)

            # --- Self features (14 dims) ---
            self_feat = [
                np.clip(dist_wp[i] / 1000.0, 0, 1.0),        # Norm distance to WP
                np.sin(rel_brg_wp[i]), np.cos(rel_brg_wp[i]),  # Bearing to WP
                np.sin(headings[i]), np.cos(headings[i]),       # Absolute heading
                speeds[i] / 30.0,                               # Norm speed
                has_wp[i],                                       # WP present flag
                best_norm,                                       # Personal best dist to WP
                d_south, d_north, d_west, d_east,               # Geofence proximity ×4
                0.0, 0.0,                                        # 2 reserved
            ]

            # --- Neighbour features (2 neighbours × 7 dims = 14 dims) ---
            neigh_feat = []
            sorted_neighbors = np.argsort(masked_dist[i])

            for neighbor_slot in range(2):
                if neighbor_slot < (n - 1):
                    nb_idx = sorted_neighbors[neighbor_slot]
                    dist   = masked_dist[i, nb_idx]

                    dx = self._dx_matrix[i, nb_idx]
                    dy = self._dy_matrix[i, nb_idx]
                    other_ac = self.aircraft_list[nb_idx]

                    # Closing speed (positive = converging)
                    v1_x = speeds[i]           * np.sin(headings[i])
                    v1_y = speeds[i]           * np.cos(headings[i])
                    v2_x = other_ac.dynamics.cruise_speed * np.sin(other_ac.heading)
                    v2_y = other_ac.dynamics.cruise_speed * np.cos(other_ac.heading)

                    v_closing = ((v1_x - v2_x) * dx + (v1_y - v2_y) * dy) / (dist + 1e-6)
                    ttc = np.clip(
                        (dist / v_closing if v_closing > 0 else 50.0) / 50.0,
                        0, 1.0
                    )

                    rel_brg = ((np.arctan2(dx, dy) - headings[i] + np.pi) % (2 * np.pi)) - np.pi
                    rel_hdg = ((other_ac.heading - headings[i] + np.pi) % (2 * np.pi)) - np.pi

                    # Urgency flag: 1 if inside caution zone AND converging, else 0.
                    # Makes deconfliction salient without dominating the observation.
                    urgency = 1.0 if (dist < self.caution_dist and v_closing > 0) else 0.0

                    neigh_feat.extend([
                        np.clip(dist / 500.0, 0, 1.0),  # Norm separation
                        ttc,                              # Norm time-to-collision
                        np.sin(rel_brg),                  # Bearing to neighbour (sin)
                        np.cos(rel_brg),                  # Bearing to neighbour (cos)
                        np.sin(rel_hdg),                  # Relative heading (sin)
                        np.cos(rel_hdg),                  # Relative heading (cos)
                        urgency,                          # Converging inside caution zone
                    ])
                else:
                    # Pad empty neighbour slots with "no threat" sentinel values
                    neigh_feat.extend([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

            obs_temp[i] = np.concatenate([self_feat, neigh_feat])

        self._obs_buffer[:] = obs_temp.flatten()

    def _apply_hard_safety_turns(self, actions):
        n = len(self.aircraft_list)
        activated = np.zeros(n, dtype=bool)

        if not self.hard_safety_enabled or n < 2:
            return activated

        headings = np.array(
            [ac.heading for ac in self.aircraft_list],
            dtype=np.float32
        )
        speeds = np.array(
            [ac.dynamics.cruise_speed for ac in self.aircraft_list],
            dtype=np.float32
        )

        for i in range(n):
            closest_dist = np.inf
            chosen_turn = self._safety_turn_memory[i]
            keep_latched_turn = self._safety_hold_remaining[i] > 0

            for j in range(n):
                if i == j:
                    continue

                dx = self._prev_local_cache[j, 0] - self._prev_local_cache[i, 0]
                dy = self._prev_local_cache[j, 1] - self._prev_local_cache[i, 1]
                dist = np.hypot(dx, dy)

                if dist >= self.hard_safety_activation_dist:
                    if (
                        self._safety_hold_remaining[i] > 0
                        and dist < self.hard_safety_release_dist
                    ):
                        keep_latched_turn = True
                    continue

                v1_x = speeds[i] * np.sin(headings[i])
                v1_y = speeds[i] * np.cos(headings[i])
                v2_x = speeds[j] * np.sin(headings[j])
                v2_y = speeds[j] * np.cos(headings[j])
                # Positive means the pair is closing along the current
                # line-of-sight from UAV i toward UAV j.
                v_closing = (
                    ((v1_x - v2_x) * dx)
                    + ((v1_y - v2_y) * dy)
                ) / (dist + 1e-6)

                if self.hard_safety_require_closing and v_closing <= 0:
                    continue

                rel_brg = (
                    (np.arctan2(dx, dy) - headings[i] + np.pi) % (2 * np.pi)
                ) - np.pi
                if abs(rel_brg) < (np.pi / 6.0):
                    # Head-on or nearly head-on: use a consistent starboard pass.
                    turn_sign = 1.0
                else:
                    # Otherwise, turn away from the intruder's side.
                    turn_sign = -np.sign(rel_brg)
                    if turn_sign == 0:
                        turn_sign = 1.0

                escape_bearing = wrap_angle(
                    headings[i] + (turn_sign * (np.pi / 2.0))
                )
                turn_cmd = self._bearing_to_normalized_action(
                    headings[i],
                    escape_bearing,
                    self.aircraft_list[i].dynamics.max_turn_rate,
                )

                if dist < closest_dist:
                    closest_dist = dist
                    chosen_turn = turn_cmd * self.hard_safety_turn_strength
                    keep_latched_turn = True

            if keep_latched_turn:
                actions[i] = np.clip(chosen_turn, -1.0, 1.0)
                activated[i] = True
                self.safety_override_counts[i] += 1
                self._safety_turn_memory[i] = actions[i]
                if closest_dist < np.inf:
                    self._safety_hold_remaining[i] = self.hard_safety_hold_steps
                elif self._safety_hold_remaining[i] > 0:
                    self._safety_hold_remaining[i] -= 1
            else:
                self._safety_turn_memory[i] = 0.0
                self._safety_hold_remaining[i] = 0

        return activated

    def _apply_anti_circling_breakout(self, actions, dist_before):
        n = len(self.aircraft_list)
        activated = np.zeros(n, dtype=bool)

        if not self.anti_circling_enabled:
            return activated

        for i, ac in enumerate(self.aircraft_list):
            if ac.flight_mode == FlightMode.LOITERING:
                self._stalled_turn_steps[i] = 0
                self._breakout_steps_remaining[i] = 0
                continue

            wm = ac.waypoint_manager
            if wm.current_waypoint is None:
                self._stalled_turn_steps[i] = 0
                self._breakout_steps_remaining[i] = 0
                continue

            if self._breakout_steps_remaining[i] <= 0:
                continue

            wp_vec = self._wp_local_cache[i] - self._prev_local_cache[i]
            rel_brg_wp = (
                (np.arctan2(wp_vec[0], wp_vec[1]) - ac.heading + np.pi)
                % (2 * np.pi)
            ) - np.pi
            wp_bearing = wrap_angle(ac.heading + rel_brg_wp)
            turn_cmd = self._bearing_to_normalized_action(
                ac.heading,
                wp_bearing,
                ac.dynamics.max_turn_rate,
            )
            if abs(turn_cmd) < 0.25:
                turn_cmd = 0.0

            if dist_before[i] > wm.arrival_threshold:
                actions[i] = np.clip(
                    turn_cmd * self.anti_circling_turn_strength,
                    -1.0,
                    1.0,
                )
                activated[i] = True
                self._breakout_steps_remaining[i] -= 1
                self.anti_circling_counts[i] += 1
            else:
                self._breakout_steps_remaining[i] = 0

        return activated

    def _update_anti_circling_state(
        self,
        policy_actions,
        best_dist_before,
        dist_after,
        anti_circling_activated,
    ):
        if not self.anti_circling_enabled:
            return

        for i, ac in enumerate(self.aircraft_list):
            wm = ac.waypoint_manager

            if (
                ac.flight_mode == FlightMode.LOITERING
                or wm.current_waypoint is None
                or anti_circling_activated[i]
            ):
                if ac.flight_mode == FlightMode.LOITERING or wm.current_waypoint is None:
                    self._stalled_turn_steps[i] = 0
                    self._breakout_steps_remaining[i] = 0
                    self._steps_since_best_progress[i] = 0
                continue

            is_turning = abs(policy_actions[i]) >= self.anti_circling_min_turn_input
            is_far_from_wp = (
                dist_after[i] >= self.anti_circling_min_distance_from_wp_m
            )
            best_before = best_dist_before[i]
            improved_best = (
                np.isfinite(best_before)
                and dist_after[i] < (best_before - self.anti_circling_min_progress_m)
            )

            if improved_best:
                self._steps_since_best_progress[i] = 0
                self._stalled_turn_steps[i] = 0
            else:
                if is_turning and is_far_from_wp:
                    self._steps_since_best_progress[i] += 1
                    self._stalled_turn_steps[i] += 1
                else:
                    self._steps_since_best_progress[i] = 0
                    self._stalled_turn_steps[i] = 0

            if (
                self._steps_since_best_progress[i] >= self.anti_circling_trigger_steps
                and self._breakout_steps_remaining[i] <= 0
            ):
                self._breakout_steps_remaining[i] = self.anti_circling_breakout_steps
                self._stalled_turn_steps[i] = 0
                self._steps_since_best_progress[i] = 0

    def _bearing_to_normalized_action(
        self,
        current_heading,
        target_bearing,
        max_turn_rate,
    ):
        heading_error = wrap_angle(target_bearing - current_heading)
        max_turn_this_step = max(max_turn_rate * self.dt, 1e-6)
        return float(np.clip(heading_error / max_turn_this_step, -1.0, 1.0))

    # ======================================================================= #
    # COLLISION HELPERS                                                       #
    # ======================================================================= #

    def _track_collision_violations(self):
        """Record pairs that breach caution/critical thresholds (always runs)."""
        n = len(self.aircraft_list)
        i_idx, j_idx = np.triu_indices(n, k=1)
        sep = self._dist_matrix[i_idx, j_idx]

        caution_mask  = sep < self.caution_dist
        critical_mask = sep < self.critical_dist

        for i, j in zip(i_idx[caution_mask], j_idx[caution_mask]):
            self.caution_dist_breakers.append(
                (self.aircraft_list[i].id_tag, self.aircraft_list[j].id_tag)
            )

        for i, j in zip(i_idx[critical_mask], j_idx[critical_mask]):
            self.crit_dist_breakers.append(
                (self.aircraft_list[i].id_tag, self.aircraft_list[j].id_tag)
            )
            # Flag crash so the episode terminates
            self._crashed = True

    def _calculate_collision_rewards(self, rewards_list):
        """
        Apply proximity penalties.

        Caution zone  – smooth linear penalty that grows as the pair closes.
                        Zero at the caution boundary; maximum at zero separation.
        Critical zone – large one-shot terminal penalty shared by both drones.
                        Applied once (via _crashed flag) rather than per-step so
                        the agent learns that reaching the critical distance is
                        catastrophic, not just expensive.
        """
        n = len(self.aircraft_list)
        i_idx, j_idx = np.triu_indices(n, k=1)
        sep = self._dist_matrix[i_idx, j_idx]

        # --- Caution zone: linear, graduated penalty ---
        caution_mask = sep < self.caution_dist
        caution_penalties = np.zeros_like(sep)
        caution_penalties[caution_mask] = (
            -self.penalty_caution_scale
            * (1.0 - sep[caution_mask] / self.caution_dist)
        )

        np.add.at(rewards_list, i_idx, caution_penalties)
        np.add.at(rewards_list, j_idx, caution_penalties)

        # --- Critical zone: large terminal penalty (applied once per crash) ---
        critical_mask = sep < self.critical_dist
        if critical_mask.any():
            # Distribute the crash penalty to every drone involved
            involved = set()
            for ci, cj in zip(i_idx[critical_mask], j_idx[critical_mask]):
                involved.add(ci)
                involved.add(cj)
            for idx in involved:
                rewards_list[idx] -= self.penalty_crash

    # ======================================================================= #
    # GEOMETRY HELPERS                                                        #
    # ======================================================================= #

    def _check_line_segment_arrival_local(self, a, b, p, radius):
        """
        Returns True if waypoint p lies within radius of the line segment a→b.
        Prevents high-speed UAVs from skipping over a waypoint between steps.
        """
        ap, ab = p - a, b - a
        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        return np.linalg.norm(p - (a + t * ab)) < radius

    # ======================================================================= #
    # BOUNDS & WAYPOINT HELPERS                                               #
    # ======================================================================= #

    def update_bounds(self, tl, br):
        self.min_lat, self.max_lat = sorted([tl[0], br[0]])
        self.min_lon, self.max_lon = sorted([tl[1], br[1]])
        self.transformer = CoordinateTransformer(self.min_lat, self.min_lon)
        l_r, n_r = self.max_lat - self.min_lat, self.max_lon - self.min_lon
        self.wp_min_lat = self.min_lat + l_r * self.boundary_margin
        self.wp_max_lat = self.max_lat - l_r * self.boundary_margin
        self.wp_min_lon = self.min_lon + n_r * self.boundary_margin
        self.wp_max_lon = self.max_lon - n_r * self.boundary_margin

    def _refill_mission(self, ac):
        """Top up a UAV's waypoint queue to mission_waypoint_count."""
        wm = ac.waypoint_manager
        needed = (
            self.mission_waypoint_count
            - (wm.queue_size() + (1 if wm.current_waypoint else 0))
        )
        for _ in range(max(0, int(needed))):
            wm.add_waypoint(Position(
                np.random.uniform(self.wp_min_lat, self.wp_max_lat),
                np.random.uniform(self.wp_min_lon, self.wp_max_lon)
            ))

    def _update_waypoint_local_cache(self, position_cache):
        """
        Keep waypoint targets aligned with the active mission state.

        Drones without an active waypoint use their current position as a
        placeholder so distance-to-waypoint features stay finite and zeroed.
        """
        n = len(self.aircraft_list)
        self._wp_local_cache[:n] = position_cache[:n]

        wp_indices, wp_lats, wp_lons = [], [], []
        for i, ac in enumerate(self.aircraft_list):
            wp = ac.waypoint_manager.current_waypoint
            if wp is not None:
                wp_indices.append(i)
                wp_lats.append(wp.latitude)
                wp_lons.append(wp.longitude)

        if wp_indices:
            wx, wy = self.transformer.geo_to_local(wp_lats, wp_lons)
            self._wp_local_cache[wp_indices, 0] = wx
            self._wp_local_cache[wp_indices, 1] = wy

    def _prime_best_distance_to_waypoint(self, position_cache):
        """
        Seed the best-distance baseline when a waypoint becomes active.

        This avoids awarding an infinite "first improvement" bonus on the first
        step after reset or waypoint advance.
        """
        for i, ac in enumerate(self.aircraft_list):
            if (
                ac.waypoint_manager.current_waypoint is not None
                and not np.isfinite(self._best_dist_to_wp[i])
            ):
                self._best_dist_to_wp[i] = np.linalg.norm(
                    self._wp_local_cache[i] - position_cache[i]
                )

    def _update_pairwise_distance_cache(self, position_cache):
        """Recompute pairwise separations for the active UAV set."""
        n = len(self.aircraft_list)
        if n == 0:
            return

        self._dx_matrix[:n, :n] = (
              position_cache[:n, None, 0]
            - position_cache[None, :n, 0]
        )
        self._dy_matrix[:n, :n] = (
              position_cache[:n, None, 1]
            - position_cache[None, :n, 1]
        )
        self._dist_matrix[:n, :n] = np.hypot(
            self._dx_matrix[:n, :n],
            self._dy_matrix[:n, :n]
        )

    def _sync_local_caches(self):
        """Batch-convert all UAV geo-positions and active waypoints to local metres."""
        n = len(self.aircraft_list)
        lats = [ac.position.latitude  for ac in self.aircraft_list]
        lons = [ac.position.longitude for ac in self.aircraft_list]
        x, y = self.transformer.geo_to_local(lats, lons)
        self._local_pos_cache[:n, 0], self._local_pos_cache[:n, 1] = x, y
        self._update_waypoint_local_cache(self._local_pos_cache)
        self._update_pairwise_distance_cache(self._local_pos_cache)

    # ======================================================================= #
    # RESET                                                                   #
    # ======================================================================= #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._crashed = False

        self.geofence_exit_counts = [0 for _ in range(self.max_uavs)]
        self._was_outside = [False for _ in range(self.max_uavs)]
        self._best_dist_to_wp[:] = np.inf
        self.safety_override_counts = [0 for _ in range(self.max_uavs)]
        self.anti_circling_counts = [0 for _ in range(self.max_uavs)]
        self._stalled_turn_steps[:] = 0
        self._breakout_steps_remaining[:] = 0
        self._steps_since_best_progress[:] = 0
        self._safety_turn_memory[:] = 0.0
        self._safety_hold_remaining[:] = 0

        # Clear safety logs
        self.caution_dist_breakers = []
        self.crit_dist_breakers = []

        for ac in self.aircraft_list:
            ac.distance_traveled = 0.0

            if self.clear_missions:
                ac.waypoint_manager.waypoint_queue.clear()
                ac.waypoint_manager.current_waypoint = None
                ac.waypoint_manager.reset()
            ac.waypoint_manager.hit_waypoints.clear()

            if self.mode == 'gen_mission':
                ac.position = Position(
                    np.random.uniform(self.min_lat, self.max_lat),
                    np.random.uniform(self.min_lon, self.max_lon)
                )
                turning_radius = max(
                    1.0,
                    np.random.uniform(
                        ac.base_turning_radius - ac.turning_variance,
                        ac.base_turning_radius + ac.turning_variance
                    )
                )
                cruise_speed = max(
                    1.0,
                    np.random.uniform(
                        ac.base_cruise_speed - ac.speed_variance,
                        ac.base_cruise_speed + ac.speed_variance
                    )
                )
                ac.set_flight_dynamics(turning_radius, cruise_speed)
                ac.heading = np.random.uniform(-np.pi, np.pi)

                if self.clear_missions:
                    self._refill_mission(ac)

            elif self.mode == 'manual_mission':
                ac.position = ac.initial_pos
                ac.heading = ac.initial_heading
                ac.set_flight_dynamics(
                    ac.base_turning_radius,
                    ac.base_cruise_speed,
                )

        # Rebuild coordinate caches
        self._sync_local_caches()
        self._prime_best_distance_to_waypoint(self._local_pos_cache)
        self._update_obs_buffer()

        return self._obs_buffer.copy(), {}

    # ======================================================================= #
    # METRICS                                                                 #
    # ======================================================================= #

    def get_uav_metrics(self):
        """Returns comprehensive telemetry and mission progress for all UAVs."""
        metrics = {
            "telemetry": [],
            "mission_stats": [],
            "safety_violations": {
                "caution": {
                    "total_count": len(self.caution_dist_breakers),
                    "pairs": self.caution_dist_breakers
                },
                "critical": {
                    "total_count": len(self.crit_dist_breakers),
                    "pairs": self.crit_dist_breakers
                },
                "geofence": {
                    "total_count": sum(self.geofence_exit_counts),
                    "counts": self.geofence_exit_counts
                },
                "hard_safety": {
                    "total_count": sum(self.safety_override_counts),
                    "counts": self.safety_override_counts
                },
                "anti_circling": {
                    "total_count": sum(self.anti_circling_counts),
                    "counts": self.anti_circling_counts
                }
            }
        }

        for i, ac in enumerate(self.aircraft_list):
            metrics["telemetry"].append({
                "id":      ac.id_tag,
                "pos":     (ac.position.latitude, ac.position.longitude),
                "speed":   ac.dynamics.cruise_speed,
                "heading": ac.heading,
                "mode":    ac.flight_mode
            })
            metrics["mission_stats"].append({
                "id":               ac.id_tag,
                "waypoints_reached": len(ac.waypoint_manager.hit_waypoints),
                "dist_navigating":  ac.distance_traveled,
            })

        return metrics
