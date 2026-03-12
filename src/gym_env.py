import gymnasium as gym
from gymnasium import spaces
import numpy as np
from flight_engine.helpers import Position, FlightMode
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
        inference_mode=False
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

        # ------------------------------------------------------------------ #
        # Reward / penalty coefficients                                      #
        # ------------------------------------------------------------------ #

        # --- Navigation rewards ---
        # Scale factor for distance-to-waypoint progress reward
        self.reward_progress_scale = 4.0
        # Scale factor for heading-alignment bonus (cos of heading error)      
        self.reward_heading_scale = 2.0 
        # Penalty coefficient on squared action magnitude (discourages large turns)      
        self.penalty_control_effort = 0.5 
        # Flat per-step penalty to encourage mission efficiency    
        self.penalty_timestep = 0.1           

        # --- Waypoint arrival bonus ---
        # Large sparse reward awarded on waypoint arrival
        self.reward_waypoint_hit = 3000.0     

        # --- Geofence / boundary penalty ---
        # Per-step penalty applied while a UAV is outside the geofence
        self.penalty_geofence = 50.0          

        # --- Collision penalties ---
        # Linear penalty scale for pairs within caution distance
        self.penalty_caution_scale = 5.0
        # Quadratic penalty scale for pairs within critical distance      
        self.penalty_critical_scale = 50.0    

        # --- Geofence violation tracking ---
        # counts distinct exit events per UAV, not sustained frames
        self.geofence_exit_counts = [0 for _ in range(self.max_uavs)]
        self._was_outside = [False for _ in range(self.max_uavs)]

        self.update_bounds(tl, br)
        # Number of observation features per UAV: 
        #   10 self-features + 2 neighbours × 6 features each
        self.obs_per_uav = 22
        
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

    def step(self, actions):
        self.current_step += 1
        n = len(self.aircraft_list)
        actions = np.asarray(actions)[:n]

        # Skip reward calculations in inference mode
        if self.inference_mode:
            rewards = 0.0
        else:
            rewards = np.zeros(n)

        # --- 1. Batch Transformation: Pre-move state & Waypoints ---
        # Convert all UAV geo-positions to local
        lats = [ac.position.latitude for ac in self.aircraft_list]
        lons = [ac.position.longitude for ac in self.aircraft_list]
        
        x_vals, y_vals = self.transformer.geo_to_local(lats, lons)
        self._prev_local_cache[:n, 0] = x_vals
        self._prev_local_cache[:n, 1] = y_vals

        # Collect waypoint coordinates for all UAVs with an active waypoint
        wp_indices = []
        wp_lats, wp_lons = [], []
        for i, ac in enumerate(self.aircraft_list):
            wp = ac.waypoint_manager.current_waypoint
            if wp:
                wp_indices.append(i)
                wp_lats.append(wp.latitude)
                wp_lons.append(wp.longitude)
        
        if wp_indices:
            wx, wy = self.transformer.geo_to_local(wp_lats, wp_lons)
            self._wp_local_cache[wp_indices, 0] = wx
            self._wp_local_cache[wp_indices, 1] = wy

        # Capture distance-to-waypoint before physics update for reward
        if not self.inference_mode:
            dist_before = np.linalg.norm(
                self._wp_local_cache[:n] - self._prev_local_cache[:n], 
                axis=1
            )

        # --- 2. Physics update ---
        for i, ac in enumerate(self.aircraft_list):
            if ac.flight_mode == FlightMode.LOITERING:
                # Loitering UAVs orbit their loiter point
                ac._update_loiter(
                    *self._prev_local_cache[i], 
                    self.dt, self.transformer
                )
            else:
                # Scale normalised action
                ac.update_simple(
                    actions[i] * ac.dynamics.max_turn_rate, 
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
        # Pairwise distances are always computed
        # penalties are only applied during training and non-inference.
        if n > 1:
            self._dx_matrix[:n, :n] = (
                  self._local_pos_cache[:n, None, 0] 
                - self._local_pos_cache[None, :n, 0]
            )
            self._dy_matrix[:n, :n] = (
                  self._local_pos_cache[:n, None, 1] 
                - self._local_pos_cache[None, :n, 1]
            )
            self._dist_matrix[:n, :n] = np.hypot(
                self._dx_matrix[:n, :n], 
                self._dy_matrix[:n, :n]
            )
            
            # Track violations for metrics (always)
            self._track_collision_violations()
            
            # Apply penalties only in training mode
            if not self.inference_mode:
                self._calculate_collision_rewards(rewards)

        # --- 5. Nav Rewards & Waypoint Management ---
        if not self.inference_mode:
            dist_after = np.linalg.norm(
                self._wp_local_cache[:n] - self._local_pos_cache[:n], 
                axis=1
            )
        
        for i, ac in enumerate(self.aircraft_list):
            if not self.inference_mode:
                # Bearing from current position to active waypoint
                dx_wp = (
                      self._wp_local_cache[i, 0] 
                    - self._local_pos_cache[i, 0]
                )
                dy_wp = (
                      self._wp_local_cache[i, 1] 
                    - self._local_pos_cache[i, 1]
                )
                hdg_to_wp = np.arctan2(dx_wp, dy_wp)
                # Signed heading error in [-π, π]
                hdg_err = np.abs(
                    (hdg_to_wp - ac.heading + np.pi) % (2 * np.pi) - np.pi
                )

                # Progress reward: positive when closing distance to waypoint
                rewards[i] += (
                      (dist_before[i] - dist_after[i]) 
                    * self.reward_progress_scale
                )
                # Alignment bonus: maximum when heading directly toward waypoint
                rewards[i] += (
                      np.cos(hdg_err) 
                    * self.reward_heading_scale
                )
                # Control effort penalty: discourages unnecessarily large turn commands
                rewards[i] -= self.penalty_control_effort * (actions[i] ** 2)
                # Constant per-step cost to encourage reaching waypoints quickly
                rewards[i] -= self.penalty_timestep

            wm = ac.waypoint_manager
            if wm.current_waypoint:
                # Check arrival using line-segment proximity
                if self._check_line_segment_arrival_local(
                    self._prev_local_cache[i], self._local_pos_cache[i], 
                    self._wp_local_cache[i], wm.arrival_threshold
                ):
                    if not self.inference_mode:
                        rewards[i] += self.reward_waypoint_hit
                    wm.advance()
                    if self.mode == 'gen_mission': self._refill_mission(ac)
                    wm.hit_waypoints.append(wm.current_waypoint)
                
                if not wm.has_waypoints(): ac._enter_loiter()

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

            # Per-step geofence penalty (training only)
            if not self.inference_mode and not inside:
                rewards[i] -= self.penalty_geofence

        self._update_obs_buffer() 

        done = self.current_step >= self.max_steps or \
            all(
                not ac.waypoint_manager.has_waypoints() for \
                    ac in self.aircraft_list
            )
        
        total_reward = float(rewards.sum()) if \
            not self.inference_mode else 0.0
        
        return \
            self._obs_buffer.copy(), \
            total_reward, \
            done, \
            False, \
            {
                "waypoints_hit": sum(
                    len(ac.waypoint_manager.hit_waypoints) for \
                        ac in self.aircraft_list
                    )
            }

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
            [1.0 if ac.waypoint_manager.has_waypoints() else \
                0.0 for ac in self.aircraft_list], 
            dtype=np.float32
        )

        # Vector and scalar distance/bearing to current waypoint for each UAV
        d_wp = self._wp_local_cache[:n] - self._local_pos_cache[:n]
        dist_wp = np.linalg.norm(d_wp, axis=1)
        brg_wp = np.arctan2(d_wp[:, 0], d_wp[:, 1])
        # Relative bearing: 0 when waypoint is dead ahead, ±π when directly behind
        rel_brg_wp = ((brg_wp - headings + np.pi) % (2 * np.pi)) - np.pi

        # Mask self-distances on the diagonal so they don't pollute nearest-neighbour lookup
        masked_dist = self._dist_matrix[:n, :n].copy()
        np.fill_diagonal(masked_dist, np.inf)
        
        obs_temp = np.zeros(
            (self.max_uavs, self.obs_per_uav), 
            dtype=np.float32
        )
        
        for i in range(n):
            # --- Self features (10 dims) ---
            self_feat = [
                np.clip(dist_wp[i] / 1000.0, 0, 1.0),   # Normalised distance to waypoint
                np.sin(rel_brg_wp[i]), np.cos(rel_brg_wp[i]),  # Sin/cos of relative bearing to WP
                np.sin(headings[i]), np.cos(headings[i]),       # Sin/cos of absolute heading
                speeds[i] / 30.0,                               # Normalised cruise speed
                has_wp[i], 0.0, 0.0, 0.0                        # Waypoint-present flag + 3 reserved dims
            ]
            
            # --- Neighbour features (2 neighbours × 6 dims = 12 dims total) ---
            neigh_feat = []
            # Sort other UAVs by ascending distance so slot 0 is always the closest
            sorted_neighbors = np.argsort(masked_dist[i])
            
            for neighbor_slot in range(2):
                if neighbor_slot < (n - 1):
                    nb_idx = sorted_neighbors[neighbor_slot]
                    dist = masked_dist[i, nb_idx]
                    
                    dx = self._dx_matrix[i, nb_idx]
                    dy = self._dy_matrix[i, nb_idx]
                    other_ac = self.aircraft_list[nb_idx]
                    
                    # Velocity components for closing-speed calculation
                    v1_x = speeds[i] * np.sin(headings[i])
                    v1_y = speeds[i] * np.cos(headings[i])
                    v2_x = other_ac.dynamics.cruise_speed * np.sin(other_ac.heading)
                    v2_y = other_ac.dynamics.cruise_speed * np.cos(other_ac.heading)
                    
                    # Radial closing speed (positive = converging); used to compute time-to-collision
                    v_closing = ((v1_x - v2_x) * dx + (v1_y - v2_y) * dy) / (dist + 1e-6)
                    ttc = np.clip(
                        (dist / v_closing if v_closing > 0 else 50.0) / 50.0, 
                        0, 
                        1.0
                    )
                    
                    # Bearing to neighbour and relative heading difference (both in [-π, π])
                    rel_brg = ((np.arctan2(dx, dy) - headings[i] + np.pi) % (2 * np.pi)) - np.pi
                    rel_hdg = ((other_ac.heading - headings[i] + np.pi) % (2 * np.pi)) - np.pi
                    
                    neigh_feat.extend(
                        [
                            np.clip(dist / 500.0, 0, 1.0),  # Normalised separation distance
                            ttc,                             # Normalised time-to-collision
                            np.sin(rel_brg),                 # Sin of bearing to neighbour
                            np.cos(rel_brg),                 # Cos of bearing to neighbour
                            np.sin(rel_hdg),                 # Sin of relative heading
                            np.cos(rel_hdg)                  # Cos of relative heading
                        ]
                    )
                else:
                    # Pad empty neighbour slots with "no threat" sentinel values
                    neigh_feat.extend([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            
            obs_temp[i] = np.concatenate([self_feat, neigh_feat])

        self._obs_buffer[:] = obs_temp.flatten()

    def _track_collision_violations(self):
        """Record pairs that breach caution/critical separation thresholds (always runs, including inference)."""
        n = len(self.aircraft_list)
        i_idx, j_idx = np.triu_indices(n, k=1)
        sep = self._dist_matrix[i_idx, j_idx]
        
        caution_mask = sep < self.caution_dist
        critical_mask = sep < self.critical_dist

        for i, j in zip(i_idx[caution_mask], j_idx[caution_mask]):
            self.caution_dist_breakers.append(
                (self.aircraft_list[i].id_tag, self.aircraft_list[j].id_tag)
            )

        for i, j in zip(i_idx[critical_mask], j_idx[critical_mask]):
            self.crit_dist_breakers.append(
                (self.aircraft_list[i].id_tag, self.aircraft_list[j].id_tag)
            )

    def _calculate_collision_rewards(self, rewards_list):
        """Apply proximity penalties to both aircraft in each violating pair (training mode only)."""
        n = len(self.aircraft_list)
        i_idx, j_idx = np.triu_indices(n, k=1)
        sep = self._dist_matrix[i_idx, j_idx]
        
        penalties = np.zeros_like(sep)

        # Caution zone: linear penalty that grows as separation shrinks toward zero
        caution_mask = sep < self.caution_dist
        penalties[caution_mask] += \
            -self.penalty_caution_scale * (1.0 - (sep[caution_mask] / self.caution_dist))

        # Critical zone: quadratic penalty for near-collision; stacks on top of caution penalty
        critical_mask = sep < self.critical_dist
        penalties[critical_mask] += \
            -self.penalty_critical_scale * (1.0 - (sep[critical_mask] / self.critical_dist))**2

        # Distribute penalties symmetrically to both aircraft in each pair
        np.add.at(rewards_list, i_idx, penalties)
        np.add.at(rewards_list, j_idx, penalties)

    def _check_line_segment_arrival_local(self, a, b, p, radius):
        """
        Returns True if the waypoint p lies within radius of the line segment a→b.
        Using a line-segment test rather than a point test prevents high-speed UAVs
        from 'skipping over' a waypoint between consecutive steps.
        """
        ap, ab = p - a, b - a
        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        return np.linalg.norm(p - (a + t * ab)) < radius

    def update_bounds(self, tl, br):
        self.min_lat, self.max_lat = sorted([tl[0], br[0]])
        self.min_lon, self.max_lon = sorted([tl[1], br[1]])
        self.transformer = CoordinateTransformer(self.min_lat, self.min_lon)
        l_r, n_r = self.max_lat - self.min_lat, self.max_lon - self.min_lon
        # Shrink the valid waypoint region by boundary_margin to keep UAVs away from the geofence edge
        self.wp_min_lat, self.wp_max_lat = \
            self.min_lat + l_r*self.boundary_margin, \
            self.max_lat - l_r*self.boundary_margin
        self.wp_min_lon, self.wp_max_lon = \
            self.min_lon + n_r*self.boundary_margin, \
            self.max_lon - n_r*self.boundary_margin

    def _refill_mission(self, ac):
        """Top up a UAV's waypoint queue to mission_waypoint_count, sampling within the geofence margins."""
        wm = ac.waypoint_manager
        needed =     self.mission_waypoint_count \
                   - (wm.queue_size() + (1 if wm.current_waypoint else 0))
        for _ in range(max(0, int(needed))):
            wm.add_waypoint(
                Position(
                    np.random.uniform(self.wp_min_lat, self.wp_max_lat), 
                    np.random.uniform(self.wp_min_lon, self.wp_max_lon)))

    def _sync_local_caches(self):
        """
        Batch-convert all UAV geo-positions and active waypoints into local metre coordinates.
        Called at reset() and any time the caches need to be rebuilt from scratch.
        """
        n = len(self.aircraft_list)
        lats = [ac.position.latitude for ac in self.aircraft_list]
        lons = [ac.position.longitude for ac in self.aircraft_list]
        x, y = self.transformer.geo_to_local(lats, lons)
        self._local_pos_cache[:n, 0], self._local_pos_cache[:n, 1] = x, y

        wp_indices, wp_lats, wp_lons = [], [], []
        for i, ac in enumerate(self.aircraft_list):
            wp = ac.waypoint_manager.current_waypoint
            if wp:
                wp_indices.append(i)
                wp_lats.append(wp.latitude)
                wp_lons.append(wp.longitude)
        
        if wp_indices:
            wx, wy = self.transformer.geo_to_local(wp_lats, wp_lons)
            self._wp_local_cache[wp_indices, 0] = wx
            self._wp_local_cache[wp_indices, 1] = wy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.geofence_exit_counts = [0 for _ in range(self.max_uavs)]
        self._was_outside = [False for _ in range(self.max_uavs)]

        # Clear safety logs
        self.caution_dist_breakers = []
        self.crit_dist_breakers = []

        for ac in self.aircraft_list:
            ac.distance_traveled = 0.0

            if self.clear_missions:
                # Only clear queues during training resets
                ac.waypoint_manager.waypoint_queue.clear()
                ac.waypoint_manager.current_waypoint = None
                ac.waypoint_manager.reset()
            ac.waypoint_manager.hit_waypoints.clear()

            if self.mode == 'gen_mission':
                # Randomize state
                ac.position = Position(
                    np.random.uniform(self.min_lat, self.max_lat),
                    np.random.uniform(self.min_lon, self.max_lon)
                )

                ac.turning_radius = np.random.uniform(
                    ac.turning_radius - ac.turning_variance,
                    ac.turning_radius + ac.turning_variance
                )

                ac.dynamics.cruise_speed = np.random.uniform(
                    ac.dynamics.cruise_speed - ac.speed_variance,
                    ac.dynamics.cruise_speed + ac.speed_variance
                )

                ac.heading = np.random.uniform(-np.pi, np.pi)

                if self.clear_missions:
                    self._refill_mission(ac)

            elif self.mode == 'manual_mission':
                ac.position = ac.initial_pos

        # rebuild coordinate caches
        self._sync_local_caches()
        self._update_obs_buffer()

        return self._obs_buffer.copy(), {}
    
    def get_uav_metrics(self):
        """
        Returns comprehensive telemetry and mission progress for all UAVs.
        """
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
                }
                
            }
        }

        for i, ac in enumerate(self.aircraft_list):
            # 1. Position and Heading
            metrics["telemetry"].append({
                "id": ac.id_tag,
                "pos": (ac.position.latitude, ac.position.longitude),
                "speed": ac.dynamics.cruise_speed,
                "heading": ac.heading,
                "mode": ac.flight_mode
            })

            # 2. Waypoints and Distance
            # Note: This assumes your aircraft objects have a 'distance_traveled' 
            # attribute. If not, you can track this in the step() function.
            metrics["mission_stats"].append({
                "id": ac.id_tag,
                "waypoints_reached": len(ac.waypoint_manager.hit_waypoints),
                "dist_navigating": ac.distance_traveled,
            })

        return metrics