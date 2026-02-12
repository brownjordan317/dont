import gymnasium as gym
from gymnasium import spaces
import numpy as np
from geopy.distance import geodesic
from itertools import combinations

# Local imports
from flight_engine.helpers import wrap_angle, Position, FlightMode
from flight_engine.trans_coorders import CoordinateTransformer

class MultiUAVEnv(gym.Env):
    def __init__(
            self, 
            aircraft_list, 
            tl, br, 
            dt=0.3, 
            max_steps=10_000,
            boundary_margin=0.15,
            mission_waypoint_count=3,
            mode='gen_mission', #gen_mission or manual_mission
            caution_dist=100.0,
            critical_dist=35.0
        ):
        super().__init__()
        self.aircraft_list = aircraft_list
        self.max_uavs = 5
        self.dt = dt
        self.max_steps = max_steps
        self.boundary_margin = boundary_margin
        self.mission_waypoint_count = mission_waypoint_count
        self.mode = mode
        self.caution_dist = caution_dist
        self.critical_dist = critical_dist
        
        # Telemetry
        self.update_bounds(tl, br)

        # Observation: [Nav(10), Neighbors(12)]
        self.obs_per_uav = 22
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(self.max_uavs * self.obs_per_uav,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.max_uavs,), 
            dtype=np.float32
        )
        self.current_step = 0

    def update_bounds(self, tl, br):
        self.min_lat, self.max_lat = sorted([tl[0], br[0]])
        self.min_lon, self.max_lon = sorted([tl[1], br[1]])
        self.transformer = CoordinateTransformer(self.min_lat, self.min_lon)
        
        lat_range, lon_range = \
            self.max_lat - self.min_lat, self.max_lon - self.min_lon
        self.wp_min_lat = self.min_lat + (lat_range * self.boundary_margin)
        self.wp_max_lat = self.max_lat - (lat_range * self.boundary_margin)
        self.wp_min_lon = self.min_lon + (lon_range * self.boundary_margin)
        self.wp_max_lon = self.max_lon - (lon_range * self.boundary_margin)

    def _refill_mission(self, ac):
        current_total = ac.waypoint_manager.queue_size() \
                    + (1 if ac.waypoint_manager.current_waypoint else 0)
        needed = self.mission_waypoint_count - current_total
        for _ in range(max(0, int(needed))):
            wp = Position(
                np.random.uniform(self.wp_min_lat, self.wp_max_lat),
                np.random.uniform(self.wp_min_lon, self.wp_max_lon)
            )
            ac.waypoint_manager.add_waypoint(wp)

    def _check_line_segment_arrival(self, p1_geo, p2_geo, wp_geo, radius):
        a = np.array(self.transformer.geo_to_local(p1_geo[0], p1_geo[1]))
        b = np.array(self.transformer.geo_to_local(p2_geo[0], p2_geo[1]))
        p = np.array(self.transformer.geo_to_local(wp_geo[0], wp_geo[1]))
        
        ap = p - a
        ab = b - a
        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        closest_point = a + t * ab
        distance_to_segment = np.linalg.norm(p - closest_point)
        return distance_to_segment < radius

    def _get_neighbor_obs(self, subject_idx, num_neighbors=2):
        neighbor_features = []
        subject_ac = self.aircraft_list[subject_idx]
        
        v1_x = subject_ac.dynamics.cruise_speed * np.sin(subject_ac.heading)
        v1_y = subject_ac.dynamics.cruise_speed * np.cos(subject_ac.heading)
        
        others = []
        for i, ac in enumerate(self.aircraft_list):
            if i == subject_idx: continue
            dist = geodesic(
                subject_ac.position.to_tuple(), 
                ac.position.to_tuple()).meters
            others.append((dist, ac))
            
        others.sort(key=lambda x: x[0])
        
        sensing_radius = 500.0
        for i in range(num_neighbors):
            if i < len(others):
                dist, other_ac = others[i]
                v2_x = other_ac.dynamics.cruise_speed * np.sin(
                    other_ac.heading
                )
                v2_y = other_ac.dynamics.cruise_speed * np.cos(
                    other_ac.heading
                )
                rvx, rvy = v1_x - v2_x, v1_y - v2_y
                
                p1 = self.transformer.geo_to_local(
                    *subject_ac.position.to_tuple()
                )
                p2 = self.transformer.geo_to_local(
                    *other_ac.position.to_tuple()
                )
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                
                v_closing = (rvx * dx + rvy * dy) / (dist + 1e-6)
                ttc = dist / v_closing if v_closing > 0 else 50.0 
                
                brg = np.arctan2(dy, dx)
                rel_brg = wrap_angle(brg - subject_ac.heading)
                rel_hdg = wrap_angle(other_ac.heading - subject_ac.heading)
                
                neighbor_features.extend([
                    np.clip(dist / sensing_radius, 0, 1.0),
                    np.clip(ttc / 50.0, 0, 1.0),
                    np.sin(rel_brg), np.cos(rel_brg),
                    np.sin(rel_hdg), np.cos(rel_hdg)
                ])
            else:
                neighbor_features.extend([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        return neighbor_features

    def _get_uav_obs(self, idx):
        if idx >= len(self.aircraft_list): 
            return np.zeros(self.obs_per_uav)
            
        ac = self.aircraft_list[idx]
        
        # FIX: Removed the "if LOITERING return zeros" check. 
        # Drones must always be visible to others.
        
        wp = ac.waypoint_manager.current_waypoint
        if wp:
            dist_wp = geodesic(
                ac.position.to_tuple(), 
                wp.to_tuple()
            ).meters
            brg_wp = np.arctan2(
                wp.longitude - ac.position.longitude, 
                wp.latitude - ac.position.latitude
            )
            rel_brg_wp = wrap_angle(brg_wp - ac.heading)
        else:
            dist_wp, rel_brg_wp = 1000.0, 0.0
        
        nav_obs = [
            np.clip(dist_wp / 1000.0, 0, 1.0), 
            np.sin(rel_brg_wp), np.cos(rel_brg_wp),
            np.sin(ac.heading), np.cos(ac.heading),
            ac.dynamics.cruise_speed / 30.0,
            1.0 if ac.waypoint_manager.has_waypoints() else 0.0,
            0.0, 0.0, 0.0 
        ]
        
        neighbor_obs = self._get_neighbor_obs(idx, num_neighbors=2)

        return np.array(nav_obs + neighbor_obs, dtype=np.float32)
    
    def _calculate_collision_rewards(self, rewards_list):
        # FIX: Smoother gradients. We use a linear penalty that is less likely to cause circling.
        for i1, i2 in combinations(range(len(self.aircraft_list)), 2):
            ac1, ac2 = self.aircraft_list[i1], self.aircraft_list[i2]
            sep = geodesic(
                ac1.position.to_tuple(), 
                ac2.position.to_tuple()
            ).meters
            
            if sep < self.caution_dist:
                # Severity 0.0 at caution_dist, 1.0 at 0m
                severity = 1.0 - (sep / self.caution_dist)
                penalty = -5.0 * severity  # Scaled down from -2000
                
                if sep < self.critical_dist:
                    # Quadratic spike only at very close range
                    danger_severity = (1.0 - (sep / self.critical_dist)) ** 2
                    penalty += -50.0 * danger_severity
                
                rewards_list[i1] += penalty
                rewards_list[i2] += penalty
        return rewards_list
    
    def step(self, actions):
        self.current_step += 1
        uav_rewards = [0.0] * len(self.aircraft_list)

        for i, ac in enumerate(self.aircraft_list):
            # Maintain the original loiter update logic
            if ac.flight_mode == FlightMode.LOITERING:
                lx, ly = self.transformer.geo_to_local(
                    ac.position.latitude, 
                    ac.position.longitude
                )
                ac._update_loiter(lx, ly, self.dt, self.transformer)
                continue

            wp = ac.waypoint_manager.current_waypoint
            if not wp: continue

            pos_prev = ac.position.to_tuple()
            dist_before = geodesic(pos_prev, wp.to_tuple()).meters
            
            # Physics Update
            turn_rate = actions[i] * ac.dynamics.max_turn_rate
            ac.heading = wrap_angle(ac.heading + (turn_rate * self.dt))
            dist_moved = ac.dynamics.cruise_speed * self.dt
            
            dx, dy = dist_moved * np.sin(ac.heading), dist_moved * np.cos(ac.heading)
            curr_x, curr_y = self.transformer.geo_to_local(
                pos_prev[0], 
                pos_prev[1]
            )
            new_lat, new_lon = self.transformer.local_to_geo(
                curr_x + dx, 
                curr_y + dy
            )
            ac.position = Position(new_lat, new_lon)
            pos_curr = ac.position.to_tuple()

            # Sequential Arrival Logic (Segment Check)
            arrived = self._check_line_segment_arrival(
                pos_prev, 
                pos_curr, 
                wp.to_tuple(), 
                ac.waypoint_manager.arrival_threshold
            )

            # Reward Shaping
            dist_after = geodesic(pos_curr, wp.to_tuple()).meters
            hdg_to_wp = np.arctan2(wp.longitude-new_lon, wp.latitude-new_lat)
            hdg_err = abs(wrap_angle(hdg_to_wp - ac.heading))
            
            # Efficiency Rewards (Dense)
            uav_rewards[i] = ((dist_before - dist_after) * 4.0) + (np.cos(hdg_err) * 2.0)
            uav_rewards[i] -= 0.5 * (actions[i]**2) # Smoothness
            uav_rewards[i] -= 0.1 # Step penalty
            
            if arrived:
                uav_rewards[i] += 3000.0 # High bonus for arrival
                ac.waypoint_manager.advance()
                if self.mode == 'gen_mission':
                    self._refill_mission(ac)
                ac.waypoint_manager.hit_waypoints.append(wp)

            if not ac.waypoint_manager.has_waypoints():
                ac._enter_loiter()

            # Geofence Penalty
            if not (self.min_lat < new_lat < self.max_lat) or not (self.min_lon < new_lon < self.max_lon):
                uav_rewards[i] -= 50.0

        if len(self.aircraft_list) > 1:
            uav_rewards = self._calculate_collision_rewards(uav_rewards)

        done = self.current_step >= self.max_steps or \
               all(not ac.waypoint_manager.has_waypoints() for ac in self.aircraft_list)

        obs = np.concatenate([self._get_uav_obs(j) for j in range(self.max_uavs)]).astype(np.float32)

        # Return the sum of rewards as requested, but we've balanced the individual components.
        return obs, sum(uav_rewards), done, False, \
            {"waypoints_hit": sum(len(ac.waypoint_manager.hit_waypoints) for ac in self.aircraft_list)}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        for ac in self.aircraft_list:
            if self.mode == 'gen_mission':
                ac.position = Position(
                    np.random.uniform(self.min_lat, self.max_lat), 
                    np.random.uniform(self.min_lon, self.max_lon)
                )
                ac.heading = np.random.uniform(-np.pi, np.pi)
                ac.waypoint_manager.waypoint_queue.clear()
                ac.waypoint_manager.current_waypoint = None
                self._refill_mission(ac)
            ac.waypoint_manager.hit_waypoints.clear()
        
        obs = np.concatenate([self._get_uav_obs(j) for j in range(self.max_uavs)]).astype(np.float32)
        return obs, {}