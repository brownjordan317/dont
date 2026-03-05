import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from flight_engine.simulator import FixedWingAircraft

# ============================================================================
# VISUALIZATION
# ============================================================================

class FlightVisualizer:
    """Handles visualization of multiple aircraft"""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 11)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.arrow_length = 0.00008
    
    def plot(self, aircraft_list: List[FixedWingAircraft], waypoint_pool=None):
        """Plot all aircraft"""
        self.ax.clear()
        
        # Draw the global "Job Board" waypoints
        if waypoint_pool:
            for wp in waypoint_pool:
                self.ax.scatter(wp.longitude, wp.latitude, c='gold', s=100, 
                            marker='*', label='Global Waypoint', edgecolors='black')

        for aircraft in aircraft_list:
            self._plot_aircraft(aircraft)
        
        self._finalize_plot(len(aircraft_list))

        self._finalize_plot(len(aircraft_list))
    
    def _plot_aircraft(self, aircraft: FixedWingAircraft):
        """Plot single aircraft"""
        color = aircraft.color
        
        # Trajectory
        if len(aircraft.path_history) > 1:
            path = np.array([p.to_tuple() for p in aircraft.path_history])
            self.ax.plot(path[:, 1], path[:, 0], '-', color=color, 
                        linewidth=2, alpha=0.6, label=f'{aircraft.id_tag} path')
        
        # Loiter center
        if aircraft.loiter_center:
            self.ax.scatter(aircraft.loiter_center.longitude, 
                          aircraft.loiter_center.latitude,
                          c=color, s=150, marker='o', alpha=0.3,
                          edgecolors=color, linewidths=2, zorder=3)
        
        # Current waypoint
        if aircraft.waypoint_manager.current_waypoint:
            wp = aircraft.waypoint_manager.current_waypoint
            self.ax.scatter(wp.longitude, wp.latitude, c=color, s=250,
                          marker='*', zorder=5, edgecolors='black', 
                          linewidths=1.5, alpha=0.8)
        
        # Queued waypoints
        if aircraft.waypoint_manager.waypoint_queue:
            queued = [wp.to_tuple() for wp in aircraft.waypoint_manager.waypoint_queue]
            queued_array = np.array(queued)
            self.ax.scatter(queued_array[:, 1], queued_array[:, 0],
                          c=color, s=120, marker='*', zorder=4,
                          alpha=0.5, edgecolors='black', linewidths=0.5)
            
            # Waypoint path
            if aircraft.waypoint_manager.current_waypoint:
                all_wps = [aircraft.waypoint_manager.current_waypoint] + \
                         list(aircraft.waypoint_manager.waypoint_queue)
                wp_array = np.array([wp.to_tuple() for wp in all_wps])
                self.ax.plot(wp_array[:, 1], wp_array[:, 0], '--',
                           color=color, alpha=0.3, linewidth=1)
        
        # Current position
        pos = aircraft.position
        self.ax.scatter(pos.longitude, pos.latitude, c=color, s=180,
                       marker='o', zorder=6, edgecolors='black', linewidths=2)
        
        # Heading arrow
        self.ax.arrow(pos.longitude, pos.latitude,
                     self.arrow_length * np.cos(aircraft.heading),
                     self.arrow_length * np.sin(aircraft.heading),
                     color=color, width=0.000008, head_width=0.00003,
                     zorder=6, length_includes_head=True,
                     edgecolor='black', linewidth=0.5)
        
        # Label
        self.ax.text(pos.longitude + 0.00005, pos.latitude + 0.00005,
                    aircraft.id_tag, fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                            alpha=0.7, edgecolor='black'))
    
    def _finalize_plot(self, num_aircraft: int):
        """Finalize plot with labels and formatting"""
        self.ax.set_xlabel("Longitude", fontsize=11)
        self.ax.set_ylabel("Latitude", fontsize=11)
        self.ax.set_title(
            f"Multi-UAV Fixed-Wing Simulation ({num_aircraft} aircraft)",
            fontsize=13, fontweight='bold'
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='best', fontsize=9)
        self.ax.axis('equal')
        plt.tight_layout()
        plt.pause(0.001)
    
    def show(self):
        """Show the final plot"""
        plt.show()

    def save(self, filename: str):
        """Save the current figure to a file"""
        self.fig.savefig(filename, dpi=300)