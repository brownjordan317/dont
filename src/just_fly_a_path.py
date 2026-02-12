import numpy as np
import random
import os
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position, FlightMode
from flight_engine.visualizer import FlightVisualizer

def generate_random_waypoint(origin, radius_miles=0.125):
    """
    Generates a random Position within a radius in miles.
    0.25 miles is approx 400 meters (~0.0036 degrees).
    """
    # Square root ensures uniform distribution across the area
    r = (radius_miles * 0.0144) * np.sqrt(random.random())
    theta = random.random() * 2 * np.pi
    
    return Position(
        origin.latitude + r * np.cos(theta),
        origin.longitude + r * np.sin(theta)
    )

def test_physics_with_visuals():
    print("Initializing Multi-Flyer Physics Test...")
    
    # Directory for output frames if you want to make a gif later
    os.makedirs("physics_frames", exist_ok=True)
    
    origin = Position(47.3977, 8.5455)
    viz = FlightVisualizer()
    
    # 1. Initialize 3 Flyers with different specs
    uavs = [
        FixedWingAircraft("UAV-ALPHA", origin, 0.0, 15.0, 25.0, color='blue'),
        FixedWingAircraft("UAV-BRAVO", origin, np.pi/2, 12.0, 20.0, color='red'),
        FixedWingAircraft("UAV-CHARLIE", origin, np.pi, 10.0, 15.0, color='green')
    ]
    
    # 2. Assign multiple random waypoints to each
    for ac in uavs:
        ac.flight_mode = FlightMode.NAVIGATING
        wps = [generate_random_waypoint(origin, 0.25) for _ in range(10)]
        ac.add_waypoints(wps)

    # 3. Simulation Loop
    dt = 0.9
    total_steps = 1500
    
    for i in range(total_steps):
        for ac in uavs:
            ac.update(dt)  # simulator.py handles waypoint arrival internally

        # 4. Generate Visuals every 5 steps
        if i % 5 == 0:
            viz.plot(uavs)
            viz.save(f"physics_frames/step_{i:04d}.png")
            print(f"Step {i:4d}: Alpha at {uavs[0].distance_traveled:.1f}m")

    # Save final flight path
    viz.plot(uavs)
    viz.save("final_physics_test.png")
    print("\nTest Complete. Final plot saved as 'final_physics_test.png'")

if __name__ == "__main__":
    test_physics_with_visuals()