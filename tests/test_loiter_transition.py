import unittest

from flight_engine.helpers import FlightMode, Position
from flight_engine.simulator import FixedWingAircraft
from flight_engine.trans_coorders import CoordinateTransformer


class LoiterTransitionTests(unittest.TestCase):
    def setUp(self):
        self.transformer = CoordinateTransformer(37.0, -122.0)

    def _make_aircraft(self) -> FixedWingAircraft:
        return FixedWingAircraft(
            id_tag="UAV-1",
            initial_position=Position(37.0, -122.0),
            initial_heading=0.0,
            cruise_speed=25.0,
            turning_radius=50.0,
            turn_response_time_s=0.5,
        )

    def test_loiter_preserves_existing_turn_direction(self):
        aircraft = self._make_aircraft()
        max_turn_rate = float(aircraft.dynamics.max_turn_rate)
        aircraft.actual_turn_rate = -0.5 * max_turn_rate
        aircraft.desired_turn_rate = -0.5 * max_turn_rate

        aircraft._enter_loiter()

        self.assertEqual(aircraft.flight_mode, FlightMode.LOITERING)
        self.assertLess(aircraft.desired_turn_rate, 0.0)

        aircraft._update_loiter(0.3, self.transformer)

        self.assertLess(aircraft.actual_turn_rate, 0.0)
        self.assertGreater(abs(aircraft.actual_turn_rate), 0.5 * max_turn_rate)
        self.assertLess(abs(aircraft.actual_turn_rate), max_turn_rate)

    def test_loiter_uses_last_commanded_direction_when_current_turn_is_flat(self):
        aircraft = self._make_aircraft()
        max_turn_rate = float(aircraft.dynamics.max_turn_rate)
        aircraft.actual_turn_rate = 0.0
        aircraft.desired_turn_rate = -0.25 * max_turn_rate

        aircraft._enter_loiter()

        self.assertLess(aircraft.desired_turn_rate, 0.0)

        aircraft._update_loiter(0.3, self.transformer)

        self.assertLess(aircraft.actual_turn_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
