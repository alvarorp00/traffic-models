"""
This module runs the simulation, using the Engine class
and the configuration specified in the Config module.
"""

from lib.engine import Engine, RunConfig
from config import POPULATION_SIZE, ROAD_LENGTH, MAX_SPEED,\
        TIME_STEPS, N_LANES, LANES_PRIORITY, LANES_DENSITY,\
        SAFE_DISTANCE
# import lib.graphics as graphics
import tests


def run():
    """
    Runs the simulation
    """

    run_config = RunConfig(
        population_size=POPULATION_SIZE,
        time_steps=0,  # Don't evolve the system for now
        road_length=ROAD_LENGTH,
        max_speed=MAX_SPEED,
        n_lanes=N_LANES,
        lane_priority=LANES_PRIORITY,
        lane_density=LANES_DENSITY,
        safe_distance=SAFE_DISTANCE
    )

    engine = Engine(run_config)
    engine.run()

    # Just test for now

    test(run_config=run_config)


def test(run_config: RunConfig):
    # Call the test functions here
    # tests.test_distances(run_config=run_config, plot=True)
    # tests.test_velocities(run_config=run_config, plot=True)
    # tests.test_locations(run_config=run_config, plot=True)
    tests.test_overtake(run_config=run_config, plot=True)
    pass


if __name__ == "__main__":
    run()
