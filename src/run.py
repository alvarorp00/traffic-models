"""
This module runs the simulation, using the Engine class
and the configuration specified in the Config module.
"""

from lib.engine import Engine, RunConfig
from config import POPULATION_SIZE, ROAD_LENGTH, MAX_SPEED,\
        TIME_STEPS, LANES, LANES_PRIORITY
from lib.graphics import print_model


def run():
    """
    Runs the simulation
    """

    run_config = RunConfig(
        population_size=POPULATION_SIZE,
        time_steps=TIME_STEPS,
        road_length=ROAD_LENGTH,
        max_speed=MAX_SPEED,
        lanes=LANES,
        lane_priority=LANES_PRIORITY,
    )

    engine = Engine(run_config)
    engine.run()

    print("Simulation complete")

    # Print the results
    print_model(engine.model)


if __name__ == "__main__":
    run()
