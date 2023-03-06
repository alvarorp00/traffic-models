"""
This module runs the simulation, using the Engine class
and the configuration specified in the Config module.
"""

from Engine import Engine, RunConfig
from Config import POPULATION_SIZE, ROAD_LENGTH, TIME_STEPS, LANES


def run():
    """
    Runs the simulation
    """

    run_config = RunConfig(
        population_size=POPULATION_SIZE,
        road_length=ROAD_LENGTH,
        time_steps=TIME_STEPS,
        lanes=LANES
    )

    model = Engine(run_config)
    model.run()

    print("Simulation complete")


if __name__ == "__main__":
    run()
