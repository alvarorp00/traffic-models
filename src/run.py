"""
This module runs the simulation, using the Engine class
and the configuration specified in the Config module.
"""

from lib.engine import Engine, RunConfig
from config import POPULATION_SIZE, ROAD_LENGTH, MAX_SPEED,\
        TIME_STEPS, LANES, LANES_PRIORITY, LANES_DENSITY,\
        SAFE_DISTANCE
import lib.graphics as graphics


def run():
    """
    Runs the simulation
    """

    run_config = RunConfig(
        population_size=POPULATION_SIZE,
        time_steps=0,  # Don't evolve the system for now
        road_length=ROAD_LENGTH,
        max_speed=MAX_SPEED,
        lanes=LANES,
        lane_priority=LANES_PRIORITY,
        lane_density=LANES_DENSITY,
        safe_distance=SAFE_DISTANCE
    )

    engine = Engine(run_config)
    engine.run()

    print("Simulation complete")

    # Print the results
    # print_model(engine.model)

    graphics.plot_locations(
        drivers=list(engine.model.drivers.values()),
        fname='img/out/model_locations.png'
    )


if __name__ == "__main__":
    run()
