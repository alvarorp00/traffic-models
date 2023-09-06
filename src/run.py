"""
This module runs the simulation, using the Engine class
and the configuration specified in the Config module.
"""

from config import Config
from lib.engine import Engine, RunConfig
from lib.stats import Stats, StatsItems
import lib.graphics as graphics
import tests


def run():
    """
    Runs the simulation
    """
    STATS = True

    # Map every configuration parameter in lib.config to a RunConfig attribute
    config = Config()
    # run_config = RunConfig(**config.__dict__)

    run_config = RunConfig.get_run_config(config)

    engine = Engine(run_config)
    engine.run()

    if STATS:
        engine_stats = Stats(engine=engine)
        stats = engine_stats.get_stats()

        __lane_changes = stats[StatsItems.LANE_CHANGES]
        __speed_changes = stats[StatsItems.SPEED_CHANGES]
        __all_drivers = engine.model.all_active_drivers()
        __all_drivers.extend(engine.model.inactive_drivers.values())
        print('Lane & Speed changes:')
        for driver in __all_drivers:
            print(f'\tDriver {driver.config.id} lane changes: {__lane_changes[driver.config.id]}')
            __formatted_speed = [round(speed, 2) for speed in __speed_changes[driver.config.id]]
            print(f'\tDriver {driver.config.id} speed changes: {__formatted_speed} m/s')
            # Print '---' to separate each driver
            print('\t-----------------------------------')

        print('Average time:')
        for d, at in stats[StatsItems.AVG_TIME_TAKEN].items():
            print(f"\tDriver {d} avg time: {at}")
            # Print how many drivers of each type finished
            print(f"\t\t{stats[StatsItems.DRIVERS_FINISHED_DRV_TYPE][d]} {d} drivers finished")
            # Print the number of cars of each type that finished for this driver type
            for car_type, num in stats[StatsItems.DRIVERS_FINISHED_DRV_CAR_TYPE][d].items():
                print(f"\t\t\t{num} {car_type} cars finished")

        print('Total cars by driver type:')
        for d, c in stats[StatsItems.CARS_BY_DRV_TYPE].items():
            print(f"\tDriver {d} total cars: {c}")

        print('Accidents by driver type:')
        for d, a in stats[StatsItems.CARS_ACCIDENTED_BY_DRV_TYPE].items():
            print(f"\tDriver {d} accidents: {a}")

    if run_config.plot:
        graphics.plot_locations_video(
            trace=engine.trace,
            fname='locations_test.mp4',
            run_config=run_config
        )


if __name__ == "__main__":
    run()
