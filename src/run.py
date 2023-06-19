"""
This module runs the simulation, using the Engine class
and the configuration specified in the Config module.
"""

from config import Config
from lib.engine import Engine, RunConfig
from lib.stats import Stats, StatsItems
# import tests


def run():
    """
    Runs the simulation
    """
    STATS = True

    # Map every configuration parameter in lib.config to a RunConfig attribute
    config = Config()
    # run_config = RunConfig(**config.__dict__)

    # Get all the members of the Config class
    namespace = [attr.lower() for attr in dir(config) if not callable(getattr(config, attr)) and not attr.startswith("__")]

    run_config_dict = dict(zip(
        namespace,
        [getattr(config, member.upper()) for member in namespace]
    ))

    # Create a RunConfig object with the dictionary
    run_config = RunConfig(**run_config_dict)

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
            print(f'\tDriver {driver.config.id} speed changes: {__speed_changes[driver.config.id]}')
            # Print '---' to separate each driver
            print('\t-----------------------------------')

        # print('Drivers that finished:')
        # for driver in engine.model.inactive_drivers.values():
        #     print(f'\tDriver {driver.config.id} finished @ {driver.config.driver_type} driver driving a {driver.config.car_type} car @ {driver.config.speed} m/s')
        # print(f'{len(engine.model.inactive_drivers)} drivers finished')
        # print(f'{len(engine.model.all_active_drivers())} drivers still active')

        print('Average time:')
        for d, at in stats[StatsItems.AVG_TIME_TAKEN].items():
            print(f"\tDriver {d} avg time: {at}")
            # Print how many drivers of each type finished
            print(f"\t\t{stats[StatsItems.DRIVERS_FINISHED_DRV_TYPE][d]} {d} drivers finished")
            # Print the number of cars of each type that finished for this driver type
            for car_type, num in stats[StatsItems.DRIVERS_FINISHED_DRV_CAR_TYPE][d].items():
                print(f"\t\t\t{num} {car_type} cars finished")

    # test(run_config=run_config)


def test(run_config: RunConfig):
    # Call the test functions here
    # tests.test_distances(run_config=run_config, plot=True)
    # tests.test_velocities(run_config=run_config, plot=True)
    # tests.test_locations(run_config=run_config, plot=True)
    # tests.test_overtake(run_config=run_config, plot=True)
    pass


if __name__ == "__main__":
    run()
