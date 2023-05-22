"""
This module runs the simulation, using the Engine class
and the configuration specified in the Config module.
"""

from lib.engine import Engine, Stats, RunConfig
from config import Config
# import tests


def run():
    """
    Runs the simulation
    """

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

    engine_stats = Stats(engine=engine)
    stats = engine_stats.get_stats()

    print('Line changes:')
    for d, lc in stats['lane_changes'].items():
        print(f"\tDriver {d} lanes: {lc}")

    print('Drivers that finished:')
    for driver in engine.model.inactive_drivers:
        print(f'\tDriver {driver.config.id} finished @ {driver.config.driver_type} driver driving a {driver.config.car_type} car @ {driver.config.speed} m/s')
    print(f'{len(engine.model.inactive_drivers)} drivers finished')
    print(f'{len(engine.model.active_drivers)} drivers still active')

    print('Average time:')
    for d, at in stats['avg_time_taken'].items():
        print(f"\tDriver {d} avg time: {at}")

    trace = engine.trace

    last = trace.last

    if 'verbose' in run_config.__dict__:
        for driver in last.active_drivers:
            print(f"Driver {driver.config.id} is a {driver.config.driver_type}"
                  f"\n\tdriver driving a {driver.config.car_type} car @"
                  f"{driver.config.speed} m/s at lane {driver.config.lane}"
                  f"\n\t at location {driver.config.location} m\n")

    # print(stats)

    # Just test for now

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
