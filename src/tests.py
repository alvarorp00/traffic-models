"""
Test cases for the project. Test functionallity of the project.
"""

import copy
import sys

# import lib.road as road
import lib.engine as engine
# import numpy as np
import lib.driver
import lib.driver_distributions
import lib.graphics as graphics


ITERS = 1000


def test_distances(run_config: engine.RunConfig, fname=None, plot=False):
    # In one line
    file = open(fname, 'w+') if fname is not None else sys.stdout

    drivers_dict = engine.initialize_drivers(run_config=run_config)
    drivers = list(drivers_dict.values())

    for driver in drivers:

        distance =\
            lib.driver_distributions.risk_overtake_distance(driver)

        print(f"Driver {driver.config.id} is a {driver.config.driver_type}"
              f"\n\tdriver driving a {driver.config.car_type} car @"
              f"{driver.config.speed} m/s"
              f"\n\tOvertake distance: {distance} m\n", file=file)

    if plot:
        graphics.plot_distances(
            drivers=drivers,
            fname='img/out/test_distances.png'
        )


def test_velocities(run_config: engine.RunConfig, fname=None, plot=False):
    # In one line
    file = open(fname, 'w+') if fname is not None else sys.stdout

    drivers_dict = engine.initialize_drivers(run_config=run_config)
    drivers = list(drivers_dict.values())

    for driver in drivers:

        velocity =\
            lib.driver_distributions.speed_initialize(
                driver_type=driver.config.driver_type,
                car_type=driver.config.car_type,
            )
        # lib.driver_distributions.risk_overtake_velocity(driver)

        print(f"Driver {driver.config.id} is a {driver.config.driver_type}"
              f"\n\tdriver driving a {driver.config.car_type} car @"
              f"{driver.config.speed} m/s"
              f"\n\tOvertake velocity: {velocity} m/s\n", file=file)

    if plot:
        graphics.plot_velocities(
            drivers=drivers,
            fname='img/out/test_velocities.png'
        )


def test_locations(run_config: engine.RunConfig, fname=None, plot=False):
    drivers_dict = engine.initialize_drivers(run_config=run_config)
    drivers = list(drivers_dict.values())

    if plot:
        graphics.plot_locations(
            drivers=drivers,
            fname='img/out/model_locations.png',
            lane_priority=run_config.lane_priority,
            n_lanes=run_config.n_lanes,
            road_length=run_config.road_length
        )


def test_overtake(run_config: engine.RunConfig, fname=None, plot=False):
    drivers_dict = engine.initialize_drivers(run_config=run_config)
    drivers = list(drivers_dict.values())
    drivers_by_lane = lib.driver.Driver.classify_by_lane(drivers=drivers)
    old_state = copy.deepcopy(drivers)
    old_state_by_id = lib.driver.Driver.classify_by_id(drivers=old_state)
    # Plot representation of the model
    if plot:
        graphics.plot_locations(
            drivers=drivers,
            fname='img/out/model_overtake_before.png',
            lane_priority=run_config.lane_priority,
            n_lanes=run_config.n_lanes,
            road_length=run_config.road_length
        )
    # Evolve the model
    old_drivers = copy.deepcopy(drivers)
    old_drivers_by_lane = copy.deepcopy(drivers_by_lane)
    __new_drivers = []
    for _ in range(run_config.time_steps):
        __new_drivers = []
        for driver in old_drivers:
            __driver = lib.driver_distributions.speed_update(
                driver=driver,
                drivers_by_lane=old_drivers_by_lane,
            )

            __driver_copy = lib.driver.Driver.copy(__driver)

            __driver_copy.config.speed = __driver.config.speed
            __driver_copy.config.lane = __driver.config.lane
            __driver_copy.config.location += __driver_copy.config.speed / 3.6

            # print(f'Driver speed: {driver.config.speed} -> {__driver.config.speed}')  # noqa

            __new_drivers.append(__driver_copy)

        old_drivers_by_lane = copy.deepcopy(lib.driver.Driver.classify_by_lane(drivers=__new_drivers))  # noqa
        old_drivers = copy.deepcopy(__new_drivers)

        # print(f'driver {driver.config.id} location: {driver.config.location} -> {__driver.config.location}')  # noqa

    # TODO: plot the difference between old state and new state of model

    if plot:
        graphics.plot_locations(
            drivers=__new_drivers,
            fname='img/out/model_overtake_after.png',
            lane_priority=run_config.lane_priority,
            n_lanes=run_config.n_lanes,
            road_length=run_config.road_length
        )

    # for new_driver in __new_drivers:
    #     old_driver = old_state_by_id[new_driver.config.id]
    #     # Print old lane & old location & new lane & new location
    #     print(f'\tOld lane: {old_driver.config.lane} -> {new_driver.config.lane}')  # noqa
    #     print(f'\tOld location: {old_driver.config.location} -> {new_driver.config.location}')  # noqa
