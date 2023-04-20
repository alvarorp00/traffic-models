"""
Test cases for the project. Test functionallity of the project.
"""

import sys

import lib.road as road
import lib.engine as engine
import numpy as np
import lib.driver

from lib.graphics import plot_distances, plot_velocities


ITERS = 1000


def test_distances(iters=ITERS, fname=None, plot=False):
    # In one line
    file = open(fname, 'w+') if fname is not None else sys.stdout

    drivers_dict = engine.initialize_drivers(population_size=iters)
    drivers = list(drivers_dict.values())

    for driver in drivers:

        distance = lib.driver.DriverDistributions.risk_overtake_distance(driver)

        print(f"Driver {driver.config.id} is a {driver.config.driver_type}"
              f"\n\tdriver driving a {driver.config.car_type} car @ {driver.config.speed} m/s"
              f"\n\tOvertake distance: {distance} m\n", file=file)

    if plot:
        plot_velocities(drivers=drivers, fname='img/out/test_velocities.png')


drivers = test_distances(fname='out/test_distances.txt', plot=True)
