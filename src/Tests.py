"""
Test cases for the project. Test functionallity of the project.
"""

import sys

import lib.Driver as Driver
import lib.Road as Road
import Engine
import numpy as np


ITERS = 1000


def test_distances(iters=ITERS, fname=None):
    # In one line
    file = open(fname, 'w+') if fname is not None else sys.stdout

    drivers_dict = Engine.initialize_drivers(population_size=iters)
    drivers = list(drivers_dict.values())

    for driver in drivers:

        distance = Driver.DriverDistributions.risk_overtake_distance(driver)

        print(f"DriverType is {driver.config.driver_type.name}")

        print(f"Driver {driver.config.id} is a {driver.config.driver_type}"
              f"\n\tdriver driving a {driver.config.car_type} car."
              f"\n\tOvertake distance: {distance} m\n", file=file)


test_distances(fname='out/test_distances.txt')
