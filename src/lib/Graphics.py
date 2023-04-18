"""
Module for graphics and plotting

This module contains functions for plotting the model and its components.
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
from Engine import Model


def print_model(model: Model, n_drivers: int = 0, fname: str = None):
    if fname is None:
        file = sys.stdout
    else:
        file = open(fname, 'w')

    # Print driver information
    print("Driver information:", file=file)
    for driver in model.drivers.values():
        print(f"\tDriver {driver.config.id} is a {driver.config.driver_type} "
              f"driver driving a {driver.config.car_type} car."
              f" @ {driver.config.speed} m/s in lane {driver.config.lane}"
              f" w/ {round(driver.config.location*100/model.road.length, 3)}"
              f" % completed.\n", file=file)
        if n_drivers > 0:
            n_drivers -= 1
            if n_drivers == 0:
                break


def plot_model_2(model: Model):
    pass
