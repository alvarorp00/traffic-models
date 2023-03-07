"""

"""

import numpy as np
from matplotlib import pyplot as plt
from Engine import Model


def print_model(model: Model, n_drivers: int = 0):
    # Print driver information
    print("Driver information:")
    for driver in model.drivers.values():
        print(f"Driver {driver.config.id} is a {driver.config.driver_type} "
              f"driver driving a {driver.config.car_type} car."
              f" @ {driver.config.speed} m/s in lane {driver.config.lane}"
              f" w/ {round(driver.config.location*100/model.road.length, 3)}"
              f" % completed.")
        if n_drivers > 0:
            n_drivers -= 1
            if n_drivers == 0:
                break


def plot_model(model: Model):
    pass  # TODO: plot the model
