"""
Module for graphics and plotting

This module contains functions for plotting the model and its components.
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
from lib.engine import Model
import lib.driver

import seaborn as sns
sns.set()

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def style_set(ax,
              title: str = '',
              xaxis: bool = True,
              yaxis: bool = True,
              xlabel: str = '',
              ylabel: str = '',
              legend: bool = False,
              tick_params: int = 18,
              title_size: int = 20,
              legend_size: int = 18,
              linewidth: int = 2,
              edgecolor: str = 'black',
              xlabels: list[str] = None,
              xticks: list[int] = None,
              ylabels: list[str] = None,):
    ax.patch.set_edgecolor(edgecolor)
    ax.patch.set_linewidth(linewidth)

    ax.xaxis.set_tick_params(labelsize=tick_params)
    ax.yaxis.set_tick_params(labelsize=tick_params)

    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=tick_params)
    ax.set_ylabel(ylabel, fontsize=tick_params)

    if legend:
        ax.legend(fontsize=legend_size)

    ax.xaxis.set_visible(xaxis)
    ax.yaxis.set_visible(yaxis)

    if xlabels is not None:
        ax.set_xticklabels(xlabels)

    if xticks is not None:
        ax.set_xticks(xticks)

    if ylabels is not None:
        ax.set_yticklabels(ylabels)


def print_model(model: Model, n_drivers: int = 0,
                fname: str = ''):
    if fname == '':
        file = sys.stdout
    else:
        file = open(fname, 'w+')

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


def plot_velocities(drivers: list[lib.driver.Driver], fname: str):
    """
    Plots the velocities of the drivers in the simulation.

    It'll plot the velocities of each driver in the simulation
    separately, one plot per car type. On each plot, the colors
    represent the driver type.

    The xaxis will represent the velocity of the driver in m/s,
    and the yaxis the amount of drivers with th at velocity (density)
    through an histogram.
    """

    classified_by_car = Model.classify_by_car(drivers)
    data = {}
    for c_type in list(lib.driver.CarType):
        c_d_by_type = Model.classify_by_driver(
            classified_by_car[c_type]
        )
        new_data = {}
        for d_type, d_collection in c_d_by_type.items():
            new_data[d_type] = np.mean(np.array(list(map(
                lambda d: d.config.speed,
                d_collection
            ))))
        if len(c_d_by_type.keys()) < len(lib.driver.DriverType):
            for d_type in list(lib.driver.DriverType):
                if d_type not in new_data:
                    new_data[d_type] = np.mean(np.zeros(shape=1))
        data[c_type] = new_data

    figure = plt.figure(figsize=(14, 7))
    ax = figure.add_subplot(111)
    X = np.arange(5)

    i = 0
    for c_type, d_info in data.items():
        values = np.array(list(d_info.values()), dtype=np.float32)
        ax.bar(X + i * 0.2, values, width=0.2, label=f'{c_type.name}')
        i += 1

    style_set(
        ax,
        title='Average driver velocity by car type',
        ylabel='Average driver velocity (km/h)',
        legend=True,
        xlabels=[f'{d.name}' for d in list(lib.driver.DriverType)],
        xticks=X+0.25,
    )

    figure.savefig(fname, dpi=300)
    plt.close()


def plot_distances(drivers: list[lib.driver.Driver], fname: str):
    """
    Plots the distances between the drivers in the simulation.

    Plots the distance of each driver to the driver in front of them.
    """
    # TODO
    pass


def plot_locations(drivers: list[lib.driver.Driver], fname: str):
    """
    Plot of the locations of the drivers in the simulation.
    """
    # Separate the locations by lane
    data = {}

    # Get number of lanes
    n_lanes = np.max(np.array(
        [d.config.lane for d in drivers]
    ))

    for lane in range(n_lanes):
        data[lane] = np.array(
            [d.config.location for d in drivers if d.config.lane == lane]
        )

    # Plot the data
    figure = plt.figure(figsize=(14, 7))

    # One plot, one column per lane
    ax = figure.add_subplot(111)  # type: ignore

    for lane, locations in data.items():
        # lane represents the xaxis position

        # Plot the points
        ax.plot(xs=np.ones(shape=locations.shape) * lane,
                ys=locations,)

        print(locations)

    style_set(
        ax,
        title='Driver locations',
        ylabel='Location (m)',
        xlabel='Lane',
    )

    figure.savefig(fname, dpi=300)

    # TODO: fix, not working yet
