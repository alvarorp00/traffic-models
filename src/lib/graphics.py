"""
Module for graphics and plotting

This module contains functions for plotting the model and its components.
"""

import logging
import sys
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from celluloid import Camera

from lib.engine import Model, RunConfig, Trace
import lib.driver

import seaborn as sns
sns.set()

from matplotlib import rc  # for LaTeX text rendering  # noqa: E402
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def style_set(ax, **kwargs):
    """
    Parameters
    ---------
    Showing parameters and default values:

    (check the function body for the actual values)
    """
    title = kwargs.get('title', '')
    xaxis = kwargs.get('xaxis', True)
    yaxis = kwargs.get('yaxis', True)
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    legend = kwargs.get('legend', False)
    tick_params = kwargs.get('tick_params', 18)
    title_size = kwargs.get('title_size', 20)
    legend_title_fontsize = kwargs.get('legend_title_fontsize', 'large')
    legend_size = kwargs.get('legend_size', 13)
    legend_loc = kwargs.get('legend_loc', 'best')
    legend_bbox_to_anchor = kwargs.get('legend_bbox_to_anchor', None)
    legend_borderaxespad = kwargs.get('legend_borderaxespad', 0.)
    legend_title = kwargs.get('legend_title', '')
    legend_frameon = kwargs.get('legend_frameon', True)
    legend_fancybox = kwargs.get('legend_fancybox', True)
    legend_borderpad = kwargs.get('legend_borderpad', 0.5)
    legend_spacing = kwargs.get('legend_spacing', 0.5)
    linewidth = kwargs.get('linewidth', 2)
    edgecolor = kwargs.get('edgecolor', 'black')
    xlabels = kwargs.get('xlabels', None)
    xticks = kwargs.get('xticks', None)
    ylabels = kwargs.get('ylabels', None)

    ax.patch.set_edgecolor(edgecolor)
    ax.patch.set_linewidth(linewidth)

    ax.xaxis.set_tick_params(labelsize=tick_params)
    ax.yaxis.set_tick_params(labelsize=tick_params)

    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=tick_params)
    ax.set_ylabel(ylabel, fontsize=tick_params)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if legend:
        ax.legend(
            title_fontsize=legend_title_fontsize,
            fontsize=legend_size,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            borderaxespad=legend_borderaxespad,
            title=legend_title,
            frameon=legend_frameon,
            fancybox=legend_fancybox,
            borderpad=legend_borderpad,
            labelspacing=legend_spacing
        )

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
    for driver in model.active_drivers:
        print(f"\tDriver {driver.config.id} is a {driver.config.driver_type} "
              f"driver driving a {driver.config.car_type} car."
              f" @ {driver.config.speed} m/s in lane {driver.config.lane}"
              f" w/ {round(driver.config.location*100/model.road.length, 3)}"
              f" % completed.\n", file=file)
        if n_drivers > 0:
            n_drivers -= 1
            if n_drivers == 0:
                break

    # Print road information
    print("Road information:", file=file)
    print(f"\tRoad length: {model.road.length} m", file=file)
    print(f"\tRoad lanes: {model.road.n_lanes}", file=file)
    # Print # of drivers in each lane
    print("\tDrivers in each lane:", file=file)
    for lane in range(model.road.n_lanes):
        n_lane_drivers = len(list(filter(
            lambda d: d.config.lane == lane,
            model.active_drivers
        )))


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

    classified_by_car = lib.driver.Driver.classify_by_car(drivers)
    data = {}
    for c_type in list(lib.driver.CarType):
        c_d_by_type = lib.driver.Driver.classify_by_type(
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
    ax = figure.add_subplot(111)  # type: ignore
    X = np.arange(len(list(lib.driver.DriverType)))

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


def plot_locations_frame(
    drivers: List[lib.driver.Driver],
    ax: plt.Axes,
    **kwargs
):
    """
    Plot the locations of the drivers in the simulation in a single
    frame.

    Parameters
    ----------
    drivers : list[lib.driver.Driver]
        List of drivers in the simulation.
    fname : str
        Name of the file to save the plot to.
    figure : plt.Figure
        Figure to plot the data to.
    ax : plt.Axes
        Axes to plot the data to.
    kwargs : dict
        - `lane_priority` : LanePriority
            Lane priority to use when plotting the locations.
            Defaults to `LanePriority.LEFT`.
        - `n_lanes` : int
            Number of lanes in the simulation.
        - `road_length` : float
            Length of the road in the simulation.
    """
    n_lanes = kwargs['n_lanes']
    lane_priority = kwargs['lane_priority']
    road_length = kwargs['road_length']
    plot_distance_to_front = kwargs.get('plot_distance_to_front', False)

    # Scatter with different colors for each driver type
    # and different shapes for each car type
    shapes = {
        lib.driver.CarType.SEDAN: 'd',
        lib.driver.CarType.SUV: 'p',
        lib.driver.CarType.TRUCK: 's',
        lib.driver.CarType.MOTORCYCLE: '^'
    }
    colors = {
        lib.driver.DriverType.CAUTIOUS: 'r',
        lib.driver.DriverType.NORMAL: 'b',
        lib.driver.DriverType.RISKY: 'y',
        lib.driver.DriverType.AGGRESSIVE: 'g',
        lib.driver.DriverType.RECKLESS: 'm'
    }

    S = 12
    Z_ORD_DRV = 1

    for __driver in drivers:
        lane_loc = n_lanes - __driver.config.lane\
            if lane_priority == lib.driver.LanePriority.LEFT \
            else __driver.config.lane
        discrepancy = 0\
            if lane_priority == lib.driver.LanePriority.LEFT \
            else 1
        ax.scatter(
            lane_loc + discrepancy,
            __driver.config.location,
            s=S,
            color=colors[__driver.config.driver_type],
            marker=shapes[__driver.config.car_type],
            zorder=Z_ORD_DRV
        )

    xticks = np.arange(start=0, stop=n_lanes+2)
    xlabels = [f'{i}' for i in np.arange(start=1, stop=n_lanes+1)]
    if lane_priority == lib.driver.LanePriority.LEFT:
        # Reverse labels
        xlabels = xlabels[::-1]
    # Add empty labels at the start and end
    xlabels = [''] + xlabels + ['']

    ylim = (0, road_length)

    style_set(
        ax,
        title='Driver locations',
        ylabel='Location (m)',
        xlabel='Lane',
        ylim=ylim,
        xticks=xticks,
        xlabels=xlabels,
        legend=False,  # Manage the legend here
    )

    # shadowplot to get the legend:

    Z_ORD_SHW = -1

    l1 = []
    for d_type in list(lib.driver.DriverType):
        _l = ax.scatter(
            xticks[0],
            road_length / 4,
            s=S,
            alpha=1,
            label=f'{d_type.name}',
            color=colors[d_type],
            marker='o',
            zorder=Z_ORD_SHW,
        )
        l1.append(_l)

    l2 = []
    for c_type in list(lib.driver.CarType):
        _l = ax.scatter(
            xticks[0],
            road_length / 4,
            s=S,
            alpha=1,
            label=f'{c_type.name}',
            color='#1e81b0',
            marker=shapes[c_type],
            zorder=Z_ORD_SHW,
        )
        l2.append(_l)

    legend1 = ax.legend(
        handles=l1,
        labels=[__l.get_label() for __l in l1],
        loc='upper left',
        title='Driver Type',
        borderpad=1.5,
        labelspacing=1.5,
        framealpha=1,
    )

    legend2 = ax.legend(
        handles=l2,
        labels=[__l.get_label() for __l in l2],
        loc='lower left',
        title='Car Type',
        borderpad=1.5,
        labelspacing=1.5,
        framealpha=1,
    )

    ax.add_artist(legend1)

    # shadowplot on border positions to leave space
    legend1 = ax.scatter(
        xticks[0],
        road_length/2,
        s=0,
        alpha=0,
    )

    ax.scatter(
        xticks[-1],
        road_length/2,
        s=0,
        alpha=0,
    )


def plot_locations(drivers: list[lib.driver.Driver], fname: str,
                   **kwargs):
    """
    Plot of the locations of the drivers in the simulation.

    Parameters
    ----------
    drivers : list[lib.driver.Driver]
        List of drivers in the simulation.
    fname : str
        Name of the file to save the plot to.
    kwargs : dict
        - `lane_priority` : LanePriority
            Lane priority to use when plotting the locations.
            Defaults to `LanePriority.LEFT`.
        - `n_lanes` : int
            Number of lanes in the simulation.
        - `road_length` : float
            Length of the road in the simulation.
            Defaults to max(driver.locations).

    NOTE: in order to plot correctly the locations, the lanes index
    must be in the range [0, n_lanes-1].
    """

    n_lanes: int = kwargs.get(
        'n_lanes',
        -1
    )

    if n_lanes == -1:
        logging.warning(
            "Number of lanes not specified. Stopping plot_locations."
        )
        return

    assert n_lanes > 0, "Number of lanes must be positive."

    # Plot the data
    figure = plt.figure(figsize=(14, 7))

    # One plot, one column per lane
    ax = figure.add_subplot(111)  # type: ignore

    # Use the plot_locations_frame function to plot the data
    plot_locations_frame(
        drivers=drivers,
        ax=ax,
        **kwargs
    )

    figure.savefig(fname)

    plt.close()


def plot_locations_video(
    trace: Trace,
    fname: str,
    run_config: RunConfig,
    plot_distance_to_front: bool = False,
) -> None:
    """
    Generate an animation of the drivers' locations all along
    the simulation, using the `trace` object.
    """
    trace_data = trace.data

    # Get the number of lanes
    n_lanes = run_config.n_lanes

    # Get the road length
    road_length = run_config.road_length

    # Get the lane priority (only to plot in the right order)
    lane_priority = run_config.lane_priority

    figure = plt.figure(figsize=(14, 7))
    ax = figure.add_subplot(111)  # type: ignore
    camera = Camera(figure)

    for data in trace_data:
        drivers = data.all_active_drivers()

        # Plot the data
        plot_locations_frame(
            drivers=drivers,
            ax=ax,
            n_lanes=n_lanes,
            road_length=road_length,
            lane_priority=lane_priority,
            plot_distance_to_front=plot_distance_to_front,
        )

        camera.snap()

    animation = camera.animate(blit=True)
    animation.save(fname, writer='ffmpeg')
