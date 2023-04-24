"""
    This module contains the Engine class,
    which is the main class of the simulation.

    The Engine class is responsible for running the simulation.
    It is responsible for creating the drivers, and road.
    It is also responsible for running the simulation and
    updating the state of the simulation.

    Attributes
    ----------
    run_config : RunConfig
        The configuration of the simulation.
"""


import logging
from typing import Dict, List
from lib.driver import Driver, DriverConfig, CarType,\
    DriverDistributions, DriverType, LanePriority
from lib.road import Road
import numpy as np
import scipy.stats as st


def initialize_drivers(run_config: 'RunConfig',
                       lognormal_mode: bool = False,
                       debug=False) -> Dict[int, Driver]:
    """
    Initializes the drivers for the simulation.

    If lognormal_mode is set to True, the method
    will take the population size and obtain a random sample from
    the log-normal distribution. Then create an histogram
    of the sample (5 bins), where each bin represents a
    driver type. Then create the drivers based on the
    histogram.

    The id assigned to the drivers is the index of the
    driver in the list of drivers. They are unique and are
    assigned in the order of creation.

    The car seleccion is retrieved from CarType.random().

    NOTE: Drivers & cars selection might change in the future.

    Parameters
    ----------
    population_size : int
        The number of drivers in the simulation.
    lognormal_mode : bool
        If True, the drivers are selected from a log-normal
        distribution. If False, the drivers are selected
        from a multinomial distribution. Check the
        function `DriverType.random_lognormal` for more
        information.
    """
    if lognormal_mode:
        driver_types = DriverType.random_lognormal(
            size=run_config.population_size
        )
    else:
        driver_types = DriverType.random(size=run_config.population_size)
    drivers = {}

    (locations_lane, safe) = DriverDistributions.lane_location_initialize(
                        start=0, end=run_config.road_length,
                        size=run_config.population_size,
                        n_lanes=run_config.lanes,
                        safe_distance=run_config.safe_distance,
                        lane_density=np.array(run_config.lane_density),
                        safe=True,
                    )

    if not safe:
        logging.warning("The drivers were not initialized safely. "
                        "The location of the drivers might be too "
                        "close to other drivers.")

        if debug is False:
            logging.critical("Cannot continue with the simulation. "
                             "Location of drivers is not safe to proceed.")
            raise Exception("Cannot continue with the simulation. ")

    lanes_array = []
    for k in locations_lane.keys():
        lanes_array.extend(
            [k for _ in range(len(locations_lane[k]))]
        )

    locations_array = []
    for k in locations_lane.keys():
        # print(f'locations_lane[{k}] = {locations_lane[k].tolist()}')
        locations_array.extend(locations_lane[k].tolist())

    for i in range(len(driver_types)):
        car_type = CarType.random()[0]
        dconfig = DriverConfig(
            id=i,
            driver_type=driver_types[i],
            car_type=car_type,
            lane=int(lanes_array[i]),
            location=float(locations_array[i]),
            speed=DriverDistributions.speed_initialize(
                car_type=car_type,
                driver_type=driver_types[i],
                size=1
            )
        )
        drivers[i] = Driver(config=dconfig)

    return drivers


class RunConfig:
    def __init__(self, **kwargs):
        """
        Constructor for the RunConfig class.

        Parameters
        ----------
        population_size : int
            The number of drivers in the simulation. Defaults to 100.
        road_length : float
            The length of the road in meters. Defaults to 1e4.
        time_steps : int
            Time steps to run the simulation for. Defaults to 1000.
        lanes : int
            The number of lanes on the road. Defaults to 1.
        lane_priority : LanePriority
            The priority of the lanes. Defaults to LanePriority.LEFT.
            The priority of the lanes is used to determine which
            lane the drivers will use when changing lanes, namely
            when the driver is in the middle lane and wants to
            overtake, the driver will use the lane with the highest
            priority; LanePriority.LEFT means that the left lane
            is the latest lane to occupy when overtaking.
        safe_distance : float
            The safe distance between drivers in meters.
            Defaults to 2 meters.
        lane_density : List[float]
            The density of the drivers in each lane.
            Defaults to equal distribution.
        max_speed : float
            The maximum speed of the drivers in the simulation.
            Defaults to 120 km/h.
        debug : bool
            If True, the simulation will run in debug mode.
            Defaults to False.

            For example, when initializing the drivers, the
            locations of the drivers are checked to make sure
            that they are not too close to each other. If they
            are too close, the simulation will not continue unless
            debug is set to True.
        """

        if 'population_size' in kwargs:
            assert isinstance(kwargs['population_size'], int)
            self.population_size = kwargs['population_size']
        else:
            self.population_size = 100  # default value
        if 'road_length' in kwargs:
            assert isinstance(kwargs['road_length'], float) or \
                        isinstance(kwargs['road_length'], int)
            self.road_length = kwargs['road_length']
        else:
            self.road_length = 1e4  # default value --> in meters
        if 'time_steps' in kwargs:
            assert isinstance(kwargs['time_steps'], int)
            self.time_steps = kwargs['time_steps']
        else:
            self.time_steps = 1000
        if 'lanes' in kwargs:
            assert isinstance(kwargs['lanes'], int)
            self.lanes = kwargs['lanes']
        else:
            self.lanes = 1
        if 'lane_priority' in kwargs:
            assert isinstance(kwargs['lane_priority'], LanePriority)
            self.lane_priority = kwargs['lane_priority']
        else:
            self.lane_priority = LanePriority.LEFT
        if 'safe_distance' in kwargs:
            assert isinstance(kwargs['safe_distance'], float)
            self.safe_distance = kwargs['safe_distance']
        else:
            self.safe_distance = 2.0  # In meters
        if 'lane_density' in kwargs:
            assert isinstance(kwargs['lane_density'], list)
            assert len(kwargs['lane_density']) == self.lanes
            self.lane_density = kwargs['lane_density']
        else:
            self.lane_density = [1.0 / self.lanes] * self.lanes
        if 'max_speed' in kwargs:
            assert isinstance(kwargs['max_speed'], float) or \
                     isinstance(kwargs['max_speed'], int)
            self.max_speed = kwargs['max_speed']
        else:
            self.max_speed = 120  # In km/h
        if 'debug' in kwargs:
            assert isinstance(kwargs['debug'], bool)
            self.debug = kwargs['debug']
        else:
            self.debug = False


class Model:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

        self.info = {}

        self.info['drivers'] = initialize_drivers(
            run_config=self.run_config,
            debug=self.run_config.debug
        )

        self.info['road'] = Road(
            length=self.run_config.road_length,
            lanes=self.run_config.lanes,
            max_speed=self.run_config.max_speed
        )

        self.info['locations'] = dict(zip(
            self.info['drivers'].keys(),
            [d.config.location for d in self.info['drivers'].values()]
        ))

        self.info['speeds'] = dict(zip(
            self.info['drivers'].keys(),
            [d.config.speed for d in self.info['drivers'].values()]
        ))

        self.info['lanes'] = dict(zip(
            self.info['drivers'].keys(),
            [d.config.lane for d in self.info['drivers'].values()]
        ))

    @property
    def run_config(self) -> "RunConfig":
        return self._run_config

    @run_config.setter
    def run_config(self, run_config: "RunConfig"):
        self._run_config = run_config

    @property
    def info(self) -> Dict:
        return self._info

    @info.setter
    def info(self, info: Dict):
        self._info = info

    @property
    def road(self) -> "Road":
        return self.info['road']

    @property
    def drivers(self) -> Dict[int, "Driver"]:
        return self.info['drivers']

    @property
    def locations(self) -> Dict[int, float]:
        return self.info['locations']

    @property
    def speeds(self) -> Dict[int, float]:
        return self.info['speeds']

    @property
    def lanes(self) -> Dict[int, int]:
        return self.info['lanes']

    def sort_by_position(self, lane: int) -> List[Driver]:
        """
        Returns a list of drivers sorted by their position on the road.
        """
        drivers_in_lane = [
            filter(lambda d: d.config.lane == lane, self.drivers.values())
        ]

        # TODO
        pass

    @staticmethod
    def classify_by_driver(drivers: list[Driver]) ->\
            Dict[DriverType, List[Driver]]:
        dict = {}

        for d in drivers:
            if d.config.driver_type in dict.keys():
                dict[d.config.driver_type].append(d)
            else:
                dict[d.config.driver_type] = [d]

        return dict

    @staticmethod
    def classify_by_car(drivers: list[Driver]) -> Dict[CarType, List[Driver]]:
        dict = {}

        for d in drivers:
            if d.config.car_type in dict.keys():
                dict[d.config.car_type].append(d)
            else:
                dict[d.config.car_type] = [d]

        return dict


class Engine:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.trace = Trace()

        # Initialize the model
        self.model = Model(self.run_config)

    @property
    def run_config(self) -> "RunConfig":
        return self._run_config

    @run_config.setter
    def run_config(self, run_config):
        self._run_config = run_config

    @property
    def trace(self) -> "Trace":
        return self._trace

    @trace.setter
    def trace(self, trace):
        self._trace = trace

    @property
    def model(self) -> "Model":
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def run(self):
        for t in range(self.run_config.time_steps):
            # TODO: update the state of the model
            self.trace.add(self.model)  # Record the state of the model


class Trace:
    def __init__(self, *args, **kwargs):
        self.data = []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def add(self, data):
        self.data.append(data)


class Utils:
    @staticmethod
    def partition_d(array: List[Driver], begin: int, end: int):
        pivot = begin
        for i in range(begin+1, end+1):
            if array[i].config.location <= array[begin].config.location:
                pivot += 1
                array[i], array[pivot] = array[pivot], array[i]
        array[pivot], array[begin] = array[begin], array[pivot]
        return pivot

    @staticmethod
    def quicksort_d(array, begin=0, end=None):
        if end is None:
            end = len(array) - 1

        def _quicksort_d(array, begin, end):
            if begin >= end:
                return
            pivot = Utils.partition_d(array, begin, end)
            _quicksort_d(array, begin, pivot-1)
            _quicksort_d(array, pivot+1, end)
        return _quicksort_d(array, begin, end)
