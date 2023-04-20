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


from typing import Dict, List
from lib.driver import Driver, DriverConfig, CarType,\
    DriverDistributions, DriverType
from lib.road import Road
import numpy as np
import scipy.stats as st


def initialize_drivers(population_size: int,
                       position_low: float = 0.0,
                       position_high: float = 1e4,
                       lognormal_mode=False,) -> Dict[int, Driver]:
    """
    Initializes the drivers for the simulation.

    Take the population size and obtain a random sample from
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
        from a multinomial distribution:
            - 40% of the drivers are type 1
            - 30% of the drivers are type 2
            - 15% of the drivers are type 3
            - 10% of the drivers are type 4
            - 5% of the drivers are type 5
    """
    driver_types = DriverType.random_lognormal(size=population_size)
    drivers = {}

    for i in range(population_size):
        car_type = CarType.random()[0]
        dconfig = DriverConfig(
            id=i,
            driver_type=driver_types[i],
            car_type=car_type,
            location=DriverDistributions.location_initialize(
                start=position_low,
                end=position_high,
                size=1
            ),
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
            assert len(kwargs['lane_priority']) == self.lanes
            assert isinstance(kwargs['lane_priority'], np.ndarray)
            self.lane_priority = kwargs['lane_priority']
        else:
            self.lane_priority = np.array([1])  # By default just one lane
        if 'max_speed' in kwargs:
            assert isinstance(kwargs['max_speed'], float) or \
                     isinstance(kwargs['max_speed'], int)
            self.max_speed = kwargs['max_speed']
        else:
            self.max_speed = 120  # In km/h


class Model:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

        self.info = {}

        self.info['drivers'] = initialize_drivers(
            population_size=self.run_config.population_size
        )

        # Change the location of the drivers, speed and lane
        for d in self.info['drivers'].values():
            d.config.location = np.random.uniform(
                low=0,
                high=self.run_config.road_length
            )
            d.config.speed = np.random.uniform(
                low=0,
                high=30
            )
            d.config.lane = np.random.choice(
                a=np.arange(0, self.run_config.lanes),
                p=self.run_config.lane_priority
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
