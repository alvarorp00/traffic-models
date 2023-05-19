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


import copy
import logging
from typing import Dict, List
from lib.driver import Driver, DriverConfig, CarType,\
    DriverType, LanePriority
import lib.driver_distributions as driver_distributions
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

    Returns
    -------
    Dict[int, Driver]
        A dictionary of drivers, where the key is the id
        of the driver and the value is the driver object.
    """
    if lognormal_mode:
        driver_types = DriverType.random_lognormal(
            size=run_config.population_size
        )
    else:
        driver_types = DriverType.random(
            size=run_config.population_size,
            probs=run_config.driver_type_density,
        )
    drivers = {}

    # Experimental:
    (locations_lane, safe) =\
        driver_distributions.lane_location_initialize_biased(
            start=0, end=run_config.road_length,
            size=run_config.population_size,
            n_lanes=run_config.n_lanes,
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
            road=Road(
                length=run_config.road_length,
                n_lanes=run_config.n_lanes,
                max_speed=run_config.max_speed
            ),
            driver_type=driver_types[i],
            car_type=car_type,
            lane=int(lanes_array[i]),
            location=float(locations_array[i]),
            speed=driver_distributions.speed_initialize(
                car_type=car_type,
                driver_type=driver_types[i],
                size=1,
                max_speed_fixed=run_config.max_speed
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
            It is just a visual representation of the lanes used while
            plotting the simulation.
        safe_distance : float
            The safe distance between drivers in meters.
            Defaults to 2 meters.
        lane_density : List[float]
            The density of the drivers in each lane.
            Defaults to equal distribution.
        driver_type_density : List[float]
            The density of the drivers in each driver type.
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
        if 'n_lanes' in kwargs:
            assert isinstance(kwargs['n_lanes'], int)
            self.n_lanes = kwargs['n_lanes']
        else:
            self.n_lanes = 1
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
            assert isinstance(kwargs['lane_density'], list) or \
                        isinstance(kwargs['lane_density'], np.ndarray)
            assert np.isclose(np.sum(kwargs['lane_density']), 1.0)
            self.lane_density = kwargs['lane_density']
        else:
            self.lane_density = [1.0 / self.n_lanes] * self.n_lanes
        if 'driver_type_density' in kwargs:
            assert isinstance(kwargs['driver_type_density'], list)
            assert np.isclose(np.sum(kwargs['driver_type_density']), 1.0)
            self.driver_type_density = kwargs['driver_type_density']
        else:
            self.driver_type_density = [1.0 / 5] * 5
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

        __drivers_by_id = initialize_drivers(
            run_config=self.run_config,
            debug=self.run_config.debug
        )

        self.info['active_drivers'] = __drivers_by_id

        self.info['inactive_drivers'] = {}

        self.info['road'] = Road(
            length=self.run_config.road_length,
            n_lanes=self.run_config.n_lanes,
            max_speed=self.run_config.max_speed
        )

        self.id_counter = len(self.active_drivers)

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
    def road(self) -> Road:
        return self.info['road']

    @property
    def active_drivers(self) -> List[Driver]:
        """
        Returns a list of all the active drivers
        (didn't reach the end of the road)
        in the simulation indexed by their id.
        """
        return self.info['active_drivers'].values()

    @property
    def inactive_drivers(self) -> List[Driver]:
        """
        Returns a list of all the active drivers in the simulation.
        """
        return self.info['inactive_drivers'].values()

    @property
    def id_counter(self) -> int:
        return self._id_counter

    @id_counter.setter
    def id_counter(self, id_counter: int):
        self._id_counter = id_counter

    def spawn_driver(self) -> None:
        """
        Spawns a new driver in the simulation.
        """
        car_type = CarType.random()[0]
        drv_type = DriverType.random(
            size=1,
            probs=self.run_config.driver_type_density
        )[0]
        new_driver = Driver(
            config=DriverConfig(
                id=self.id_counter,
                driver_type=drv_type,
                road=self.road,
                car_type=car_type,
                lane=driver_distributions.lane_initialize_weighted(
                    n_lanes=self.run_config.n_lanes,
                    probs=self.run_config.lane_density  # type: ignore
                ),
                location=0,  # Start at the beginning of the road
                speed=driver_distributions.speed_initialize(
                    driver_type=drv_type,
                    car_type=car_type,
                    size=1,
                    max_speed_fixed=self.run_config.max_speed
                ),
            ),
        )
        self.info['active_drivers'][self.id_counter] = new_driver
        self.id_counter += 1
        self.stats = ModelStats(model=self)


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

    @property
    def stats(self) -> "ModelStats":
        return self._stats

    @stats.setter
    def stats(self, stats):
        self._stats = stats

    def run(self):
        for t in range(self.run_config.time_steps):
            self.trace.add(copy.deepcopy(self.model))
            state = Driver.classify_by_lane(self.trace.last.active_drivers)
            new_state_drivers = []
            for driver in self.model.active_drivers:
                driver.action(
                    state=state,  # type: ignore
                    update_fn=driver_distributions.speed_update,
                    max_speed_fixed=self.run_config.max_speed,
                )
                new_state_drivers.append(driver)
                # Check if driver has reached the end of the road
                if driver.config.location >= self.model.road.length:
                    self.exit_driver(driver)
            self.model.info['active_drivers'] = Driver.classify_by_id(
                new_state_drivers
            )  # Still need to check if driver is active

    def exit_driver(self, driver: Driver) -> None:
        """
        This method is called whenever a driver reaches
        the end of the road.
        """
        # TODO: set driver as inactive
        pass


class Trace:
    def __init__(self, *args, **kwargs):
        self.data = []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def last(self) -> 'Model':
        return self.data[-1]

    def add(self, data):
        self.data.append(data)


class ModelStats:
    def __init__(self, *args, **kwargs):
        if 'model' not in kwargs:
            raise Exception("Cannot create ModelStats without a model.")
        self._model = kwargs['model']

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, model: Model):
        self._model = model

    def get_stats(self):
        """
        Returns a dictionary of statistics about the model.
        """
        pass
