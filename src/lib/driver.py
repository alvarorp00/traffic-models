"""
This class defines the interface that all drivers must implement. It
also provides a few utility methods that are useful for all drivers.

The driver interface is designed to be as simple as possible. The
action() method will be called to act on a discrete time step,
where the time step is defined by the simulation.

The action() method would depend on the parameters of the driver
which will be set depending on the type of driver. For example, a
driver that takes risks will consider overtaking other vehicles
within less distance than a driver that is more cautious.

NOTE: above statement might change.

Almost all the methods in this class will be implemented by itself,
but the parameters of the driver will be set by the simulation
and so are required to be passed as arguments to the constructor.
"""

import enum
import random
from typing import Dict, List, Union
import numpy as np
import scipy.stats as st


class LanePriority(enum.Enum):
    """
    Represents the priority of a lane. 2 priorities are defined:
    LEFT, RIGHT

    It is just a visual representation that will be used to
    plot the drivers according to their lane priority.
    """
    LEFT = 0,
    RIGHT = 1

    @staticmethod
    def random(size=1) -> list['LanePriority']:
        """
        Returns a random lane priority.

        Probability distribution is defined by a multinomial
        distribution.

        Probability of LEFT is 0.5 and probability of RIGHT is 0.5.

        Returns
        -------
        LanePriority
            A random lane priority.
        """
        choice = random.choices(
            population=list(LanePriority),
            weights=[.5, .5],
            k=size
        )

        return choice


class CarType(enum.Enum):
    """
    Enum for the type of car. 4 types are defined:
    """
    TRUCK = 1
    SUV = 2
    SEDAN = 3
    MOTORCYCLE = 4

    @staticmethod
    def random(size=1) -> list['CarType']:
        """
        Returns a random car type.

        Probability distribution is defined by a multinomial
        distribution.

        Returns
        -------
        CarType
            A random car type.
        """
        choice = random.choices(
            population=list(CarType),
            weights=[.2, .4, .3, .1],
            k=size
        )

        return choice

    @staticmethod
    def random_lognormal(size=1) -> list['CarType']:
        """
        Returns a random car type.

        Probability distribution is defined by a log-normal
        distribution.

        Returns
        -------
        CarType
            A random car type.
        """
        stats = st.lognorm.rvs(0.5, size=int(1e5))

        # Create the histogram
        hist, _ = np.histogram(stats, bins=5)

        # Get probs
        probs = np.linalg.norm(np.array(hist))

        choice = random.choices(
            population=list(CarType),
            weights=probs,
            k=size
        )

        # return choice
        return choice

    @staticmethod
    def get_max_speed(car_type: 'CarType') -> float:
        """
        Returns the maximum speed of the car.

        Parameters
        ----------
        car_type : CarType
            The type of car.

        Returns
        -------
        float
            The maximum speed of the car.
        """
        if car_type == CarType.MOTORCYCLE:
            return 130
        elif car_type == CarType.SEDAN:
            return 120
        elif car_type == CarType.SUV:
            return 100
        elif car_type == CarType.TRUCK:
            return 80
        else:
            raise ValueError(f"Invalid car type @ {car_type}")

    @staticmethod
    def get_min_speed(car_type: 'CarType') -> float:
        """
        Returns the minimum speed of the car.

        Parameters
        ----------
        car_type : CarType
            The type of car.

        Returns
        -------
        float
            The minimum speed of the car.
        """
        return 60  # All cars have a minimum speed of 60 km/h

    @staticmethod
    def max_speed() -> float:
        """
        Returns the maximum speed of any car.

        Returns
        -------
        float
            The maximum speed of any car.
        """
        return CarType.get_max_speed(CarType.MOTORCYCLE)


class DriverType(enum.Enum):
    """
    Enum for the type of driver. 5 types are defined:
    """
    CAUTIOUS = 1
    NORMAL = 2
    RISKY = 3
    AGGRESSIVE = 4
    RECKLESS = 5

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def random(size=1) -> list['DriverType']:
        """
        Returns a random driver type.

        Probability distribution is defined by a multinomial
        distribution.

        Probabilities:
        - CAUTIOUS: 0.4
        - NORMAL: 0.3
        - RISKY: 0.15
        - AGGRESSIVE: 0.1
        - RECKLESS: 0.05

        Returns
        -------
        DriverType
            A random driver type.
        """
        choice = random.choices(
            population=list(DriverType),
            weights=[.4, .3, .15, .1, .05],
            k=size
        )

        return choice

    @staticmethod
    def random_lognormal(size=1) -> list['DriverType']:
        """
        Returns a random driver type.

        Probability distribution is defined by a log-normal
        distribution.

        Returns
        -------
        DriverType
            A random driver type.
        """
        stats = st.lognorm.rvs(0.5, size=int(1e5))

        # Create the histogram
        hist, _ = np.histogram(stats, bins=5)

        # Get probs by normalizing
        probs = np.array(hist) / np.sum(np.array(hist))

        choice = random.choices(
            population=list(DriverType),
            weights=probs,
            k=size
        )

        # return choice
        return choice

    @staticmethod
    def from_int(int) -> 'DriverType':
        return DriverType(int)

    @staticmethod
    def as_index(driver_type: 'DriverType') -> int:
        """
        Returns the index of the driver type.

        Parameters
        ----------
        driver_type : DriverType
            The driver type.

        Returns
        -------
        int
            The index of the driver type.
        """
        return driver_type.value - 1


class DriverConfig:
    """
    Configuration for a driver.

    This class is used to configure the parameters of a driver. It
    is used by the simulation to create a new driver.

    Attributes
    ----------
    driver_type : DriverType
        The type of driver.
    car_type : CarType
        The type of car.
    location : float
        The initial location of the driver.
    speed : float
        The initial speed of the driver.
    lane : int
    """
    def __init__(self, id, **kwargs):
        """
        Constructor for the DriverConfig class.

        Parameters
        ----------
        driver_type : DriverType
            The type of driver.
        """
        self.id = id

        if 'driver_type' not in kwargs:
            raise ValueError("driver_type not passed to the constructor")
        self.driver_type = kwargs['driver_type']

        if 'car_type' not in kwargs:
            raise ValueError("car_type not passed to the constructor")
        self.car_type = kwargs['car_type']

        if 'location' in kwargs:
            # assert isinstance(kwargs['location'], float)
            self.location = kwargs['location']
        else:
            self.location = 0

        if 'speed' in kwargs:
            # assert isinstance(kwargs['speed'], float)
            self.speed = kwargs['speed']
        else:
            self.speed = 0

        if 'lane' in kwargs:
            assert isinstance(kwargs['lane'], int)
            self.lane = kwargs['lane']
        else:
            self.lane = 0

    @property
    def driver_type(self) -> DriverType:
        return self._driver_type

    @driver_type.setter
    def driver_type(self, driver_type: DriverType):
        self._driver_type = driver_type

    @property
    def car_type(self) -> CarType:
        return self._car_type

    @car_type.setter
    def car_type(self, car_type: CarType):
        self._car_type = car_type

    @property
    def location(self) -> float:
        return self._location

    @location.setter
    def location(self, location: float):
        self._location = location

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, speed: float):
        self._speed = speed

    @property
    def lane(self) -> int:
        return self._lane

    @lane.setter
    def lane(self, lane: int):
        self._lane = lane


class Driver:
    def __init__(self, *args, **kwargs):
        """
        Constructor for the driver class.

        This method will be called by the simulation to create a new
        driver. It is expected to set the parameters of the driver
        based on the parameters passed to it.

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters of the driver.
            --> "config": DriverConfig
        """
        if 'config' not in kwargs:
            raise ValueError("config not passed to the constructor")
        assert isinstance(kwargs['config'], DriverConfig)
        self.config = kwargs['config']

    @property
    def config(self) -> DriverConfig:
        return self._config

    @config.setter
    def config(self, config: DriverConfig):
        self._config = config

    def action(self, state, **kwargs):
        # TODO
        return NotImplementedError("action() not implemented yet")


class Utils:
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

    @staticmethod
    def classify_by_lane(drivers: list[Driver]) -> Dict[int, List[Driver]]:
        """
        Returns a dictionary that maps lane numbers to a list of drivers
        in that lane.
        """

        dict = {}

        for d in drivers:
            if d.config.lane in dict.keys():
                dict[d.config.lane].append(d)
            else:
                dict[d.config.lane] = [d]

        return dict

    @staticmethod
    def sort_by_position(
        drivers_by_lane: Dict[int, List[Driver]]
    ) -> List[Driver]:
        """
        Returns a list of drivers sorted by their position on the track.
        """
        # Collect all drivers into a list
        drivers = []
        for drivers_in_lane in drivers_by_lane.values():
            drivers += drivers_in_lane

        # Sort by location
        drivers.sort(key=lambda d: d.config.location)

        return drivers

    @staticmethod
    def sort_by_position_in_lane(
        drivers_by_lane: Dict[int, List[Driver]]
    ) -> Dict[int, List[Driver]]:
        """
        Returns a dictionary that maps lane numbers to a list of drivers
        in that lane, sorted by their position in the lane.
        """
        ret = {}
        for lane, drivers in drivers_by_lane.items():
            ret[lane] = sorted(drivers, key=lambda d: d.config.location)
        return ret

    @staticmethod
    def max_close_distance(driver_type: DriverType) -> float:
        """
        Returns the maximum a car can be from the car in front of it
        depending on the driver type.
        """
        MAXIMUM = 25  # 25 meters in the case of driver_type == CAUTIOUS
        GAP = 5  # 5 meters per driver_type
        return MAXIMUM - GAP * (driver_type.value - 1)

    @staticmethod
    def driver_at_front(
        driver: Driver,
        drivers_by_lane: Dict[int, List[Driver]]
    ) -> Union[Driver, None]:
        """
        Returns the driver in front of the given driver in the same lane.
        If the given driver is at the front of the lane, returns None.
        """

        # Get the list of drivers in the same lane
        drivers_in_lane = drivers_by_lane[driver.config.lane]

        # Sort by location
        drivers_in_lane.sort(key=lambda d: d.config.location)

        # Find the given driver in the list
        index = drivers_in_lane.index(driver)

        # If the given driver is at the front of the lane, return None
        if index == len(drivers_in_lane) - 1:
            return None

        # Otherwise, return the driver in front
        return drivers_in_lane[index + 1]

    @staticmethod
    def safe_overtake(
        driver: Driver,
        drivers_by_lane: Dict[int, List[Driver]],
        n_lanes: int
    ) -> bool:
        """
        Returns whether the given driver can safely overtake the driver

        It checks whether the driver can move to the next lane, i.e.
        if there are no cars close to the driver in the next lane, taking into
        account the type of driver and current speed; and whether the driver
        is moving faster than the driver in front of it.

        Parameters
        ----------
        driver : Driver
            The driver that wants to overtake.
        drivers_by_lane : Dict[int, List[Driver]]
            A dictionary that maps lane numbers to a list of drivers
            in that lane.
        n_lanes : int
            The number of lanes in the track.
        """
        # Check that the current lane is not the rightmost lane
        if driver.config.lane == n_lanes - 1:
            return False
        # TODO
        # Get driver in front
        # Check if driver is moving faster than driver in front
        # Check if driver can move to next lane
        # Check if there are no cars close to the driver in the next lane

    @staticmethod
    def safe_return(
        driver: Driver,
        drivers_by_lane: Dict[int, List[Driver]],
        n_lanes: int
    ) -> bool:
        """
        Returns whether the given driver can safely return to previous lane.

        It checks whether the driver can move to the previous lane, i.e.
        if there are no cars close to the driver in the previous lane, taking
        into account the type of driver and current speed; and whether the
        driver is moving slower than the driver behind it.
        """
        # Check that the current lane is not the leftmost lane
        if driver.config.lane == 0:
            return False
        # TODO
        # Get driver in back
        # Check if driver is moving slower than driver in back
        # Check if driver can move to previous lane
        # Check if there are no cars close to the driver in the previous lane
