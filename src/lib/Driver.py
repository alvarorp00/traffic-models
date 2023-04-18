"""
Abstract base class for all drivers.

This class defines the interface that all drivers must implement. It
also provides a few utility methods that are useful for all drivers.

The driver interface is designed to be as simple as possible. The
action() method will be called to act on a discrete time step,
where the time step is defined by the simulation.

The action() method would depend on the parameters of the driver
which will be set depending on the type of driver. For example, a
driver that takes risks will consider overtaking other vehicles
within less distance than a driver that is more cautious.

Almost all the methods in this class will be implemented by itself,
but the parameters of the driver will be set by the simulation
and so are required to be passed as arguments to the constructor.
"""

import enum
from typing import List
import numpy as np
from scipy import stats as st


class CarType(enum.Enum):
    """
    Enum for the type of car. 4 types are defined:
    """
    MOTORCYCLE = 1
    SEDAN = 2
    SUV = 3
    TRUCK = 4

    @staticmethod
    def random() -> 'CarType':
        """
        Returns a random car type.

        Returns
        -------
        CarType
            A random car type.
        """
        return np.random.choice(
                    a=list(CarType),
                    p=[.2, .4, .3, .1]
                )

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
            raise ValueError("Invalid car type")

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


class DriverDistributions:
    @staticmethod
    def risk_overtake_distance(driver: "Driver", size=1):
        """
        Returns a random rample gathered from a halfnormal distribution
        (a normal distribution with only right side values so we control
        the minimum distances that a driver will consider overtaking)
        that represents the minimum distance that a driver will
        consider overtaking another vehicle. This distance will be
        considered both for overtaking in front and the car comming
        from behind on left lane.

        It considers the type of driver and the type of car.

        Parameters
        ----------
        driver_type : DriverType
            The type of driver.
        car_type : CarType
            The type of car.
        size : int

        Formula:
            mean = 25 - 5 * (driver_type.value - 1)
            std = 1 / (CarType.get_max_speed(car_type) / CarType.max_speed())

        Returns
        -------
        ndarray: A random sample from the normal distribution.

        Notes:
            Mean --> 25 meters for CAUTIOUS driver, decreasing by 5 meters
            for each driver type.

            Standard deviation --> 1 /
                                    (CarType.get_max_speed(car_type) /
                                    CarType.max_speed()),
            For std is considered the current speed of the car and
            the maximum speed it can reach, so that the standard deviation
            is 1 meter for the fastest car and increases as the car
            gets slower, which has sense as the

            Size --> The number of samples to be generated. For example, two
                     samples: one to the car in front and one to the car in
                     the left lane.
        """
        MINIMUM = 25  # 25 meters in the case of driver_type == CAUTIOUS
        GAP = 5  # 5 meters per driver_type
        mean = MINIMUM - GAP * (driver.config.driver_type.value - 1)
        std = 1 / (driver.config.speed
                   /
                   CarType.get_max_speed(driver.config.car_type))
        rvs = st.halfnorm.rvs(loc=mean, scale=std, size=size)
        return rvs


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
            assert isinstance(kwargs['location'], float)
            self.location = kwargs['location']
        else:
            self.location = 0

        if 'speed' in kwargs:
            assert isinstance(kwargs['speed'], float)
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
