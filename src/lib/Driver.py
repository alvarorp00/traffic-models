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


class CarType(enum.Enum):
    """
    Enum for the type of car. 4 types are defined:
    """
    MOTORCYCLE = 1
    SEDAN = 2
    SUV = 3
    TRUCK = 4

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


class DriverType(enum.Enum):
    """
    Enum for the type of driver. 5 types are defined:
    """
    CAUTIOUS = 1
    NORMAL = 2
    RISKY = 3
    AGGRESSIVE = 4
    RECKLESS = 5


class DriverConfig:
    """
    Configuration for a driver.

    This class is used to configure the parameters of a driver. It
    is used by the simulation to create a new driver.

    Attributes
    ----------
    driver_type : DriverType
        The type of driver.
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
