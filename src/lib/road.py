"""

"""


from typing import Dict
from lib.driver import Driver


class Road:
    def __init__(self, **kwargs):
        """
        Constructor for the Road class.

        Parameters
        ----------
        length : float
            The length of the road in meters.
        lanes : int
            The number of lanes on the road.
        """
        self.length = kwargs['length']
        self.n_lanes = kwargs['n_lanes']
        self.drivers = {}

    @property
    def drivers(self) -> Dict[int, Driver]:
        return self._drivers

    @drivers.setter
    def drivers(self, drivers: Dict[int, Driver]):
        self._drivers = drivers

    def add_driver(self, driver: Driver):
        self.drivers[driver.config.id] = driver

    def del_driver(self, driver: Driver):
        self.drivers.pop(driver.config.id)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    @property
    def n_lanes(self):
        return self._n_lanes

    @n_lanes.setter
    def n_lanes(self, n_lanes):
        self._n_lanes = n_lanes
