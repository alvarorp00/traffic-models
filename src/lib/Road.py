"""

"""


from typing import List
from lib.Driver import Driver


class Road:
    def __init__(self, length: float, lanes: int, **kwargs):
        """
        Constructor for the Road class.

        Parameters
        ----------
        length : float
            The length of the road in meters.
        lanes : int
            The number of lanes on the road.
        """
        self.length = length
        self.lanes = lanes
        self.drivers = []

    @property
    def drivers(self) -> List[Driver]:
        return self.drivers

    @drivers.setter
    def drivers(self, drivers):
        self.drivers = drivers

    def add_driver(self, driver):
        self.drivers.append(driver)

    def del_driver(self, driver):
        self.drivers.remove(driver)

    @property
    def length(self):
        return self.length

    @length.setter
    def length(self, length):
        self.length = length

    @property
    def lanes(self):
        return self.lanes

    @lanes.setter
    def lanes(self, lanes):
        self.lanes = lanes
