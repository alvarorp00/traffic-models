"""

"""


class Road:
    def __init__(self, length, lanes, **kwargs):
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
    def drivers(self):
        return self.drivers

    def add_driver(self, driver):
        self.drivers.append(driver)

    @property
    def length(self):
        return self.length

    @property
    def lanes(self):
        return self.lanes
