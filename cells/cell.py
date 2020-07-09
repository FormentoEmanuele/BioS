from neuron import h


class Cell(object):
    """ Interface class to build different types of cells. """

    def __init__(self):
        """ Object initialization. """
        # Neuron cell - to be used as the source of a h.NetCon
        self.cell = None

    def connect_to_target(self, target, weight: float = 0, delay: float = 1, **kwargs) -> h.NetCon:
        """ Connect the current cell to a target cell and return the netCon object.

        Keyword arguments:
        target -- the target object to which we want to connect
        weight -- the weight of the connection (default 0)
        delay -- communication time delay in ms (default 1)
        """

        nc = h.NetCon(self.cell, target)
        nc.delay = delay
        nc.weight[0] = weight
        return nc

    def is_artificial(self) -> bool:
        """ Return a flag to check whether the cell is an integrate-and-fire or artificial cell.

        By default the flag is set to True.
        """
        return True
