from neuron import h
import numpy as np
from cells.cell import Cell


class MyelinatedFiberMcInt(Cell):
    """ Neuron Biophysical myelinated fiber model.

    This model implements the axon model developed by McIntyre 2002.
    """

    def __init__(self, diameter_ind: int = 3):
        """ Object initialization.

        Keyword arguments:
        diameter_ind -- index of the implemented_diameters_um list (default = 3)
        implemented_diameters_um = [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]
        """
        super(MyelinatedFiberMcInt, self).__init__()

        self._init_parameters(diameter_ind)
        self._create_sections()
        self._build_topology()
        self._define_biophysics()

    """
    Specific Methods of this class
    """

    def _init_parameters(self, diameter_ind: int):
        """ Initialize all cell parameters. """

        # topological parameters
        self.n_nodes = 21
        self._axon_nodes = self.n_nodes
        self._para_nodes1 = 40
        self._para_nodes2 = 40
        self._axon_inter = 120
        self._axon_total = 221
        # morphological parameters
        implemented_diameters_um = [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]
        self.fiber_d_um = implemented_diameters_um[diameter_ind]
        self._para_length1 = 3
        self._node_length = 1.0
        self._space_p1 = 0.002
        self._space_p2 = 0.004
        self._space_i = 0.004
        # electrical parameters
        self._rhoa = 0.7e6  # Ohm-um
        self._mycm = 0.1  # uF/cm2/lamella membrane
        self._mygm = 0.001  # S/cm2/lamella membrane

        if self.fiber_d_um == 5.7:
            self._g = 0.605
            self._axon_d = 3.4
            self._node_d = 1.9
            self._para_d1 = 1.9
            self._para_d2 = 3.4
            self._deltax = 500
            self._para_length2 = 35
            self._nl = 80
        if self.fiber_d_um == 8.7:
            self._g = 0.661
            self._axon_d = 5.8
            self._node_d = 2.8
            self._para_d1 = 2.8
            self._para_d2 = 5.8
            self._deltax = 1000
            self._para_length2 = 40
            self._nl = 110
        if self.fiber_d_um == 10.0:
            self._g = 0.690
            self._axon_d = 6.9
            self._node_d = 3.3
            self._para_d1 = 3.3
            self._para_d2 = 6.9
            self._deltax = 1150
            self._para_length2 = 46
            self._nl = 120
        if self.fiber_d_um == 11.5:
            self._g = 0.700
            self._axon_d = 8.1
            self._node_d = 3.7
            self._para_d1 = 3.7
            self._para_d2 = 8.1
            self._deltax = 1250
            self._para_length2 = 50
            self._nl = 130
        if self.fiber_d_um == 12.8:
            self._g = 0.719
            self._axon_d = 9.2
            self._node_d = 4.2
            self._para_d1 = 4.2
            self._para_d2 = 9.2
            self._deltax = 1350
            self._para_length2 = 54
            self._nl = 135
        if self.fiber_d_um == 14.0:
            self._g = 0.739
            self._axon_d = 10.4
            self._node_d = 4.7
            self._para_d1 = 4.7
            self._para_d2 = 10.4
            self._deltax = 1400
            self._para_length2 = 56
            self._nl = 140
        if self.fiber_d_um == 15.0:
            self._g = 0.767
            self._axon_d = 11.5
            self._node_d = 5.0
            self._para_d1 = 5.0
            self._para_d2 = 11.5
            self._deltax = 1450
            self._para_length2 = 58
            self._nl = 145
        if self.fiber_d_um == 16.0:
            self._g = 0.791
            self._axon_d = 12.7
            self._node_d = 5.5
            self._para_d1 = 5.5
            self._para_d2 = 12.7
            self._deltax = 1500
            self._para_length2 = 60
            self._nl = 150

        self._rpn0 = (self._rhoa * .01) / (
                np.pi * ((((self._node_d / 2) + self._space_p1) ** 2) - ((self._node_d / 2) ** 2)))
        self._rpn1 = (self._rhoa * .01) / (
                np.pi * ((((self._para_d1 / 2) + self._space_p1) ** 2) - ((self._para_d1 / 2) ** 2)))
        self._rpn2 = (self._rhoa * .01) / (
                np.pi * ((((self._para_d2 / 2) + self._space_p2) ** 2) - ((self._para_d2 / 2) ** 2)))
        self._rpx = (self._rhoa * .01) / (
                np.pi * ((((self._axon_d / 2) + self._space_i) ** 2) - ((self._axon_d / 2) ** 2)))
        self._inter_length = (self._deltax - self._node_length - (2 * self._para_length1) - (
                    2 * self._para_length2)) / 6

    def _create_sections(self):
        """ Create the sections of the cell. """
        # NOTE: cell=self is required to tell NEURON of this object.
        self.node = [h.Section(name='node', cell=self) for x in range(self._axon_nodes)]
        self.mysa = [h.Section(name='mysa', cell=self) for x in range(self._para_nodes1)]
        self.flut = [h.Section(name='flut', cell=self) for x in range(self._para_nodes2)]
        self.stin = [h.Section(name='stin', cell=self) for x in range(self._axon_inter)]

    def _define_biophysics(self):
        """ Assign the membrane properties across the cell. """
        for node in self.node:
            node.nseg = 1
            node.diam = self._node_d
            node.L = self._node_length
            node.Ra = self._rhoa / 10000
            node.cm = 2
            node.insert('axnode')
            node.insert('extracellular')
            node.xraxial[0] = self._rpn0
            node.xg[0] = 1e10
            node.xc[0] = 0

        for mysa in self.mysa:
            mysa.nseg = 1
            mysa.diam = self.fiber_d_um
            mysa.L = self._para_length1
            mysa.Ra = self._rhoa * (1 / (self._para_d1 / self.fiber_d_um) ** 2) / 10000
            mysa.cm = 2 * self._para_d1 / self.fiber_d_um
            mysa.insert('pas')
            mysa.g_pas = 0.001 * self._para_d1 / self.fiber_d_um
            mysa.e_pas = -80
            mysa.insert('extracellular')
            mysa.xraxial[0] = self._rpn1
            mysa.xg[0] = self._mygm / (self._nl * 2)
            mysa.xc[0] = self._mycm / (self._nl * 2)

        for flut in self.flut:
            flut.nseg = 1
            flut.diam = self.fiber_d_um
            flut.L = self._para_length2
            flut.Ra = self._rhoa * (1 / (self._para_d2 / self.fiber_d_um) ** 2) / 10000
            flut.cm = 2 * self._para_d2 / self.fiber_d_um
            flut.insert('pas')
            flut.g_pas = 0.0001 * self._para_d2 / self.fiber_d_um
            flut.e_pas = -80
            flut.insert('extracellular')
            flut.xraxial[0] = self._rpn2
            flut.xg[0] = self._mygm / (self._nl * 2)
            flut.xc[0] = self._mycm / (self._nl * 2)

        for stin in self.stin:
            stin.nseg = 1
            stin.diam = self.fiber_d_um
            stin.L = self._inter_length
            stin.Ra = self._rhoa * (1 / (self._axon_d / self.fiber_d_um) ** 2) / 10000
            stin.cm = 2 * self._axon_d / self.fiber_d_um
            stin.insert('pas')
            stin.g_pas = 0.0001 * self._axon_d / self.fiber_d_um
            stin.e_pas = -80
            stin.insert('extracellular')
            stin.xraxial[0] = self._rpx
            stin.xg[0] = self._mygm / (self._nl * 2)
            stin.xc[0] = self._mycm / (self._nl * 2)

    def _build_topology(self):
        """ connect the sections together """
        # childSection.connect(parentSection, [parentX], [childEnd])
        for i in range(self._axon_nodes - 1):
            self.node[i].connect(self.mysa[2 * i], 0, 1)
            self.mysa[2 * i].connect(self.flut[2 * i], 0, 1)
            self.flut[2 * i].connect(self.stin[6 * i], 0, 1)
            self.stin[6 * i].connect(self.stin[6 * i + 1], 0, 1)
            self.stin[6 * i + 1].connect(self.stin[6 * i + 2], 0, 1)
            self.stin[6 * i + 2].connect(self.stin[6 * i + 3], 0, 1)
            self.stin[6 * i + 3].connect(self.stin[6 * i + 4], 0, 1)
            self.stin[6 * i + 4].connect(self.stin[6 * i + 5], 0, 1)
            self.stin[6 * i + 5].connect(self.flut[2 * i + 1], 0, 1)
            self.flut[2 * i + 1].connect(self.mysa[2 * i + 1], 0, 1)
            self.mysa[2 * i + 1].connect(self.node[i + 1], 0, 1)

    """
    Redefinition of inherited methods
    """

    def connect_to_target(self, target, weight: float = 0, delay: float = 0, node_n: int = -1, **kwargs) -> h.NetCon:
        nc = h.NetCon(self.node[node_n](1)._ref_v, target, sec=self.node[node_n])
        nc.threshold = -30
        nc.delay = delay
        return nc

    def is_artificial(self) -> bool:
        """ Return a flag to check whether the cell is an integrate-and-fire or artificial cell. """
        return False
