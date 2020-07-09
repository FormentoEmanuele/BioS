from neuron import h
import numpy as np
import matplotlib.pyplot as plt
from cells.myelinated_fiber_mcint import MyelinatedFiberMcInt


class MyelinatedFiberMcIntContinuous(MyelinatedFiberMcInt):
    """ Neuron Biophysical myelinated fiber model.

    This model extend the model developed by McIntyre 2002 to allow generating fibers with any diameter in the range of
    6 to 20 um (range of diameters for which model parameters are known). A 2nd degree polynomial fit is used to
    estimate model parameters as a function of the fiber's diameter.
    """
    MAX_DIAMETER_UM = 20
    MIN_DIAMETER_UM = 6
    INTERPOLATION_DEGREE = 2

    def __init__(self, diameter_um: float = 10):
        """ Object initialization.

        Keyword arguments:
        diameter_um -- fiber diameter_um in micrometers [5-20]
        """
        if not self.MIN_DIAMETER_UM <= diameter_um <= self.MAX_DIAMETER_UM:
            raise ValueError(f"Fiber's diameter need to be within the range [{self.MIN_DIAMETER_UM}-"
                             f"{self.MAX_DIAMETER_UM}] um. A diameter of {diameter_um} um was provided.")

        super(MyelinatedFiberMcInt, self).__init__()
        self._init_parameters(diameter_um)
        self._create_sections()
        self._build_topology()
        self._define_biophysics()

    def _init_parameters(self, diameter_um: float):
        """ Initialize all cell parameters. """

        # topological parameters
        self.n_nodes = 41
        self._axon_nodes = self.n_nodes
        self._para_nodes1 = 80
        self._para_nodes2 = 80
        self._axon_inter = 240
        self.axon_total = 441
        # morphological parameters
        self.fiber_d_um = diameter_um
        self._para_length1 = 3
        self._node_length = 1.0
        self._space_p1 = 0.002
        self._space_p2 = 0.004
        self._space_i = 0.004
        # electrical parameters
        self._rhoa = 0.7e6  # Ohm-um
        self._mycm = 0.1  # uF/cm2/lamella membrane
        self._mygm = 0.001  # S/cm2/lamella membrane

        # fit the parameters with polynomials to allow any diameter, values in um
        self._experimental_diameters = [5.7, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]
        self._experimental_g = [0.605, 0.661, 0.690, 0.700, 0.719, 0.739, 0.767, 0.791]
        self._experimental_axon_d = [3.4, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7]
        self._experimental_node_d = [1.9, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5]
        self._experimental_para_d1 = [1.9, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5]
        self._experimental_para_d2 = [3.4, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7]
        self._experimental_delta_x = [500, 1000, 1150, 1250, 1350, 1400, 1450, 1500]
        self._experimental_para_length2 = [35, 40, 46, 50, 54, 56, 58, 60]
        self._experimental_nl = [80, 110, 120, 130, 135, 140, 145, 150]
        self._fit_g = np.poly1d(np.polyfit(self._experimental_diameters,
                                           self._experimental_g, self.INTERPOLATION_DEGREE))
        self._fit_axond = np.poly1d(
            np.polyfit(self._experimental_diameters, self._experimental_axon_d, self.INTERPOLATION_DEGREE))
        self._fit_node_d = np.poly1d(
            np.polyfit(self._experimental_diameters, self._experimental_node_d, self.INTERPOLATION_DEGREE))
        self._fit_para_d1 = np.poly1d(
            np.polyfit(self._experimental_diameters, self._experimental_para_d1, self.INTERPOLATION_DEGREE))
        self._fit_para_d2 = np.poly1d(
            np.polyfit(self._experimental_diameters, self._experimental_para_d2, self.INTERPOLATION_DEGREE))
        self._fit_deltax = np.poly1d(
            np.polyfit(self._experimental_diameters, self._experimental_delta_x, self.INTERPOLATION_DEGREE))
        self._fit_para_length2 = np.poly1d(
            np.polyfit(self._experimental_diameters, self._experimental_para_length2, self.INTERPOLATION_DEGREE))
        self._fit_nl = np.poly1d(np.polyfit(self._experimental_diameters, self._experimental_nl,
                                            self.INTERPOLATION_DEGREE))
        # interpolate
        self._g = self._fit_g(self.fiber_d_um)
        self._axon_d = self._fit_axond(self.fiber_d_um)
        self._node_d = self._fit_node_d(self.fiber_d_um)
        self._para_d1 = self._fit_para_d1(self.fiber_d_um)
        self._para_d2 = self._fit_para_d2(self.fiber_d_um)
        self._deltax = self._fit_deltax(self.fiber_d_um)
        self._para_length2 = self._fit_para_length2(self.fiber_d_um)
        self._nl = self._fit_nl(self.fiber_d_um)

        self._rpn0 = (self._rhoa * .01) / (
                    np.pi * ((((self._node_d / 2) + self._space_p1) ** 2) - ((self._node_d / 2) ** 2)))
        self._rpn1 = (self._rhoa * .01) / (
                    np.pi * ((((self._para_d1 / 2) + self._space_p1) ** 2) - ((self._para_d1 / 2) ** 2)))
        self._rpn2 = (self._rhoa * .01) / (
                    np.pi * ((((self._para_d2 / 2) + self._space_p2) ** 2) - ((self._para_d2 / 2) ** 2)))
        self._rpx = (self._rhoa * .01) / (
                    np.pi * ((((self._axon_d / 2) + self._space_i) ** 2) - ((self._axon_d / 2) ** 2)))
        self._inter_length = (self._deltax - self._node_length - (2 * self._para_length1) - (2 * self._para_length2)) / 6

        self.node_to_node_distance = (self._node_length + 2 * self._para_length1 + 2 * self._para_length2 +
                                      6 * self._inter_length)
        self.total_fiber_length = (self._node_length * self._axon_nodes + self._para_nodes1 * self._para_length1 +
                                   self._para_nodes2 * self._para_length2 + self._axon_inter * self._inter_length)

    def plot_fits(self):
        f, ax = plt.subplots(2, 4)
        diameters = np.linspace(3, 20, 50)
        ax[0, 0].plot(diameters, self._fit_g(diameters))
        ax[0, 0].plot(self._experimental_diameters, self._experimental_g)
        ax[0, 1].plot(diameters, self._fit_axond(diameters))
        ax[0, 1].plot(self._experimental_diameters, self._experimental_axon_d)
        ax[0, 2].plot(diameters, self._fit_node_d(diameters))
        ax[0, 2].plot(self._experimental_diameters, self._experimental_node_d)
        ax[0, 3].plot(diameters, self._fit_para_d1(diameters))
        ax[0, 3].plot(self._experimental_diameters, self._experimental_para_d1)
        ax[1, 0].plot(diameters, self._fit_para_d2(diameters))
        ax[1, 0].plot(self._experimental_diameters, self._experimental_para_d2)
        ax[1, 1].plot(diameters, self._fit_deltax(diameters))
        ax[1, 1].plot(self._experimental_diameters, self._experimental_delta_x)
        ax[1, 2].plot(diameters, self._fit_para_length2(diameters))
        ax[1, 2].plot(self._experimental_diameters, self._experimental_para_length2)
        ax[1, 3].plot(diameters, self._fit_nl(diameters))
        ax[1, 3].plot(self._experimental_diameters, self._experimental_nl)
        plt.show()

    def _create_sections(self):
        """ Create the sections of the cell. """
        # NOTE: cell=self is required to tell NEURON of this object.
        self.node = [h.Section(name='node', cell=self) for x in range(self._axon_nodes)]
        self.mysa = [h.Section(name='mysa', cell=self) for x in range(self._para_nodes1)]
        self.flut = [h.Section(name='flut', cell=self) for x in range(self._para_nodes2)]
        self.stin = [h.Section(name='stin', cell=self) for x in range(self._axon_inter)]
        self.segments = []
        for i, node in enumerate(self.node):
            self.segments.append([node, self._node_length / 2 + i * (
                        self._node_length + 2 * self._para_length1 + 2 * self._para_length2 + 6 * self._inter_length)])
        for i, mysa in enumerate(self.mysa):
            if i % 2 == 0:
                self.segments.append([mysa, self._node_length + self._para_length1 / 2 + (round(i / 2)) * (
                            self._node_length + 2 * self._para_length1 + 2 * self._para_length2 + 6 *
                            self._inter_length)])
            else:
                self.segments.append([mysa, self._node_length + self._para_length1 + 2 * self._para_length2 +
                                      6 * self._inter_length + self._para_length1 / 2 + (round(i / 2)) * (
                                              self._node_length + 2 * self._para_length1 + 2 * self._para_length2 + 6 *
                                              self._inter_length)])
        for i, flut in enumerate(self.flut):
            if i % 2 == 0:
                self.segments.append([flut, self._node_length + self._para_length2 + self._para_length2 / 2 +
                                      (round(i / 2)) * (self._node_length + 2 * self._para_length1 + 2 *
                                                        self._para_length2 + 6 * self._inter_length)])
            else:
                self.segments.append([flut, self._node_length + self._para_length1 + self._para_length2 + 6 *
                                      self._inter_length + self._para_length2 / 2 + (round(i / 2)) * (
                                              self._node_length + 2 * self._para_length1 + 2 *
                                              self._para_length2 + 6 * self._inter_length)])
        for i, stin in enumerate(self.stin):
            self.segments.append([stin,
                                  self._node_length + self._para_length1 + self._para_length2 + i % 6 *
                                  self._inter_length + self._inter_length / 2 + (
                                      round(i / 6)) * (self._node_length + 2 * self._para_length1 + 2 *
                                                       self._para_length2 + self._axon_inter * self._inter_length)])


if __name__ == '__main__':
    aCell = MyelinatedFiberMcIntContinuous()
    aCell.plot_fits()
