import abc
import os
import time
from neuron import h

from simulations.constants import TEMPERATURE_C, RESULT_FOLDER


class Simulation(object):
    """ Interface class to design different types of neuronal simulation.

    The simulations are based on the python Neuron module. MPI is not supported.
    """

    def __init__(self):

        # Set the temperature in Neuron
        h.celsius = TEMPERATURE_C
        # Set the Neuron integration dt in ms
        h.dt = 0.025
        # To plot results with a high temporal resolution we use an integration step equal to the Neuron dt
        self.__integration_step = h.dt
        self.__t_stop = None
        self.__log_period = 250  # ms
        self.simulation_time = None
        self.set_results_folder(RESULT_FOLDER)

    def __check_parameters(self):
        """ Check whether some parameters necessary for the simulation have been set or not. """
        if self.__t_stop is None:
            raise Exception("Undefined maximum time of simulation")

    def _get_t_stop(self):
        """ Return the time at which we want to stop the simulation. """
        return self.__t_stop

    def _set_t_stop(self, t_stop: float):
        """ Set the time at which we want to stop the simulation.

        Keyword arguments:
        t_stop -- time at which we want to stop the simulation in ms.
        """
        if t_stop > 0:
            self.__t_stop = t_stop
        else:
            raise Exception("The maximum time of simulation has to be greater than 0")

    def _get_integration_step(self):
        """ Return the integration time step. """
        return self.__integration_step

    def _set_integration_step(self, dt: float):
        """ Set the integration time step. This will be rounded to a multiple of h.dt

        Keyword arguments:
        dt -- integration time step in ms.
        """
        if dt > 0:
            self.__integration_step = dt
        else:
            raise Exception("The integration step has to be greater than 0")

    def _initialize(self):
        """ Initialize the simulation.

        Set the __maxStep varibale and initialize the membrane potential of real cell to -70mV.
        """
        h.finitialize(-69.35)
        self._start = time.time()
        self.__t_log = 0

    def _integrate(self):
        """ Integrate the neuronal cells for a defined integration time step ."""
        for i in range(int(self.__integration_step//h.dt)):
            h.fadvance()

    def _update(self):
        """ Update simulation parameters. """
        raise Exception("pure virtual function")

    def _get_print_period(self):
        """ Return the period of time between printings to screen. """
        return self.__log_period

    def _set_print_period(self, t):
        """ Set the period of time between printings to screen.

        Keyword arguments:
        t -- period of time between printings in ms.
        """
        if t > 0:
            self.__log_period = t
        else:
            raise Exception("The print period has to be greater than 0")

    def _print_sim_status(self):
        """ Print to screen the simulation state. """
        if h.t - self.__t_log >= (self.__log_period - 0.5 * self.__integration_step):
            if self.__t_log == 0:
                print("\nStarting simulation:")
            self.__t_log = h.t
            print("\t" + str(round(h.t)) + "ms of " + str(self.__t_stop) + "ms integrated...")

    def _end_integration(self):
        """ Print the total simulation time.

        This function, executed at the end of time integration is meant to be modified
        by daughter classes according to specific needs.
        """
        self.simulation_time = time.time() - self._start
        print("tot simulation time: " + str(int(self.simulation_time)) + "s")

    def run(self):
        """ Run the simulation. """
        self.__check_parameters()
        self._initialize()
        while h.t < self.__t_stop:
            self._integrate()
            self._update()
            self._print_sim_status()
        self._end_integration()

    def set_results_folder(self, results_folder_path):
        """ Set a new folder in which to save the results """
        self._results_folder = results_folder_path
        if not os.path.exists(self._results_folder):
            os.makedirs(self._results_folder)

    @abc.abstractmethod
    def save_results(self, name=""):
        """ Save the simulation results.

        Keyword arguments:
        name -- string to add at predefined file name (default = "").
        """
        pass

    @abc.abstractmethod
    def plot(self, **kwargs):
        """ Plot the simulation results. """
        pass
