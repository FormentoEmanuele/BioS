from neuron import h
from cells.myelinated_fiber_mcint_continuous import MyelinatedFiberMcIntContinuous
from simulations.afferent_natural import AfferentNatural
from simulations.constants import DIAG_CONDUCTIVITY, MIN_FIBER_DIAMETER, MAX_FIBER_DIAMETER, RECORDING_SAMPLING_RATE, \
    TEMPERATURE_C
from simulations.simulation import Simulation
from tools import firings_tools as frt
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from typing import List, Optional, Union
from tools.general_tools import get_truncated_normal
from simulations.parameters import FiberParameters, StimParameters


class AfferentStimulation(Simulation):
    """ Simulation to evaluate the effect of electrical stimulation on a population of myelinated fibers.
    """

    FIXED_YLIMS = True
    PLOT_OFFSET = 2
    MAX_Y_MAV = 5
    MAX_Y_P2P = 16
    MAX_Y_STD = 5
    MAX_Y_REC = 100
    MAX_Y_NAP = 2

    PRINT_PERIOD = 5
    ELECTRODE_ON_NODE_N = 7
    # Disable the stimulation for this amount of time (ms) at the beginning of the simulation
    START_STIM_TIME_MS = 15.

    def __init__(self, fiber_parameters: FiberParameters, stim_parameters: StimParameters, t_stop: float = 100):

        super(AfferentStimulation, self).__init__()
        np.random.seed(0)  # To replicate simulation results
        self._fiber_parameters: FiberParameters = fiber_parameters
        self._stim_parameters: StimParameters = stim_parameters
        self._set_t_stop(t_stop)
        if self._stim_parameters.bios:
            self._stim_waveform_duration = self._stim_parameters.burst_duration_ms
            if self._stim_parameters.pulse_width_ms > (1000. / self._stim_parameters.burst_frequency) / 2:
                self._stim_parameters.pulse_width_ms = (1000. / self._stim_parameters.burst_frequency) / 2
                print("\nWarning: pulse width too long for burst frequency.\nPulse width set to: {} us\n".format(
                    self._stim_parameters.pulse_width_ms * 1000.))
            if self._stim_parameters.pulse_width_ms < h.dt:
                raise (ValueError("Burst pulse width too low for simulation dt."))
        else:
            self._stim_waveform_duration = self._stim_parameters.pulse_width_ms
        self._mean_node_to_node_distance = MyelinatedFiberMcIntContinuous(self._fiber_parameters.mean_diameter_um
                                                                          ).node_to_node_distance
        self._electrode_offset = self._mean_node_to_node_distance * self.ELECTRODE_ON_NODE_N  # in micro m
        self._stimulation_interval = 1000. / self._stim_parameters.frequency  # in ms
        self._to_samples = RECORDING_SAMPLING_RATE / 1000.

        self._stim_vals = self._stats = self._syn = self._netcons = self._membrane_pot = None
        self._membrane_pot_node_fibers = None
        self._param_str = self.parameters_to_string(stim_parameters=self._stim_parameters,
                                                    fiber_parameters=self._fiber_parameters)

    def _initialize(self):
        self._stim_vals = []
        self._syn = []
        self._netcons = []
        self._time_since_stim_start = 0
        self._step_idx = 0
        self._time_since_last_stim = np.inf
        self._create_fibers()
        self._find_closest_node_to_electrode()
        self._compute_and_set_field(0)
        self._membrane_pot = np.zeros(
            (self.fibers[0].axon_total, int(np.ceil(self._get_t_stop() / self._get_integration_step() + 1))))
        self._membrane_pot_node_fibers = np.nan * np.zeros(
            (self._fiber_parameters.n_fibers, int(np.ceil(self._get_t_stop() / self._get_integration_step() + 1))))
        self._set_print_period(self.PRINT_PERIOD)
        super(AfferentStimulation, self)._initialize()

    def _update(self):
        """ Update simulation parameters. """
        # Gather fibers membrane potential at the closest node to the electrode
        for i, fiber in enumerate(self.fibers):
            if i == 0:
                for j, segment in enumerate(fiber.segments):
                    self._membrane_pot[j, self._step_idx] = segment[0](0.5).v
            self._membrane_pot_node_fibers[i, self._step_idx] = fiber.segments[
                self._closest_node_to_electrode[i]][0](0.5).v
        self._step_idx += 1

        # Control stimulation parameters
        if h.t > self.START_STIM_TIME_MS and self._time_since_last_stim >= self._stimulation_interval \
                and self._time_since_stim_start < self._stim_waveform_duration:
            # Increment counter to see how much time has passed since stim started (current pulse/burst)
            self._time_since_stim_start += self._get_integration_step()
            # stimulate with the right amplitude
            self._compute_and_set_field(self._amplitude_modulation(self._time_since_stim_start))
        elif h.t > self.START_STIM_TIME_MS and self._time_since_stim_start >= self._stim_waveform_duration:
            self._time_since_last_stim = self._time_since_stim_start + self._get_integration_step()
            self._time_since_stim_start = 0
            self._compute_and_set_field(0)
        else:
            self._time_since_last_stim += self._get_integration_step()

    def save_results(self, name=""):
        file_name = time.strftime("%Y_%m_%d_%H_%M_afferentStim_results_" + self._param_str + name + ".p")
        with open(self._results_folder + file_name, 'wb') as pickle_file:
            pickle.dump(self._firings, pickle_file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._stats, pickle_file, pickle.HIGHEST_PROTOCOL)

    def _end_integration(self):
        super(AfferentStimulation, self)._end_integration()
        # Extract action potentials timing
        action_potentials = frt.extract_action_potentials(self._action_potentials,
                                                          self._get_t_stop() * self.PLOT_OFFSET)
        action_potentials_temp = frt.extract_action_potentials(self._action_potentials_end,
                                                               self._get_t_stop() * self.PLOT_OFFSET)

        # Adjust propagation time
        distance_between_spikes = np.array(
            [fiber.node_to_node_distance * self._spike_detectors_distance for fiber in self.fibers])
        actual_length = np.array([fiber.total_fiber_length for fiber in self.fibers])
        desired_length = np.array([self._fiber_parameters.length_um for fiber in self.fibers])
        spike_start_position = np.array(
            [self.fibers[fiber_ind].segments[nodeInd][1] - self.fibers_position[fiber_ind][0] for fiber_ind, nodeInd in
             enumerate(self._closest_node_to_electrode)])
        _adjusted_action_potentials = frt.adjust_propagation_time(action_potentials, action_potentials_temp,
                                                                  distance_between_spikes, actual_length,
                                                                  desired_length,
                                                                  spike_start_position)

        self._firings = frt.extract_firings(_adjusted_action_potentials, self._get_t_stop() * self.PLOT_OFFSET,
                                            RECORDING_SAMPLING_RATE)
        if np.sum(self._firings) != 0:
            self._mean_fr = frt.compute_mean_firing_rate(self._firings, RECORDING_SAMPLING_RATE)
            self._neural_response = frt.compute_neural_response(self._firings, self._to_samples)
            (n_stims, cumulative_firings, response_mav, response_p2p, stim_response, n_activated_fiber,
             average_spikes_per_activated_fiber) = frt.compute_firing_stats(
                self._firings, self.START_STIM_TIME_MS, self._get_t_stop(), self._stimulation_interval,
                self._to_samples,
                self._fiber_parameters.propagation_delay_ms)
            self._stats = {"n_stims": n_stims,
                           "cumulative_firings": cumulative_firings,
                           "response_mav": response_mav,
                           "response_p2p": response_p2p,
                           "stim_response": stim_response,
                           "n_activated_fiber": n_activated_fiber,
                           "average_spikes_per_activated_fiber": average_spikes_per_activated_fiber
                           }

        # Padd stim parameters
        self._stim_vals.append([self._stim_vals[-1][0], 0])
        self._stim_vals.append([self._get_t_stop() * self.PLOT_OFFSET, 0])

        """ Compute natural response stats """
        if np.sum(self._firings) != 0:
            # find the average firing rate
            temp = np.nonzero(np.sum(self._firings, axis=0))[0]
            first_spike = temp[0]
            last_spike = temp[-1]
            self._average_firing_rate = float(np.mean(self._mean_fr[first_spike:last_spike]))
            # Compare to natural activity
            self._naturalActivity = AfferentNatural(self._average_firing_rate, RECORDING_SAMPLING_RATE,
                                                    self._fiber_parameters.n_fibers,
                                                    tot_time_ms=self._get_t_stop())
            self._naturalMeanFr = frt.compute_mean_firing_rate(self._naturalActivity.firings,
                                                               RECORDING_SAMPLING_RATE)
            self._naturalNeuralResponse = frt.compute_neural_response(self._naturalActivity.firings, self._to_samples)
            (_, self._cumulative_firingsNat, self._response_mav_nat, self._response_p2p_nat, self._stim_response_nat,
             self._n_activate_fFiberNat, self._average_spikes_per_activated_fiberNat) = frt.compute_firing_stats(
                self._naturalActivity.firings, 0, self._naturalActivity.tStop, self._stimulation_interval,
                self._to_samples, self._fiber_parameters.propagation_delay_ms)

        if np.sum(self._firings) == 0:
            print('No fiber was recruited by the stimulation.')
        print("finished simulation")

    """
    Specific Methods of this class
    """

    def _create_fibers(self):
        """ Create the fibers and records the APs. """
        self.fibers = []
        self.fibers_position = []
        self._fibers_nc_list = []
        self._action_potentials = []
        self._action_potentials_end = []
        self.fibersId = []
        self._spike_detectors_distance = 20

        # Computing diameters using a truncated normal distribution. Fibers below or above these limits are not
        # supported in the original fiber model of McIntyre 2002.
        diam_generator = get_truncated_normal(mean=self._fiber_parameters.mean_diameter_um,
                                              sd=self._fiber_parameters.std_diameter,
                                              low=MIN_FIBER_DIAMETER,
                                              upp=MAX_FIBER_DIAMETER)
        diameters = diam_generator.rvs(self._fiber_parameters.n_fibers)
        diameters = np.sort(diameters)

        for i in range(self._fiber_parameters.n_fibers):
            # Create the cell
            diameter = diameters[i]
            self.fibers.append(MyelinatedFiberMcIntContinuous(diameter))
            # Position the fiber in space (longitudinal shift, latitudinal shift)
            self.fibers_position.append(
                [self._get_random_shift(self.fibers[-1]), self._get_random_distance()])

            # Ad NetCons to record action potentials
            self._fibers_nc_list.append(self.fibers[-1].connect_to_target(target=None))
            self._action_potentials.append(h.Vector())
            self._fibers_nc_list[-1].record(self._action_potentials[-1])

            self._fibers_nc_list.append(self.fibers[-1].connect_to_target(target=None,
                                                                          node_n=-self._spike_detectors_distance))
            self._action_potentials_end.append(h.Vector())
            self._fibers_nc_list[-1].record(self._action_potentials_end[-1])

    @staticmethod
    def _get_random_shift(fiber):
        return np.random.uniform(0, fiber.node_to_node_distance)

    def _get_random_distance(self):
        # get random distance in a circle of radius equal to the modeled fascicle radius.
        min_r = 0.001  # because of fibrotic tissue no fiber can be really close to the electrode
        distance = min_r + (self._fiber_parameters.fascicle_radius_um - min_r) * np.sqrt(np.random.uniform())
        return distance

    def _find_closest_node_to_electrode(self):
        self._closest_node_to_electrode = []
        for i, fiber in enumerate(self.fibers):
            min_distance = 999999
            min_distance_index = None
            for j, segment in enumerate(fiber.segments):
                if j < fiber.n_nodes and np.abs(
                        self._electrode_offset - segment[1] + self.fibers_position[i][0]) < min_distance:
                    min_distance = np.abs(self._electrode_offset - segment[1] + self.fibers_position[i][0])
                    min_distance_index = j
                if j >= fiber.n_nodes:
                    break
            self._closest_node_to_electrode.append(min_distance_index)

    def _compute_and_set_field(self, amplitude):
        if self._stim_vals:
            self._stim_vals.append([h.t - self._get_integration_step(), self._stim_vals[-1][1]])
        self._stim_vals.append([h.t, amplitude])
        for fiber_index, fiber in enumerate(self.fibers):
            for segment in fiber.segments:
                x_distance = (segment[1] - self.fibers_position[fiber_index][0] - self._electrode_offset) / 1000000.
                y_distance = self.fibers_position[fiber_index][1] / 1000000.
                # Compute the extracellular potential given an anisotropic electrical conductivity as in
                # Neuromodulation 2 (2009) Elliot S Krames, Pag 148.
                segment[0].e_extracellular = amplitude / (4 * np.pi * np.sqrt(
                    DIAG_CONDUCTIVITY[1] * DIAG_CONDUCTIVITY[2] * x_distance ** 2
                    + DIAG_CONDUCTIVITY[0] * DIAG_CONDUCTIVITY[2] * y_distance ** 2
                ))

    def _amplitude_modulation(self, delta_t):
        if self._stim_parameters.bios:
            if delta_t % (1. / (
                    self._stim_parameters.burst_frequency / 1000.)) < self._stim_parameters.pulse_width_ms:
                return (self._stim_parameters.amplitude_ma - self._stim_parameters.min_amplitude_ma
                        ) / self._stim_parameters.burst_duration_ms * delta_t + self._stim_parameters.min_amplitude_ma
            elif delta_t % (1. / (
                    self._stim_parameters.burst_frequency / 1000.)) < 2 * self._stim_parameters.pulse_width_ms:
                return -(self._stim_parameters.amplitude_ma - self._stim_parameters.min_amplitude_ma
                         ) / self._stim_parameters.burst_duration_ms * delta_t - self._stim_parameters.min_amplitude_ma
            else:
                return 0
        else:
            if delta_t > self._stim_parameters.pulse_width_ms / 2:
                return -self._stim_parameters.amplitude_ma
            else:
                return self._stim_parameters.amplitude_ma

    @staticmethod
    def parameters_to_string(stim_parameters: StimParameters, fiber_parameters: FiberParameters) -> str:
        param_string = f"_temp_{TEMPERATURE_C:.1f}deg_fibers_n{int(fiber_parameters.n_fibers)}_length" \
                       f"{fiber_parameters.length_um / 10000:.0f}cm_diameter_" \
                       f"mean{fiber_parameters.mean_diameter_um:.1f}_" \
                       f"std{fiber_parameters.std_diameter:.1f}_" \
                       f"fascicle_r{fiber_parameters.fascicle_radius_um:.0f}_stim_" \
                       f"freq{stim_parameters.frequency:.0f}Hz_" \
                       f"amp{stim_parameters.amplitude_ma:.3f}mA_pw{stim_parameters.pulse_width_ms * 1000:.0f}us"
        if stim_parameters.bios:
            param_string += f"_bios_freq{stim_parameters.burst_frequency:.0f}Hz_" \
                            f"min{stim_parameters.min_amplitude_ma:.3f}mA_" \
                            f"dur{stim_parameters.burst_duration_ms:.0f}ms"
        return param_string

    """
    Plotting
    """

    def plot(self, name="", block=True, window_ms: Optional[List] = None, **kwargs):
        " Plot the simulation results if the stimulation induced at least one action potential. """

        fibers_diameters = [fiber.fiber_d_um for fiber in self.fibers]
        if np.sum(self._firings) != 0:
            ax = []
            size_factor = 0.8
            tick_every = 0.1

            fig = plt.figure(figsize=(18 * size_factor, 7 * size_factor))
            gs = gridspec.GridSpec(13, 15, figure=fig)
            gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.9, wspace=0.25)

            """ Stimulation """
            ax.append(plt.subplot(gs[0, :-1]))
            stim_vals = np.array(self._stim_vals)
            ax[-1].plot(stim_vals[:, 0], stim_vals[:, 1], color='#00ADEE')
            if window_ms is None:
                x_lim = [0, stim_vals[-1, 0]]
            else:
                x_lim = window_ms
            _setup_axes(ax=ax[-1], show_left_spine=True, x_lim=x_lim, y_label='(mA)',
                        x_tick_labels=[], out_left_spine=True)

            """ Membrane potential """
            ax.append(plt.subplot(gs[1:3, :-1]))
            n_fibers_to_plot = 3
            x_values = None
            for i in range(n_fibers_to_plot):
                random_fiber = int(np.random.uniform(0, self._fiber_parameters.n_fibers - 1))
                membrane_pot = np.append(self._membrane_pot_node_fibers[random_fiber], np.nan)
                membrane_pot = np.append(membrane_pot, membrane_pot[-1])
                x_values = list(range(int(np.ceil(self._get_t_stop() / self._get_integration_step() + 1))))
                x_values.append(x_values[-1])
                x_values.append(int(np.ceil(self._get_t_stop() * self.PLOT_OFFSET / self._get_integration_step() + 1)))
                ax[-1].plot(x_values, membrane_pot + i * 60, color='#24aa5d')
            if x_values is not None:
                if window_ms is None:
                    x_lim = [0, x_values[-1]]
                else:
                    x_lim = window_ms
                _setup_axes(ax=ax[-1], x_lim=x_lim, y_label='(mV)', x_tick_labels=[], out_left_spine=True)

            """ Raster plot """
            ax.append(plt.subplot(gs[3:-2, :-1]))
            for i, neuron in enumerate(self._firings):
                ax[-1].vlines(np.where(neuron), i - 0.45, i + 0.45, color='#121212')
            for i, neuron in enumerate(self._naturalActivity.firings):
                ax[-1].vlines(np.where(neuron), i - 0.45, i + 0.45, color='#99e9a8')
            if window_ms is None:
                x_lim = [0, self._get_t_stop() / 1000. * RECORDING_SAMPLING_RATE * self.PLOT_OFFSET]
            else:
                x_lim = np.array(window_ms) / 1000. * RECORDING_SAMPLING_RATE
            _setup_axes(ax=ax[-1], x_tick_labels=[], x_lim=x_lim, y_lim=[-0.5, len(self._firings) - 0.5],
                        y_label='Fibers', out_left_spine=True,
                        y_ticks=np.arange(0, len(self._firings), len(self._firings) * tick_every),
                        y_tick_labels=np.arange(1, len(self._firings) + 1, len(self._firings) * tick_every).astype(int))

            """ fibers characteristics """
            ax.append(plt.subplot(gs[1:3, -1]))
            fiber_section = plt.Circle((0, 0), self._fiber_parameters.fascicle_radius_um, color="#cbc0b8", alpha=0.5)
            ax[-1].add_artist(fiber_section)
            activated_fibers = np.sum(
                self._firings[:, int(RECORDING_SAMPLING_RATE * self.START_STIM_TIME_MS / 1000.):],
                axis=1)
            for i, ((shift, radius), diameter) in enumerate(zip(self.fibers_position, fibers_diameters)):
                rad = np.random.uniform(0, 2 * np.pi)
                if activated_fibers[i]:
                    ax[-1].plot(radius * np.cos(rad), radius * np.sin(rad), "o", markersize=1, color="#3798b1")
                else:
                    ax[-1].plot(radius * np.cos(rad), radius * np.sin(rad), ".", markersize=1, color="#2e2e2e")
            ax[-1].plot(0, 0, "o", markersize=2, color="#cc0000")
            _setup_axes(ax=ax[-1], x_tick_labels=[], out_right_spine=True,
                        x_lim=[-self._fiber_parameters.fascicle_radius_um * 1.2,
                               self._fiber_parameters.fascicle_radius_um * 1.2],
                        y_lim=[-self._fiber_parameters.fascicle_radius_um * 1.2,
                               self._fiber_parameters.fascicle_radius_um * 1.2],
                        y_label='Fibers', show_left_spine=False, show_right_spine=True)
            ax.append(plt.subplot(gs[3:5, -1]))
            bins = np.linspace(MIN_FIBER_DIAMETER, MAX_FIBER_DIAMETER, 16)
            ax[-1].hist(fibers_diameters, bins=bins, color="#4a4a4b")
            _setup_axes(ax=ax[-1], show_left_spine=False, show_right_spine=True, x_lim=[min(bins), max(bins)])

            """ Grand average firing rates """
            temp = np.nonzero(np.sum(self._firings, axis=0))[0]
            ax.append(plt.subplot(gs[-2:, -1]))
            ax[-1].bar(0, np.mean(self._mean_fr[temp[0]:temp[-1]]), 1, color='#121212')
            _setup_axes(ax=ax[-1], x_tick_labels=[],
                        x_label=f'AverageFR {int(self._average_firing_rate)}',
                        show_right_spine=True, show_left_spine=False)

            """ Neural response """
            ax.append(plt.subplot(gs[-2:, :-1]))
            ax[-1].plot(np.arange(self._neural_response.size) * (1000. / RECORDING_SAMPLING_RATE),
                        self._neural_response, color='#121212')
            ax[-1].plot(np.arange(self._naturalNeuralResponse.size) * (1000. / RECORDING_SAMPLING_RATE),
                        self._naturalNeuralResponse, color='#99e9a8')
            x_tick_max = self._get_t_stop() * self.PLOT_OFFSET + 1
            x_tick_space = x_tick_max * tick_every
            if window_ms is None:
                x_lim = [0, len(self._mean_fr) * (1000. / RECORDING_SAMPLING_RATE)]
            else:
                x_lim = window_ms
            _setup_axes(ax=ax[-1],
                        x_lim=x_lim,
                        y_label='Neural response', x_label='Time (ms)',
                        out_bottom_spine=True, show_bottom_spine=True,
                        x_ticks=np.arange(0, x_tick_max, x_tick_space),
                        x_tick_labels=np.arange(0, int(x_tick_max), int(x_tick_space)).astype(int))

            if window_ms is None:
                file_name = time.strftime("%Y_%m_%d_%H_%M_afferent_stim_window_" + self._param_str + name + ".pdf")
            else:
                file_name = time.strftime("%Y_%m_%d_%H_%M_afferent_stim_full_" + self._param_str + name + ".pdf")
            plt.savefig(self._results_folder + file_name, format="pdf", transparent=True)
            plt.show(block=block)

    def plot_stim_response_stats(self, name="", block=True):
        " Plot the simulation results. """
        if np.sum(self._firings) != 0 and self._stats is not None:

            ax = []
            tick_every = 0.1
            size_factor = 0.8
            fig = plt.figure(figsize=(12 * size_factor, 9 * size_factor))
            gs = gridspec.GridSpec(4, 5, figure=fig)
            gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.95, wspace=0.25)

            """ Stimulation """
            ax.append(plt.subplot(gs[0, 0:2]))
            stim_vals = np.array(self._stim_vals)
            start_index = np.argwhere(stim_vals[:, 0] > self.START_STIM_TIME_MS)
            start_index = start_index[0][0]
            stop_index = np.argwhere(stim_vals[:, 0] <= self.START_STIM_TIME_MS +
                                     1000. / self._stim_parameters.frequency)
            stop_index = stop_index[-1][0]
            ax[-1].plot(stim_vals[:, 0], stim_vals[:, 1], color='#00ADEE')
            _setup_axes(ax[-1], out_left_spine=True, x_tick_labels=[],
                        x_lim=[stim_vals[start_index, 0], stim_vals[stop_index + 1, 0] - 1],
                        y_label='(mA)')

            """ 1 stim raster """
            ax.append(plt.subplot(gs[1, 0:2]))
            for i, neuron in enumerate(self._stats["stim_response"]):
                ax[-1].vlines(np.where(neuron), i - 0.45, i + 0.45, color='#121212')
            _setup_axes(ax[-1], x_lim=[0, self._stimulation_interval * self._to_samples],
                        y_lim=[-0.5, len(self._firings) - 0.5], x_tick_labels=[],
                        y_ticks=np.arange(0, len(self._firings), len(self._firings) * tick_every),
                        y_tick_labels=np.arange(1, len(self._firings) + 1, len(self._firings) * tick_every).astype(int),
                        out_left_spine=True, y_label='Fibers')

            ax.append(plt.subplot(gs[1, 2:]))
            for i, neuron in enumerate(self._stim_response_nat):
                ax[-1].vlines(np.where(neuron), i - 0.45, i + 0.45, color='#121212')
            _setup_axes(ax[-1], x_lim=[0, self._stimulation_interval * self._to_samples],
                        y_lim=[-0.5, len(self._firings) - 0.5], x_tick_labels=[],
                        y_ticks=np.arange(0, len(self._firings), len(self._firings) * tick_every),
                        y_tick_labels=np.arange(1, len(self._firings) + 1, len(self._firings) * tick_every).astype(int),
                        show_left_spine=False)

            """ Averaged response """
            ax.append(plt.subplot(gs[2, 0:2]))
            if self._stats["n_stims"] == 1:
                x_vals = np.arange(len(self._stats["cumulative_firings"]))
                ax[-1].plot(x_vals, self._stats["cumulative_firings"], color='#00ADEE')
            else:
                y_vals = np.mean(self._stats["cumulative_firings"], axis=0)
                x_vals = np.arange(len(y_vals))
                ax[-1].plot(x_vals, y_vals, color='#00ADEE')
                ax[-1].fill_between(x_vals, y_vals + np.std(self._stats["cumulative_firings"], axis=0),
                                    y_vals - np.std(self._stats["cumulative_firings"], axis=0),
                                    facecolor='#00ADEE', alpha=0.3)
            _setup_axes(ax[-1], x_lim=[0, self._stimulation_interval * self._to_samples],
                        y_lim=[0, np.max([np.max(self._stats["response_p2p"]), np.max(self._response_p2p_nat)])],
                        x_ticks=[0, self._stimulation_interval * self._to_samples],
                        x_tick_labels=[0, int(self._stimulation_interval)],
                        y_ticks=np.arange(0, len(self._firings), len(self._firings) * tick_every),
                        y_tick_labels=np.arange(1, len(self._firings) + 1, len(self._firings) * tick_every).astype(int),
                        show_left_spine=True, show_bottom_spine=True, out_left_spine=True, out_bottom_spine=True,
                        y_label='Averaged response')

            ax.append(plt.subplot(gs[2, 2:]))
            y_vals = np.mean(self._cumulative_firingsNat, axis=0)
            x_vals = np.arange(len(y_vals))
            ax[-1].plot(x_vals, y_vals, color='#d4900d')
            ax[-1].fill_between(x_vals, y_vals + np.std(self._cumulative_firingsNat, axis=0),
                                y_vals - np.std(self._cumulative_firingsNat, axis=0),
                                facecolor='#d4900d', alpha=0.3)
            _setup_axes(ax[-1], x_lim=[0, self._stimulation_interval * self._to_samples],
                        y_lim=[0, np.max([np.max(self._stats["response_p2p"]), np.max(self._response_p2p_nat)])],
                        x_ticks=[0, self._stimulation_interval * self._to_samples],
                        x_tick_labels=[0, int(self._stimulation_interval)], y_tick_labels=[],
                        show_left_spine=False, show_bottom_spine=True, out_bottom_spine=True,
                        y_label='Averaged response')

            """ Responses stats """
            # MAV
            ax.append(plt.subplot(gs[3, 0]))
            if self._stats["n_stims"] == 1:
                ax[-1].bar(1, self._stats["response_mav"], 0.5, color='#00ADEE')
            else:
                ax[-1].bar(1, np.mean(self._stats["response_mav"]), 0.5, yerr=np.std(self._stats["response_mav"]),
                           color='#00ADEE')
            ax[-1].bar(2, np.mean(self._response_mav_nat), 0.5, yerr=np.std(self._response_mav_nat),
                       color='#d4900d')
            if self.FIXED_YLIMS:
                y_lim = y_tick_labels = y_ticks = [0, self.MAX_Y_MAV]
            else:
                y_lim = y_ticks = [0, np.max([np.max(self._stats["response_mav"]), np.max(self._response_mav_nat)])]
                y_tick_labels = [0, round(np.max([np.max(self._stats["response_mav"]),
                                                  np.max(self._response_mav_nat)]), 2)]
            _setup_axes(ax[-1], x_lim=[0, 3], x_tick_labels=[], out_left_spine=True,
                        y_lim=y_lim, y_ticks=y_ticks, y_tick_labels=y_tick_labels, y_label='MAV')

            # P2P
            ax.append(plt.subplot(gs[3, 1]))
            if self._stats["n_stims"] == 1:
                ax[-1].bar(1, self._stats["response_p2p"], 0.5, color='#00ADEE')
            else:
                ax[-1].bar(1, np.mean(self._stats["response_p2p"]), 0.5, yerr=np.std(self._stats["response_p2p"]),
                           color='#00ADEE')
            ax[-1].bar(2, np.mean(self._response_p2p_nat), 0.5, yerr=np.std(self._response_p2p_nat), color='#d4900d')
            if self.FIXED_YLIMS:
                y_lim = y_tick_labels = y_ticks = [0, self.MAX_Y_P2P]
            else:
                y_lim = y_ticks = [0, np.max([np.max(self._stats["response_p2p"]), np.max(self._response_p2p_nat)])]
                y_tick_labels = [0, round(np.max([np.max(self._stats["response_p2p"]),
                                                  np.max(self._response_p2p_nat)]), 2)]
            _setup_axes(ax[-1], x_lim=[0, 3], x_tick_labels=[], out_left_spine=True,
                        y_lim=y_lim, y_ticks=y_ticks, y_tick_labels=y_tick_labels, y_label='P2P')

            # NACTIVE
            ax.append(plt.subplot(gs[3, 2]))
            if self._stats["n_stims"] == 1:
                ax[-1].bar(1, self._stats["n_activated_fiber"], 0.5, color='#00ADEE')
            else:
                ax[-1].bar(1, np.mean(self._stats["n_activated_fiber"]), 0.5,
                           yerr=np.std(self._stats["n_activated_fiber"]),
                           color='#00ADEE')
            ax[-1].bar(2, np.mean(self._n_activate_fFiberNat), 0.5, yerr=np.std(self._n_activate_fFiberNat),
                       color='#d4900d')
            if self.FIXED_YLIMS:
                y_lim = y_tick_labels = y_ticks = [0, self.MAX_Y_REC]
            else:
                y_lim = y_ticks = [0, np.max([np.max(self._stats["n_activated_fiber"]),
                                              np.max(self._n_activate_fFiberNat)])]
                y_tick_labels = [0, int(np.max([np.max(self._stats["n_activated_fiber"]),
                                                np.max(self._n_activate_fFiberNat)]))]
            _setup_axes(ax[-1], x_lim=[0, 3], x_tick_labels=[], out_left_spine=True,
                        y_lim=y_lim, y_ticks=y_ticks, y_tick_labels=y_tick_labels, y_label='N activated fibers')

            # SPIKES4ACT
            ax.append(plt.subplot(gs[3, 3]))
            if self._stats["n_stims"] == 1:
                ax[-1].bar(1, self._stats["average_spikes_per_activated_fiber"], 0.5, color='#00ADEE')
            else:
                ax[-1].bar(1, np.mean(self._stats["average_spikes_per_activated_fiber"]), 0.5,
                           yerr=np.std(self._stats["average_spikes_per_activated_fiber"]), color='#00ADEE')
            ax[-1].bar(2, np.mean(self._average_spikes_per_activated_fiberNat), 0.5,
                       yerr=np.std(self._average_spikes_per_activated_fiberNat), color='#d4900d')
            if self.FIXED_YLIMS:
                y_lim = y_tick_labels = y_ticks = [0, self.MAX_Y_NAP]
            else:
                y_lim = y_ticks = [0, np.max([np.max(self._stats["average_spikes_per_activated_fiber"]),
                                              np.max(self._average_spikes_per_activated_fiberNat)])]
                y_tick_labels = [0, np.max([np.max(self._stats["average_spikes_per_activated_fiber"]),
                                            np.max(self._average_spikes_per_activated_fiberNat)])]
            _setup_axes(ax[-1], x_lim=[0, 3], x_tick_labels=[], out_left_spine=True,
                        y_lim=y_lim, y_ticks=y_ticks, y_tick_labels=y_tick_labels, y_label='spikes x activ. fiber')

            # STD
            ax.append(plt.subplot(gs[3, 4]))
            if self._stats["n_stims"] > 1:
                ax[-1].bar(1, np.mean(np.std(self._stats["cumulative_firings"], axis=0)), 0.5, color='#00ADEE')
                ax[-1].bar(2, np.mean(np.std(self._cumulative_firingsNat, axis=0)), 0.5, color='#d4900d')

                if self.FIXED_YLIMS:
                    y_lim = y_tick_labels = y_ticks = [0, self.MAX_Y_STD]
                else:
                    y_lim = y_ticks = y_tick_labels = [0, np.max(np.std(self._stats["cumulative_firings"], axis=0))]
                _setup_axes(ax[-1], x_lim=[0, 3], x_tick_labels=[], out_left_spine=True,
                            y_lim=y_lim, y_ticks=y_ticks, y_tick_labels=y_tick_labels,
                            y_label='STD')

            file_name = time.strftime("%Y_%m_%d_responseStats_" + self._param_str + name + ".pdf")
            plt.savefig(self._results_folder + file_name, format="pdf", transparent=True)
            plt.show(block=block)


def _setup_axes(ax,
                show_bottom_spine: Optional[bool] = False,
                out_bottom_spine: Optional[bool] = False,
                show_right_spine: Optional[bool] = False,
                out_right_spine: Optional[bool] = False,
                show_left_spine: Optional[bool] = True,
                out_left_spine: Optional[bool] = False,
                show_top_spine: Optional[bool] = False,
                out_top_spine: Optional[bool] = False,
                y_label: Optional[str] = None,
                x_label: Optional[str] = None,
                y_lim: Optional[List] = None,
                x_lim: Optional[List] = None,
                y_ticks: Optional[Union[List, np.ndarray]] = None,
                x_ticks: Optional[Union[List, np.ndarray]] = None,
                y_tick_labels: Optional[Union[List, np.ndarray]] = None,
                x_tick_labels: Optional[Union[List, np.ndarray]] = None):
    ax.spines['bottom'].set_visible(show_bottom_spine)
    ax.spines['right'].set_visible(show_right_spine)
    ax.spines['left'].set_visible(show_left_spine)
    ax.spines['top'].set_visible(show_top_spine)

    if show_left_spine:
        ax.yaxis.set_ticks_position('left')
    elif show_right_spine:
        ax.yaxis.set_ticks_position('right')

    if out_bottom_spine:
        ax.spines['bottom'].set_position(('outward', 5))
    if out_right_spine:
        ax.spines['right'].set_position(('outward', 5))
    if out_left_spine:
        ax.spines['left'].set_position(('outward', 5))
    if out_top_spine:
        ax.spines['top'].set_position(('outward', 5))

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
        if len(x_tick_labels) == 0:
            ax.xaxis.set_ticks_position('none')
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_tick_labels is not None:
        ax.set_yticklabels(y_tick_labels)
        if len(y_tick_labels) == 0:
            ax.yaxis.set_ticks_position('none')
