import numpy as np
from scipy import signal as ss


def extract_action_potentials(ap_list_vector, max_time) -> np.ndarray:
    n_ap = [ap_vector.size() for ap_vector in ap_list_vector]
    n_cells = len(ap_list_vector)
    if not n_ap:
        max_nap = 0
    else:
        max_nap = max(n_ap)
    # Extracting the matrix with the ap time (nFibers x max_nap)
    action_potentials = None
    if n_ap:
        action_potentials = np.nan * np.ones([int(n_cells), int(max_nap)])
        for i, ap_vector in enumerate(ap_list_vector):
            for ap in range(int(ap_vector.size())):
                action_potentials[i, ap] = ap_vector.x[ap]
    return action_potentials


def adjust_propagation_time(action_potentials1, action_potentials2, distance_between_spikes, actual_length,
                            desired_length, spike_start_position):
    min_size = min(action_potentials1.shape[1], action_potentials2.shape[
        1])  # there could be an action potential that did not have the time to reach the end of the fiber...
    action_potentials1 = action_potentials1[:, :min_size - 1]
    action_potentials2 = action_potentials2[:, :min_size - 1]
    time_difference = action_potentials1 - action_potentials2
    ap_propagation_velocity = distance_between_spikes[:, np.newaxis] / time_difference
    traveled_distance = actual_length - spike_start_position
    dt = traveled_distance[:, np.newaxis] / ap_propagation_velocity
    time_to_reach_desired_fiber_end = desired_length[:, np.newaxis] / ap_propagation_velocity
    return action_potentials1 - dt + time_to_reach_desired_fiber_end


def extract_firings(action_potentials: np.ndarray, max_time: float, sampling_rate: float = 10000.) -> np.ndarray:
    # extracting the firings binary matrix
    n_cells = action_potentials.shape[0]
    dt = 1000. / sampling_rate
    firings = np.zeros([n_cells, 1 + int(max_time / dt)])
    if action_potentials is not None:
        action_potentials = (action_potentials / dt).astype(int)
        for i in range(n_cells):
            indx = action_potentials[i, :] >= 0
            firings[i, action_potentials[i, indx]] = 1
    return firings


def compute_mean_firing_rate(firings: np.ndarray, sampling_rate: float = 10000.):
    """ Return the mean firing rates given the cell firings.

    Keyword arguments:
        firings -- Cell firings, a 2d numpy array (n_cells x time).
        sampling_rate -- Sampling rate of the extracted signal in Hz (default = 1000).
    """

    mean_fr = None
    interval = 40 * sampling_rate / 1000  # ms
    n_cells = firings.shape[0]
    n_samples = firings.shape[1]

    mean_fr_temp = np.zeros(n_samples)
    mean_fr = np.zeros(n_samples)
    for i in range(int(interval), n_samples):
        tot_ap = firings[:, i - int(interval):i].sum()
        mean_fr_temp[i - int(round(interval / 2))] = tot_ap / n_cells * sampling_rate / interval

    # Smooth the data with a moving average
    window_size = int(50 * sampling_rate / 1000)  # ms
    for i in range(window_size, n_samples):
        mean_fr[i - int(round(window_size / 2))] = mean_fr_temp[i - window_size:i].mean()
    return mean_fr


def neural_response(to_samples, response_std_ms: float = 0.33, response_duration_ms: float = 1.):
    response = ss.gaussian(int(response_duration_ms * to_samples), response_std_ms * to_samples)
    return response


def compute_neural_response(firings: np.ndarray, to_samples):
    single_response = neural_response(to_samples)
    response = np.convolve(np.sum(firings, axis=0), single_response, "same")
    return response


def compute_firing_stats(firings: np.ndarray, start_stim_time, t_stop,
                         stimulation_interval, to_samples, shift=0, debug=False,
                         response_std_ms: float = 0.33, response_duration_ms: float = 1.):
    n_stims = np.arange(start_stim_time, t_stop - stimulation_interval, stimulation_interval).size
    if n_stims > 2:
        n_stims -= 1  # to remove init effects
    if debug:
        print("\nN analyzed stimulations pulses:", str(n_stims))

    cumulative_firings = np.zeros([n_stims, int(stimulation_interval * to_samples)])
    n_activated_fiber = []
    average_spikes_per_activated_fiber = []
    single_response = neural_response(to_samples, response_std_ms=response_std_ms,
                                      response_duration_ms=response_duration_ms)
    stim_response = []
    for i, time in enumerate(np.arange(start_stim_time, t_stop - stimulation_interval - shift, stimulation_interval)):
        if n_stims > 1 and i == 0:
            continue  # skip first to remove init effects
        start_index = int((time + shift) * to_samples)
        end_index = int((time + stimulation_interval + shift) * to_samples + 1)
        stim_response = firings[:, start_index:end_index]
        n_activated_fiber.append(np.where(np.sum(stim_response, axis=1) > 0)[0].size)
        if n_activated_fiber[-1] > 0:
            average_spikes_per_activated_fiber.append(np.sum(stim_response) / n_activated_fiber[-1])
        else:
            average_spikes_per_activated_fiber.append(0)
        response = np.convolve(np.sum(stim_response, axis=0), single_response, "same")
        if n_stims == 1:
            cumulative_firings = response
        else:
            cumulative_firings[i - 1, :] = response[:int(stimulation_interval * to_samples)]
    if n_stims == 1:
        response_mav = np.mean(cumulative_firings)
        response_p2p = np.max(cumulative_firings) - np.min(cumulative_firings)
    else:
        response_mav = np.mean(cumulative_firings, axis=1)
        response_p2p = np.max(cumulative_firings, axis=1) - np.min(cumulative_firings, axis=1)

    return n_stims, cumulative_firings, response_mav, response_p2p, stim_response, n_activated_fiber, \
           average_spikes_per_activated_fiber


def real_neural_response(to_samples):
    len_response = 2  # ms
    len_response_samples = int(len_response * to_samples)
    response = np.sin(np.linspace(0, 2 * np.pi, len_response_samples))
    return response


def compute_real_neural_response(firings, to_samples):
    len_response = 2  # ms
    len_response_samples = int(len_response * to_samples)
    single_response = np.sin(np.linspace(0, 2 * np.pi, len_response_samples))
    response = np.convolve(np.sum(firings, axis=0), single_response, "same")
    return response
