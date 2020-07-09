import argparse
import os
from simulations.afferent_stimulation import AfferentStimulation
from simulations.constants import RESULT_FOLDER
from simulations.parameters import StimParameters, FiberParameters, FIBERS_LENGTH_UM, PROPAGATION_DELAY_MS, \
    FASCICLE_RADIUS_UM, MEAN_DIAMETER_UM, STD_DIAMETER
from tools.general_tools import SubprocessRunner
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from tools import general_tools as gt
from typing import Tuple

FIX_CLIMS = True
MAX_CNSPIKES = 7
MAX_CP2P = 16


def parameter_sweep(bios: bool, tonic_burst: bool, sim_name: str = "", n_processes: int = 6,
                    blocking_plot: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                         Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Runs several AfferentStimulation simulations with a range of different stimulation parameters.
    Useful for sensitivity analyses (e.g., Figure 3 of the manuscript).
    Multiple processes are used to run simulations with different parameters.


    Args:
        bios: Flag to simulate BioS bursts
        tonic_burst: Flag to simulate constant-amplitude high-frequency bursts
        sim_name: Sim name to append at the results file name
        n_processes: total number of processes to run simultaneously.
        blocking_plot: Flag to either use blocking or not blocking plots.

    Returns: A nested Tuple of numpy arrays containing: (stimulation amplitudes, burst frequencies),
     (number of activated fibers, n of spikes elicited in the activated fibers, simulated neural response peak to peak
     amplitude).

    """

    if tonic_burst:
        print("Running parameter sweep for tonic burst")
        results_folder = os.path.join(RESULT_FOLDER, "tonic_burst", "")
        bios_type_stim = True
    elif bios:
        print("Running parameter sweep for BioS")
        results_folder = os.path.join(RESULT_FOLDER, "bios", "")
        bios_type_stim = True
    else:
        print("Running parameter sweep for single pulse stimulation")
        results_folder = os.path.join(RESULT_FOLDER, "single_pulse", "")
        bios_type_stim = False

    """ Fixed tested parameters """
    min_stim_amp = -0.01
    stim_amplitudes = np.round(np.arange(min_stim_amp, -0.06, -0.005), 3)
    burst_frequencies = np.round(np.arange(1000., 8500., 500.), 0)
    if not bios_type_stim:
        burst_frequencies = np.array([0.])
    n_sims = stim_amplitudes.size * burst_frequencies.size
    n_fibers = 50
    stim_freq = 40
    sim_time = 200
    sim_name = f"_batch{sim_name}"
    pulse_width = 50.
    burst_duration = 19.

    program_commons = ['python3',
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_afferent_stimulation.py"),
                       "--n-fibers", str(n_fibers),
                       "--stim-freq", str(stim_freq),
                       "--sim-time", str(sim_time),
                       "--sim-name", str(sim_name),
                       "--pulse-width", str(pulse_width),
                       "--burst-duration", str(burst_duration),
                       "--results-folder", results_folder,
                       "--plot-response-stats",
                       "--non-blocking-plots"]
    if bios_type_stim:
        program_commons.append("--bios")

    count = 0.
    perc_last_print = 0.
    print_period = 0.05
    process_id = 0
    processes = []
    for stim_amp in stim_amplitudes:
        for burst_frequency in burst_frequencies:
            result_file_name = get_sim_name(stim_freq, pulse_width, stim_amp, bios_type_stim, tonic_burst, burst_frequency,
                                            burst_duration, min_stim_amp, n_fibers, sim_name)
            result_file = gt.find("*" + result_file_name + ".p", results_folder)
            if not result_file:
                print(f"\tFile not found: {result_file_name}")
                program_extras = ["--stim-amp", str(stim_amp), "--burst-frequency", str(burst_frequency)]
                if tonic_burst:
                    program_extras.extend(["--min-stim-amp", str(stim_amp)])
                else:
                    program_extras.extend(["--min-stim-amp", str(min_stim_amp)])
                processes.append(SubprocessRunner(program_commons + program_extras, process_id))
                processes[-1].run()
                process_id += 1
                process_id %= n_processes
            while len(processes) >= n_processes:
                time.sleep(1)
                processes[:] = [x for x in processes if not x.is_finished]
            count += 1
            if count / n_sims - perc_last_print >= print_period:
                perc_last_print = count / n_sims
                print(str(round(count / n_sims * 100)) + "% of simulations performed...")
    for process in processes:
        process.wait()

    stats = []
    n_activated_fiber = np.zeros([stim_amplitudes.size, burst_frequencies.size])
    n_spikes4_active = np.zeros([stim_amplitudes.size, burst_frequencies.size])
    response_p2p = np.zeros([stim_amplitudes.size, burst_frequencies.size])

    for i, stim_amp in enumerate(stim_amplitudes):
        stats.append([])
        for j, burst_frequency in enumerate(burst_frequencies):
            result_file_name = get_sim_name(stim_freq, pulse_width, stim_amp, bios_type_stim, tonic_burst, burst_frequency,
                                            burst_duration, min_stim_amp, n_fibers, sim_name)
            result_file = gt.find("*" + result_file_name + ".p", results_folder)
            file_idx = 0
            if len(result_file) > 1:
                dates = [f.split('/')[-1][:16] for f in result_file]
                dates = [int("".join([c for c in date if c != "_"])) for date in dates]
                file_idx = np.argmax(dates)
                print(f"Warning: multiple result files found.\nSelecting latest one: {dates[file_idx]} over {dates}")

            with open(result_file[file_idx], 'rb') as pickle_file:
                _firings = pickle.load(pickle_file)
                _stats = pickle.load(pickle_file)
            stats[-1].append(_stats)
            if _stats is not None:
                n_activated_fiber[i, j] = np.mean(_stats["n_activated_fiber"])
                n_spikes4_active[i, j] = np.mean(_stats["average_spikes_per_activated_fiber"])
                response_p2p[i, j] = np.mean(_stats["response_p2p"])

    fig, ax = plt.subplots(3, figsize=(6, 9))
    vmax = MAX_CNSPIKES if FIX_CLIMS else np.ceil(np.max(n_spikes4_active))
    im = ax[0].imshow(n_spikes4_active, cmap=plt.cm.bone_r, interpolation='nearest', origin="lower", vmin=1, vmax=vmax)
    set_axes(ax[0], "", burst_frequencies, stim_amplitudes)
    fig.colorbar(im, ax=ax[0], orientation='vertical', label="# spikes for active fiber", extend='max')

    im2 = ax[1].imshow(n_activated_fiber, cmap=plt.cm.bone_r, interpolation='nearest', origin="lower", vmin=0,
                       vmax=n_fibers)
    set_axes(ax[1], "", burst_frequencies, stim_amplitudes)
    fig.colorbar(im2, ax=ax[1], orientation='vertical', label="# activated fiber")

    vmax = MAX_CP2P if FIX_CLIMS else np.ceil(np.max(response_p2p))
    im3 = ax[2].imshow(response_p2p, cmap=plt.cm.bone_r, interpolation='nearest', origin="lower", vmin=0, vmax=vmax)
    set_axes(ax[2], "", burst_frequencies, stim_amplitudes)
    fig.colorbar(im3, ax=ax[2], orientation='vertical', label="P2P amplitude")
    ax[2].set_xlabel('Burst frequencies (Hz)')

    result_file_name = get_sim_name(stim_freq, pulse_width, 0, bios_type_stim, tonic_burst,
                                     0, 0, min_stim_amp, n_fibers, sim_name)
    file_name = time.strftime("/%Y_%m_%d_%H_%M_SS_results_" + result_file_name + ".pdf")
    plt.savefig(results_folder + file_name, format="pdf", transparent=True)
    plt.show(block=blocking_plot)

    return (stim_amplitudes, burst_frequencies), (n_activated_fiber, n_spikes4_active, response_p2p)


def set_axes(ax, title, burst_frequencies, stim_amplitudes):
    ax.set_title(title)
    ax.set_xticks(range(burst_frequencies.size))
    ax.set_xticklabels(burst_frequencies)
    ax.set_yticks(range(stim_amplitudes.size))
    ax.set_yticklabels(stim_amplitudes)
    ax.set_ylabel('Stim amplitudes (mA)')


def get_sim_name(stim_freq, pulse_width, stim_amp, bios, tonic_burst, burst_frequency, burst_duration,
                 min_stim_amp, n_fibers, sim_name) -> str:
    if tonic_burst:
        _min_stim_amp = stim_amp
    else:
        _min_stim_amp = min_stim_amp

    stim_parameters = StimParameters(
        frequency=stim_freq,
        pulse_width_ms=pulse_width / 1000.,
        amplitude_ma=stim_amp,
        bios=bios,
        burst_frequency=burst_frequency,
        burst_duration_ms=burst_duration,
        min_amplitude_ma=_min_stim_amp,
    )
    fiber_parameters = FiberParameters(
        n_fibers=n_fibers,
        length_um=FIBERS_LENGTH_UM,
        mean_diameter_um=MEAN_DIAMETER_UM,
        std_diameter=STD_DIAMETER,
        propagation_delay_ms=PROPAGATION_DELAY_MS,
        fascicle_radius_um=FASCICLE_RADIUS_UM
    )
    result_file_name = AfferentStimulation.parameters_to_string(stim_parameters=stim_parameters,
                                                                fiber_parameters=fiber_parameters)
    return result_file_name + sim_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a batch of AfferentStimulation simulations")
    parser.add_argument("-b", "--bios", help="flag to use bios", action="store_true")
    parser.add_argument("-t", "--tonic-burst", help="flag stimulate w/ constant amplitude bursts", action="store_true")
    parser.add_argument("--sim-name", help="String to append at the end of the result files", type=str, default="")
    parser.add_argument("--n-processes", help="Number of processes to spawn", type=int, default=6)
    args = parser.parse_args()
    _ = parameter_sweep(bios=args.bios,
                        tonic_burst=args.tonic_burst,
                        sim_name=args.sim_name,
                        n_processes=args.n_processes,
                        blocking_plot=True)
