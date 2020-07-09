import time
import matplotlib.pyplot as plt
from run_parameters_sweep import parameter_sweep
from simulations.constants import RESULT_FOLDER
from tools.general_tools import MidpointNormalize
import numpy as np


def compute_ratio(burst_p2p: np.ndarray,
                  burst_n_act_fib: np.ndarray,
                  single_n_act_fib: np.ndarray,
                  single_p2p: np.ndarray):
    burst_vs_sing = np.zeros(burst_p2p.shape)
    for i in range(burst_n_act_fib.shape[0]):
        for j in range(burst_n_act_fib.shape[1]):
            id_single = np.argmin(abs(single_n_act_fib-burst_n_act_fib[i, j]))
            print(f" min is {min(abs(single_n_act_fib-burst_n_act_fib[i, j]))}")
            burst_vs_sing[i, j] = burst_p2p[i, j] / single_p2p[id_single]
    return burst_vs_sing


def stim_compare():
    """ Compare different stimulation paradigms for a given range of stimulation parameters. The 'parameter_sweep'
    function defined in run_parameters_sweep.py is used to run the simulation results (or load their results if results
    files are found).
    """

    _, (bios_n_act_fib, _, bios_p2p) = parameter_sweep(bios=True,
                                                       tonic_burst=False,
                                                       blocking_plot=False)
    (amplitudes, b_frequencies), (tonic_n_act_fib, _, tonic_p2p) = parameter_sweep(bios=False,
                                                                                   tonic_burst=True,
                                                                                   blocking_plot=False)
    _, (single_n_act_fib, _, single_p2p) = parameter_sweep(bios=False,
                                                           tonic_burst=False,
                                                           blocking_plot=False)

    bios_vs_sing = compute_ratio(bios_p2p, bios_n_act_fib, single_n_act_fib, single_p2p)
    tonic_vs_sing = compute_ratio(tonic_p2p, tonic_n_act_fib, single_n_act_fib, single_p2p)
    bios_vs_tonic = bios_p2p/tonic_p2p
    tonic_vs_bios = tonic_p2p/bios_p2p

    fig, ax = plt.subplots(4, figsize=(6, 9))
    v_min, v_max = 0.5, 1.5
    for i, (stat, stat_name) in enumerate(zip([bios_vs_sing, tonic_vs_sing, bios_vs_tonic, tonic_vs_bios],
                                              ["P2P BioS vs Single", "P2P Tonic vs Single", "P2P Bios vs Tonic",
                                               "P2P Tonic vs Bios"])):
        im = ax[i].imshow(stat, cmap=plt.cm.RdBu_r, interpolation='nearest', origin="lower", vmin=v_min,
                          vmax=v_max, norm=MidpointNormalize(midpoint=1, vmin=v_min, vmax=v_max))
        set_axes(ax[i], "", b_frequencies, amplitudes)
        fig.colorbar(im, ax=ax[i], orientation='vertical', label=stat_name, extend='both')
    ax[-1].set_xlabel('Burst frequencies (Hz)')
    file_name = time.strftime("/%Y_%m_%d_%H_%M_SS_results_stim_comparison.pdf")
    plt.savefig(RESULT_FOLDER + file_name, format="pdf", transparent=True)
    plt.show()


def set_axes(ax, title, burst_frequencies, stim_amplitudes):
    ax.set_title(title)
    ax.set_xticks(range(burst_frequencies.size))
    ax.set_xticklabels(burst_frequencies)
    ax.set_yticks(range(stim_amplitudes.size))
    ax.set_yticklabels(stim_amplitudes)
    ax.set_ylabel('Stim amplitudes (mA)')


if __name__ == '__main__':
    stim_compare()
