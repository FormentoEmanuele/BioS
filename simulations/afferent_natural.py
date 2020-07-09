import os

from tools import firings_tools as frt
import time
import numpy as np
import matplotlib.pyplot as plt


class AfferentNatural(object):

    def __init__(self, frequency: float, recording_sampling_rate: float, n_fibers: int = 100, tot_time_ms: float = 500,
                 debug: bool = False):
        """
        Simulation of poisson-like neural activity - used to simulate asynchronous natural neural activity.
        """
        self._results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
        self._frequency = frequency
        self._recording_sampling_rate = recording_sampling_rate
        self.interval = 1000. / self._frequency  # in ms

        self._nRepetitions = int(np.ceil(tot_time_ms / self.interval))
        self._n_fibers = n_fibers
        self.tStop = self.interval * self._nRepetitions
        margin = 5

        self._actionPotentialTimes = np.nan * np.zeros([self._n_fibers, self._nRepetitions + margin])
        for i in range(0, self._n_fibers):
            time = 0
            count = 0
            while time < self.tStop:
                time += np.random.exponential(self.interval)

                if time < self.tStop and count < self._nRepetitions + margin:
                    self._actionPotentialTimes[i, count] = time
                    count += 1
        self.firings = frt.extract_firings(self._actionPotentialTimes, self.tStop, self._recording_sampling_rate)
        self.mean_fr = frt.compute_mean_firing_rate(self.firings, self._recording_sampling_rate)
        if debug: print("Average firing rate: ", np.sum(self.firings) / 100. / (self.tStop / 1000.), " Imp/s")

    def plot_firings(self, block=True):
        """ Raster plot """
        fig, ax = plt.subplots(2)

        for i, neuron in enumerate(self.firings):
            ax[0].vlines(np.where(neuron), i - 0.45, i + 0.45, color='#121212')
        ax[0].set_xlim([0, self.tStop / 1000. * self._recording_sampling_rate])
        ax[0].set_ylim([-0.5, len(self.firings) - 0.5])
        ax[0].set_xticklabels([])

        tick_every = 0.1
        ax[0].set_yticks(np.arange(0, len(self.firings), len(self.firings) * tick_every))
        ax[0].set_yticklabels(np.arange(1, len(self.firings) + 1, len(self.firings) * tick_every).astype(int))
        # Move left and bottom spines outward by 5 points
        ax[0].spines['left'].set_position(('outward', 5))
        # Hide the right, bottom and top spines
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax[0].yaxis.set_ticks_position('left')
        ax[0].xaxis.set_ticks_position('none')

        ax[0].set_ylabel('Fibers')

        """ Grand average firing rates """
        ax[1].plot(np.arange(self.mean_fr.size) * (1000. / self._recording_sampling_rate), self.mean_fr,
                   color='#121212')
        tick_every = 0.1
        x_tick_max = self.tStop + 1
        x_tick_space = x_tick_max * tick_every
        ax[1].set_xticks(np.arange(0, x_tick_max, x_tick_space))
        ax[1].set_xticklabels(np.arange(0, int(self.tStop) + 1, int(self.tStop) * tick_every).astype(int))

        # Move left and bottom spines outward by 5 points
        ax[1].spines['bottom'].set_position(('outward', 5))
        # Hide the right and top spines
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax[1].yaxis.set_ticks_position('left')
        ax[1].xaxis.set_ticks_position('bottom')

        ax[1].set_xlim([0, len(self.mean_fr) * (1000. / self._recording_sampling_rate)])
        ax[1].set_ylabel('Mean firing rate')
        ax[1].set_xlabel('Time (ms)')

        file_name = time.strftime("%Y_%m_%d_natural_" + str(self._frequency) + "Hz.pdf")
        file_path = os.path.join(self._results_folder, file_name)
        plt.savefig(file_path, format="pdf", transparent=True)
        plt.show(block=block)


if __name__ == '__main__':
    ap = AfferentNatural(frequency=50, recording_sampling_rate=5000)
    ap.plot_firings()
