import os
import fnmatch
import subprocess
from threading import Thread
from scipy.stats import truncnorm
import matplotlib.colors as colors
import numpy as np


def get_truncated_normal(mean=0., sd=1., low=0., upp=10.):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def find(pattern, path):
    " Finds the files in a path with a given pattern. "
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def run_subprocess(program):
    """ Runs a given program as a subrocess. """
    print("\tRunning subprocess: %s" % (" ".join(program)))
    return_code = None
    while not return_code == 0:
        p = subprocess.Popen(program, stdout=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True, bufsize=1,
                             close_fds=True)
        for stdout_line in iter(p.stdout.readline, ""):
            print(stdout_line, end='')
        p.stdout.close()
        return_code = p.wait()
        if return_code != 0: print("\t\t\t\t Error n: ", return_code, " resetting simulation...")


class SubprocessRunner(object):

    def __init__(self, program, id: int):
        self._program = program
        self._return_code = None
        self._p = None
        self.thread = None
        self.id = id
        self.return_code = None
        self.is_stopped = False

    def run(self):
        print("\tRunning subprocess: %s" % (" ".join(self._program)))
        self.thread = Thread(target=self._run)
        self.thread.start()

    def _run(self):
        while not self.return_code == 0 and not self.is_stopped:
            p = subprocess.Popen(self._program, stdout=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True)
            for stdout_line in iter(p.stdout.readline, ""):
                print(f'process {self.id}:\t', stdout_line, end='')
            p.stdout.close()
            self.return_code = p.wait()
            if self.return_code != 0:
                print(f"\t\t\t\t Process {self.id}:\tError n: ", self.return_code, " resetting simulation...")

    @property
    def is_finished(self):
        if self.return_code == 0:
            return True
        else:
            return False

    def wait(self):
        if self.thread is not None:
            self.thread.join()

    def stop(self):
        self.is_stopped = True
        if self.thread is not None:
            self.thread.join()


class MidpointNormalize(colors.Normalize):
    """
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	from: http://chris35wills.github.io/matplotlib_diverging_colorbar/
	"""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
