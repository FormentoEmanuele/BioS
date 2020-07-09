import numpy as np
import os

TEMPERATURE_C = 37  # To mimic body temperature
# TEMPERATURE_C = 23  # To mimic experimental conditions
# Conductivity tensor used to simulate white matter properties - from Capogrosso et al 2013.
CONDUCTIVITY = np.array([[0.6, 0, 0], [0, 0.083, 0], [0, 0, 0.083]])
RESISTIVITY = np.linalg.inv(CONDUCTIVITY)
DIAG_CONDUCTIVITY = np.array([CONDUCTIVITY[0, 0], CONDUCTIVITY[1, 1], CONDUCTIVITY[2, 2]])

MAX_FIBER_DIAMETER = 16
MIN_FIBER_DIAMETER = 6
RECORDING_SAMPLING_RATE = 5000

RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "")
