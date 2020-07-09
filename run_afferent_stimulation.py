import argparse
from simulations.afferent_stimulation import AfferentStimulation
from simulations.parameters import StimParameters, FiberParameters, FIBERS_LENGTH_UM, PROPAGATION_DELAY_MS, \
    FASCICLE_RADIUS_UM, MEAN_DIAMETER_UM, STD_DIAMETER


def main():
    """ Main script running an AfferentStimulation simulation with cli defined input parameters.
    The results from this simulation are saved in the results folder (see Simulation._results_folder).
    """
    parser = argparse.ArgumentParser(description="Run a AfferentStimulation simulation.")
    parser.add_argument("-n", "--n-fibers", help="number of fibers", type=int, default=100)
    parser.add_argument("-a", "--stim-amp", help="Simulation amplitude (mA)", type=float, default=-0.045)
    parser.add_argument("--min-stim-amp", help="Simulation min amplitude (mA)", type=float, default=-0.01)
    parser.add_argument("-f", "--stim-freq", help="Stimulation frequency (Hz)", type=int, default=40)
    parser.add_argument("-p", "--pulse-width", help="Stimulation pulse width (us)", type=float, default=50.)
    parser.add_argument("-b", "--bios", help="flag to use bios burst stimulation", action="store_true")
    parser.add_argument("--burst-frequency", help="Stimulation frequency within a bios burst (Hz)", type=float,
                        default=8000.)
    parser.add_argument("-d", "--burst-duration", help="Bios burst duration (ms)", type=float, default=20.)
    parser.add_argument("-t", "--sim-time", help="Simulation time (ms)", type=int, default=500)
    parser.add_argument("--sim-name", help="String to append at the end of the result files", type=str, default="")
    parser.add_argument("--plot-response-stats", help="Flag to plot the stimulation response statistics",
                        action="store_true")
    parser.add_argument("-w", "--plot-window", help="Flag to plot a specific window of data", action="store_true")
    parser.add_argument("--plot-window-duration", help="Duration in ms of the window to plot", type=float, default=150.)
    parser.add_argument("--non-blocking-plots", help="Flag to use non-blocking plots", action="store_true")
    parser.add_argument("--results-folder", help="Path to folder where the results are saved", type=str, default=None)
    args = parser.parse_args()

    stim_parameters = StimParameters(
        frequency=args.stim_freq,
        pulse_width_ms=args.pulse_width / 1000.,
        amplitude_ma=args.stim_amp,
        bios=args.bios,
        burst_frequency=args.burst_frequency,
        burst_duration_ms=args.burst_duration,
        min_amplitude_ma=args.min_stim_amp,
    )

    fiber_parameters = FiberParameters(
        n_fibers=args.n_fibers,
        length_um=FIBERS_LENGTH_UM,
        mean_diameter_um=MEAN_DIAMETER_UM,
        std_diameter=STD_DIAMETER,
        propagation_delay_ms=PROPAGATION_DELAY_MS,
        fascicle_radius_um=FASCICLE_RADIUS_UM
    )

    simulation = AfferentStimulation(fiber_parameters, stim_parameters, args.sim_time)
    if args.results_folder is not None:
        simulation.set_results_folder(args.results_folder)
    simulation.run()
    if args.plot_response_stats:
        simulation.plot_stim_response_stats(args.sim_name, block=False)
    block = not args.non_blocking_plots
    simulation.plot(args.sim_name, block)
    if args.plot_window:
        start_from_stim_event_n = 3
        start_ms = AfferentStimulation.START_STIM_TIME_MS + start_from_stim_event_n * (1000./args.stim_freq) - 1
        simulation.plot(args.sim_name, block, window_ms=[start_ms, start_ms+args.plot_window_duration])
    simulation.save_results(args.sim_name)


if __name__ == '__main__':
    main()
