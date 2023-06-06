import os
import numpy as np
from collections.abc import Iterable


def get_min_max_psds(psds, picks):
    max_psd = None
    min_psd = None
    for spectrum in psds.values():
        data = 10 * np.log10(10**12 * spectrum.copy().pick(picks)._data)
        data[data == -np.inf] = np.nan
        max_psd = (
            np.nanmax(data) if max_psd is None or max_psd < np.nanmax(data) else max_psd
        )
        min_psd = (
            np.nanmin(data) if min_psd is None or min_psd > np.nanmin(data) else min_psd
        )
    return min_psd, max_psd


def create_dashboard(
    subject_code: str | os.PathLike,
    path_to_mff: str | os.PathLike,
    resampling_freq: float,
    path_to_hypnogram: str | os.PathLike,
    hypno_freq: float,
    path_to_bad_channels: str | os.PathLike,
    path_to_annotations: str | os.PathLike,
    bandpass_filter_freqs: Iterable[float | None],
    reference: str,
    output_dir: str | os.PathLike,
    sleep_stages: dict = {"Wake": 0, "N1": 1, "N2/3": (2, 3), "REM": 4},
    path_to_ica_fif: str | os.PathLike = None,
    save_fif: bool = False,
):
    """Applies cleaning, runs psd analyses and plots them on the dashboard.

    Args:
        subject_code: Subject code
        path_to_mff: Path to the raw mff file
        resampling_freq: New frequency in Hz
        path_to_hypnogram: Path to the hypnogram yasa-style hypnogram
        hypno_freq: Sampling rate of the hypnogram in Hz.
        path_to_bad_channels: Path to bad_channels.txt saved by plot() method.
        path_to_annotations: Path to annotations saved by plot() method.
        bandpass_filter_freqs: Lower and upper bounds of the filter.
        reference: Reference to apply. Can be "mastoids", "average" or "VREF".
        output_dir: Directory to save the dashboard image in.
        sleep_stages: Mapping between stage names and indices in hypnogram.
            Defaults to {"Wake": 0, "N1": 1, "N2/3": (2, 3), "REM": 4}.
        path_to_ica_fif: Path to ica components file. Defaults to None.
        save_fif: Whether to save cleaned fif. Defaults to False.
    """
    from sleepeeg.pipeline import CleaningPipe, SpectralPipe
    import matplotlib.pyplot as plt
    import time
    from mne.io.pick import _picks_to_idx
    from mne import pick_types

    if reference == "mastoids":
        ref_channels = ["E94", "E190"]
    elif reference == "average":
        ref_channels = "average"
    elif reference == "VREF":
        ref_channels = ["VREF"]

    fig = plt.figure(layout="constrained", figsize=(1600 / 96, 1200 / 96), dpi=96)
    gs = fig.add_gridspec(5, 4)
    info_subfig = fig.add_subfigure(gs[0:2, 0:2])
    topo_subfig = fig.add_subfigure(gs[0:2, 2:4])
    info_axes = info_subfig.subplots(1, 2)
    topo_axes = topo_subfig.subplots(2, 2)
    spectrum_before = fig.add_subfigure(gs[2:5, 0:2])
    gs_s_bef = spectrum_before.add_gridspec(3, 1)
    spectrum_after = fig.add_subfigure(gs[2:5, 2:4])
    gs_s_aft = spectrum_after.add_gridspec(3, 1)
    hypno_before = spectrum_before.add_subplot(gs_s_bef[0:1])
    hypno_after = spectrum_after.add_subplot(gs_s_aft[0:1])
    psd_before = spectrum_before.add_subplot(gs_s_bef[1:3])
    psd_after = spectrum_after.add_subplot(gs_s_aft[1:3])

    fig.suptitle(f"Dashboard <{subject_code}>")
    spectrum_before.suptitle("Spectra after filtering")
    spectrum_after.suptitle("Spectra after removing bad channels & epochs")

    pipe = CleaningPipe(path_to_eeg=path_to_mff, output_dir=output_dir)
    pipe.mne_raw.load_data()
    pipe.resample(sfreq=resampling_freq, n_jobs=-1)
    pipe.filter(
        l_freq=bandpass_filter_freqs[0], h_freq=bandpass_filter_freqs[1], n_jobs=-1
    )
    notch_freqs = np.arange(50, int(pipe.sf / 2), 50)
    pipe.notch(freqs=notch_freqs, n_jobs=-1)
    pipe.set_eeg_reference(ref_channels=ref_channels)
    s_pipe = SpectralPipe(
        prec_pipe=pipe,
        path_to_hypno=path_to_hypnogram,
        hypno_freq=hypno_freq,
    )

    s_pipe.compute_psds_per_stage(
        sleep_stages=sleep_stages,
        reference=ref_channels,
        method="welch",
        fmin=0,
        fmax=25,
        save=False,
        picks="eeg",
        reject_by_annotation=True,
        n_jobs=-1,
        verbose=False,
        n_fft=2048,
        n_per_seg=768,
        n_overlap=512,
        window="hamming",
    )

    win_sec = s_pipe.mne_raw.n_times / s_pipe.sf / 900
    s_pipe.plot_hypnospectrogram(
        picks=["E101"],
        win_sec=win_sec,
        freq_range=(0, 25),
        cmap="Spectral_r",
        overlap=True,
        axis=hypno_before,
    )
    min_psd, max_psd = get_min_max_psds(s_pipe.psds, ["E101"])
    r = max_psd - min_psd
    s_pipe.plot_psds(
        picks=["E101"],
        psd_range=(min_psd - 0.1 * r, max_psd + 0.1 * r),
        freq_range=(0, 25),
        dB=True,
        xscale="linear",
        axis=psd_before,
        legend_args=dict(loc="upper right", fontsize="medium"),
    )

    pipe.read_bad_channels(path=path_to_bad_channels)
    bads = pipe.mne_raw.info["bads"]
    pipe.interpolate_bads(reset_bads=True)
    pipe.read_annotations(path=path_to_annotations)
    if path_to_ica_fif:
        from sleepeeg.pipeline import ICAPipe

        is_ica = True
        pipe = ICAPipe(prec_pipe=pipe, path_to_ica=path_to_ica_fif)
        pipe.apply()

    if save_fif:
        pipe.save_raw(fname="dashboard_cleaned_raw.fif")

    interpolated = _picks_to_idx(pipe.mne_raw.info, bads)
    pipe.plot_sensors(
        legend=["", "", "", "", "", "", "Interpolated", ""],
        axes=info_axes[0],
        legend_args=dict(loc="lower left", bbox_to_anchor=(-0.1, 0), fontsize="small"),
        ch_groups=[[], [], [], [], [], [], interpolated, []],
        pointsize=20,
        linewidth=1.5,
    )

    recording_time = time.strftime(
        "%H:%M:%S", time.gmtime(pipe.mne_raw.n_times / pipe.sf)
    )
    df = pipe.mne_raw.annotations.to_data_frame()
    bad_epochs_percent = round(
        100
        * (df[df.description.str.contains("bad", case=False)].duration.sum() * pipe.sf)
        / pipe.mne_raw.n_times,
        2,
    )
    interpolated_channels_percent = round(
        100 * len(interpolated) / len(pick_types(pipe.mne_raw.info, eeg=True)), 2
    )

    textstr = "\n\n".join(
        (
            f"Recording duration: {recording_time}",
            f"Sampling frequency: {resampling_freq} Hz",
            f"Bad epochs: {bad_epochs_percent}%",
            f"Interpolated channels: {interpolated_channels_percent}%",
            f"EEG reference: {reference}",
            f"Band-pass filter: [{bandpass_filter_freqs[0]}, {bandpass_filter_freqs[1]}] Hz",
            f"Notch filter: {set(notch_freqs)} Hz",
            f"ICA performed: {'Yes' if is_ica else 'No'}",
        )
    )
    info_axes[1].set_axis_off()
    info_axes[1].text(
        0,
        0.5,
        textstr,
        transform=info_axes[1].transAxes,
        fontsize="large",
        verticalalignment="center",
        horizontalalignment="left",
    )

    s_pipe = SpectralPipe(
        prec_pipe=pipe,
        path_to_hypno=path_to_hypnogram,
        hypno_freq=hypno_freq,
    )

    s_pipe.compute_psds_per_stage(
        sleep_stages=sleep_stages,
        reference=ref_channels,
        method="welch",
        fmin=0,
        fmax=25,
        save=False,
        picks="eeg",
        reject_by_annotation=True,
        n_jobs=-1,
        verbose=False,
        n_fft=2048,
        n_per_seg=768,
        n_overlap=512,
        window="hamming",
    )

    s_pipe.plot_hypnospectrogram(
        picks=["E101"],
        win_sec=win_sec,
        freq_range=(0, 25),
        cmap="Spectral_r",
        overlap=True,
        axis=hypno_after,
    )

    s_pipe.plot_psds(
        picks=["E101"],
        psd_range=(min_psd - 0.1 * r, max_psd + 0.1 * r),
        freq_range=(0, 25),
        dB=True,
        xscale="linear",
        axis=psd_after,
        legend_args=dict(loc="upper right", fontsize="medium"),
    )
    if ref_channels != "average":
        s_pipe.compute_psds_per_stage(
            sleep_stages=sleep_stages,
            reference="average",
            method="welch",
            fmin=0,
            fmax=25,
            save=False,
            picks="eeg",
            reject_by_annotation=True,
            n_jobs=-1,
            verbose=False,
            n_fft=2048,
            n_per_seg=2048,
            n_overlap=1024,
            window="hamming",
        )
    topo_axes[0, 0].set_title("Wake, Alpha (8-12 Hz)")
    s_pipe.plot_topomap(
        stage="Wake",
        band={"Alpha": (8, 12)},
        dB=False,
        axis=topo_axes[0, 0],
        topomap_args=dict(cmap="plasma"),
        cbar_args=None,
    )

    topo_axes[0, 1].set_title("N2, SMR (12-15 Hz)")
    s_pipe.plot_topomap(
        stage="N2",
        band={"SMR": (12, 15)},
        dB=False,
        axis=topo_axes[0, 1],
        topomap_args=dict(cmap="plasma"),
        cbar_args=None,
    )
    topo_axes[1, 0].set_title("N3, Delta (0.5-4 Hz)")
    s_pipe.plot_topomap(
        stage="N3",
        band={"Delta": (0.5, 4)},
        dB=False,
        axis=topo_axes[1, 0],
        topomap_args=dict(cmap="plasma"),
        cbar_args=None,
    )
    topo_axes[1, 1].set_title("REM, Theta (4-8 Hz)")
    s_pipe.plot_topomap(
        stage="REM",
        band={"Theta": (4, 8)},
        dB=False,
        axis=topo_axes[1, 1],
        topomap_args=dict(cmap="plasma"),
        cbar_args=None,
    )

    fig.savefig(f"{output_dir}/dashboard_{subject_code}.png")
