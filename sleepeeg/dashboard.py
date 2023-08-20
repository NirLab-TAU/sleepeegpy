import os
import time
from collections.abc import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mne import pick_types
from mne.io.pick import _picks_to_idx

from .pipeline import CleaningPipe, SpectralPipe


def _init_s_pipe(prec_pipe, path_to_hypnogram, hypno_freq):
    if path_to_hypnogram is None:
        s_pipe = SpectralPipe(
            prec_pipe=prec_pipe,
        )
        s_pipe.hypno = np.zeros(s_pipe.mne_raw.n_times)
        s_pipe.hypno_freq = s_pipe.sf
        s_pipe._upsample_hypno()
        sleep_stages = {"All": 0}
    else:
        s_pipe = SpectralPipe(
            prec_pipe=prec_pipe,
            path_to_hypno=path_to_hypnogram,
            hypno_freq=hypno_freq,
        )
        sleep_stages = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    return s_pipe, sleep_stages


def _get_min_max_psds(psds, picks):
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


def _filter(pipe, sfreq, fmin, fmax):
    if sfreq is None or sfreq >= pipe.sf:
        sfreq = pipe.sf
    else:
        pipe.resample(sfreq=sfreq, n_jobs=-1)

    notch_freqs = np.arange(50, int(pipe.sf / 2), 50)
    hp = pipe.mne_raw.info["highpass"]
    lp = pipe.mne_raw.info["lowpass"]
    if fmin is None or fmin <= hp:
        fmin = hp
    else:
        pipe.filter(l_freq=fmin, h_freq=None, n_jobs=-1)
    if fmax is None or fmax >= lp:
        fmax = lp
    else:
        pipe.filter(l_freq=None, h_freq=fmax, n_jobs=-1)

    if fmax > 50:
        pipe.notch(freqs=notch_freqs, n_jobs=-1)
    else:
        notch_freqs = [None]

    return sfreq, fmin, fmax, notch_freqs


def _hypno_psd(
    s_pipe,
    sleep_stages,
    hypno_psd_pick,
    ax_hypno,
    ax_psd,
    min_psd,
    max_psd,
    rba,
):
    s_pipe.compute_psds_per_stage(
        sleep_stages=sleep_stages,
        reference=None,
        method="welch",
        fmin=0,
        fmax=25,
        save=False,
        picks="eeg",
        reject_by_annotation=rba,
        n_jobs=-1,
        verbose=False,
        n_fft=2048,
        n_per_seg=768,
        n_overlap=512,
        window="hamming",
    )

    if min_psd is None and max_psd is None:
        min_psd, max_psd = _get_min_max_psds(s_pipe.psds, hypno_psd_pick)

    win_sec = s_pipe.mne_raw.n_times / s_pipe.sf / 900
    s_pipe.plot_hypnospectrogram(
        picks=hypno_psd_pick,
        win_sec=win_sec,
        freq_range=(0, 25),
        cmap="Spectral_r",
        overlap=True,
        axis=ax_hypno,
        reject_by_annotation="NaN" if rba else None,
    )
    r = max_psd - min_psd
    s_pipe.plot_psds(
        picks=hypno_psd_pick,
        psd_range=(min_psd - 0.1 * r, max_psd + 0.1 * r),
        freq_range=(0, 25),
        dB=True,
        xscale="linear",
        axis=ax_psd,
        legend_args=dict(loc="upper right", fontsize="medium"),
    )
    ax_psd.axvspan(0, s_pipe.mne_raw.info["highpass"], alpha=0.3, color="gray")
    return min_psd, max_psd


def _topo(s_pipe, reference, sleep_stages, topo_axes, topo_lims):
    if len(sleep_stages) == 1:
        stages = ["All"] * 4
        topo_axes[0, 0].set_title(f"Alpha (8-12 Hz)")
        topo_axes[0, 1].set_title(f"SMR (12-15 Hz)")
        topo_axes[1, 0].set_title(f"Delta (0.5-4 Hz)")
        topo_axes[1, 1].set_title(f"Theta (4-8 Hz)")
    else:
        stages = ["Wake", "N2", "N3", "REM"]
        topo_axes[0, 0].set_title(f"{stages[0]}, Alpha (8-12 Hz)")
        topo_axes[0, 1].set_title(f"{stages[1]}, SMR (12-15 Hz)")
        topo_axes[1, 0].set_title(f"{stages[2]}, Delta (0.5-4 Hz)")
        topo_axes[1, 1].set_title(f"{stages[3]}, Theta (4-8 Hz)")

    if reference != "average":
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

    s_pipe.plot_topomap(
        stage=stages[0],
        band={"Alpha": (8, 12)},
        dB=False,
        axis=topo_axes[0, 0],
        topomap_args=dict(cmap="plasma", vlim=topo_lims[0]),
        cbar_args=None,
    )

    s_pipe.plot_topomap(
        stage=stages[1],
        band={"SMR": (12, 15)},
        dB=False,
        axis=topo_axes[0, 1],
        topomap_args=dict(cmap="plasma", vlim=topo_lims[1]),
        cbar_args=None,
    )

    s_pipe.plot_topomap(
        stage=stages[2],
        band={"Delta": (0.5, 4)},
        dB=False,
        axis=topo_axes[1, 0],
        topomap_args=dict(cmap="plasma", vlim=topo_lims[2]),
        cbar_args=None,
    )

    s_pipe.plot_topomap(
        stage=stages[3],
        band={"Theta": (4, 8)},
        dB=False,
        axis=topo_axes[1, 1],
        topomap_args=dict(cmap="plasma", vlim=topo_lims[3]),
        cbar_args=None,
    )


def create_dashboard(
    subject_code: str | os.PathLike,
    path_to_eeg: str | os.PathLike,
    path_to_hypnogram: str | os.PathLike | None,
    hypno_freq: float | None,
    path_to_bad_channels: str | os.PathLike | None,
    path_to_annotations: str | os.PathLike | None,
    output_dir: str | os.PathLike,
    reference: Iterable[str] | str | None = None,
    resampling_freq: float | None = None,
    bandpass_filter_freqs: Iterable[float | None] = None,
    hypno_psd_pick: Iterable[str] | str = ["E101"],
    path_to_ica_fif: str | os.PathLike = None,
    save_fif: bool = False,
    topomap_cbar_limits: Sequence[tuple[float, float]] | None = None,
):
    """Applies cleaning, runs psd analyses and plots them on the dashboard.
    Can accept raw, resampled, filtered or cleaned (annotated) recording,
    but not after ica component exclusion.
    If annotated recording with already interpolated channels is provided -
    the interpolated channels will be extracted from mne_raw.info['description'].

    Args:
        subject_code: Subject code.
        path_to_eeg: Path to the raw mff file.
        path_to_hypnogram: Path to the hypnogram yasa-style hypnogram.
        hypno_freq: Sampling rate of the hypnogram in Hz.
        path_to_bad_channels: Path to bad_channels.txt saved by plot() method.
        path_to_annotations: Path to annotations saved by plot() method.
        output_dir: Directory to save the dashboard image in.
        reference: Reference to apply as accepts
            :py:meth:`mne:mne.io.Raw.set_eeg_reference`.
            Defaults to None.
        resampling_freq: New frequency in Hz. Defaults to None.
        bandpass_filter_freqs: Lower and upper bounds of the filter.
            Defaults to None.
        hypno_psd_pick: Channel to compute spectrogram and PSD plots for.
            Defaults to ['E101'].
        path_to_ica_fif: Path to ica components file. Defaults to None.
        save_fif: Whether to save cleaned fif. Defaults to False.
        topomap_cbar_limits: Power limits for topography plots.
            If None - will be adaptive. Defaults to None.
    """

    if bandpass_filter_freqs is None:
        bandpass_filter_freqs = [None, None]
    if topomap_cbar_limits is None:
        topomap_cbar_limits = [(None, None)] * 4
        is_adaptive_topo = True
    else:
        is_adaptive_topo = False

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
    pick = (
        hypno_psd_pick
        if isinstance(hypno_psd_pick, str)
        else ", ".join(str(x) for x in hypno_psd_pick)
    )
    spectrum_before.suptitle(f"Spectra after interpolating bad channels ({pick})")
    spectrum_after.suptitle(f"Spectra after rejecting bad data spans ({pick})")

    pipe = CleaningPipe(path_to_eeg=path_to_eeg, output_dir=output_dir)

    sfreq, fmin, fmax, notch_freqs = _filter(
        pipe,
        resampling_freq,
        bandpass_filter_freqs[0],
        bandpass_filter_freqs[1],
    )

    if path_to_bad_channels is not None:
        pipe.read_bad_channels(path=path_to_bad_channels)
        bads = pipe.mne_raw.info["bads"]
        pipe.interpolate_bads(reset_bads=True)
    else:
        from ast import literal_eval

        bads = literal_eval(pipe.mne_raw.info["description"])

    pipe.set_eeg_reference(ref_channels=reference)
    s_pipe, sleep_stages = _init_s_pipe(pipe, path_to_hypnogram, hypno_freq)
    min_psd, max_psd = _hypno_psd(
        s_pipe,
        sleep_stages,
        hypno_psd_pick,
        hypno_before,
        psd_before,
        min_psd=None,
        max_psd=None,
        rba=False,
    )

    if path_to_annotations is not None:
        pipe.read_annotations(path=path_to_annotations)

    is_ica = False
    if path_to_ica_fif:
        from sleepeeg.pipeline import ICAPipe

        is_ica = True
        pipe = ICAPipe(prec_pipe=pipe, path_to_ica=path_to_ica_fif)
        pipe.apply()

    if save_fif:
        pipe.save_raw(fname="dashboard_cleaned_raw.fif", overwrite=True)

    interpolated = _picks_to_idx(pipe.mne_raw.info, bads)
    cmap = LinearSegmentedColormap.from_list("", ["red", "red"])
    pipe.plot_sensors(
        legend=["Interpolated"],
        axes=info_axes[0],
        legend_args=dict(loc="lower left", bbox_to_anchor=(-0.1, 0), fontsize="small"),
        ch_groups=[interpolated],
        pointsize=20,
        linewidth=1.5,
        cmap=cmap,
    )

    recording_time = time.strftime(
        "%H:%M:%S", time.gmtime(pipe.mne_raw.n_times / pipe.sf)
    )

    interpolated_channels_percent = round(
        100 * len(interpolated) / len(pick_types(pipe.mne_raw.info, eeg=True)), 2
    )

    textstr = "\n\n".join(
        (
            f"Recording duration: {recording_time}",
            f"Sampling frequency: {sfreq} Hz",
            f"Bad data spans: {pipe.bad_data_percent}%",
            f"Interpolated channels: {interpolated_channels_percent}%",
            f"EEG reference: {reference}",
            f"Band-pass filter: [{round(fmin, 2)}, {round(fmax, 2)}] Hz",
            f"Notch filter: {set(notch_freqs)} Hz",
            f"ICA performed: {'Yes' if is_ica else 'No'}",
            f"Adaptive topomaps: {'Yes' if is_adaptive_topo else 'No'}",
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

    s_pipe, sleep_stages = _init_s_pipe(pipe, path_to_hypnogram, hypno_freq)

    _hypno_psd(
        s_pipe,
        sleep_stages,
        hypno_psd_pick,
        hypno_after,
        psd_after,
        min_psd,
        max_psd,
        rba=True,
    )

    _topo(s_pipe, reference, sleep_stages, topo_axes, topomap_cbar_limits)

    fig.savefig(f"{output_dir}/dashboard_{subject_code}.png")
