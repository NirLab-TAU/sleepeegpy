import os
import time
from ast import literal_eval
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mne.io.pick import _picks_to_idx
from numba.cuda.cudadrv.nvvm import logger

from .pipeline import CleaningPipe, ICAPipe, SpectralPipe


def _init_spectral_pipe(prec_pipe, hypnogram, hypno_freq, predict_hypno_args):
    if hypnogram is None:
        s_pipe = SpectralPipe(
            prec_pipe=prec_pipe,
        )
        s_pipe.hypno = np.zeros(s_pipe.mne_raw.n_times)
        s_pipe.hypno_freq = s_pipe.sf
        s_pipe._upsample_hypno()
        sleep_stages = {"All": 0}
    elif hypnogram == "predict":
        s_pipe = SpectralPipe(
            prec_pipe=prec_pipe,
            hypno_freq=hypno_freq,
        )
        s_pipe.predict_hypno(**predict_hypno_args)
        sleep_stages = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    else:
        s_pipe = SpectralPipe(
            prec_pipe=prec_pipe,
            path_to_hypno=hypnogram,
            hypno_freq=hypno_freq,
        )
        sleep_stages = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    return s_pipe, sleep_stages


def _get_min_max_psds(psds, hypno_psds_picks):
    max_psd = None
    min_psd = None
    for spectrum in psds.values():
        hypno_psd_data = spectrum.copy().pick(hypno_psds_picks)._data
        data = 10 * np.log10(10**12 * hypno_psd_data)
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
        pipe.resample(sfreq=sfreq)

    notch_freqs = np.arange(50, int(pipe.sf / 2), 50)
    hp = pipe.mne_raw.info["highpass"]
    lp = pipe.mne_raw.info["lowpass"]
    if fmin is None or fmin <= hp:
        fmin = hp
    else:
        pipe.filter(l_freq=fmin, h_freq=None)
    if fmax is None or fmax >= lp:
        fmax = lp
    else:
        pipe.filter(l_freq=None, h_freq=fmax)

    if fmax > 50:
        pipe.notch(freqs=notch_freqs)
    else:
        notch_freqs = [None]

    return sfreq, fmin, fmax, notch_freqs


def _hypno_psd(
    spectral_pipe,
    sleep_stages,
    hypno_psd_pick,
    hypno_axes,
    psd_axes,
    min_psd,
    max_psd,
    rba,
):
    spectral_pipe.compute_psd(
        sleep_stages=sleep_stages,
        reference=None,
        fmin=0,
        fmax=25,
        save=False,
        picks="eeg",
        reject_by_annotation=rba,
        verbose=False,
        n_fft=2048,
        n_per_seg=768,
        n_overlap=512,
        window="hamming",
    )

    if min_psd is None and max_psd is None:
        min_psd, max_psd = _get_min_max_psds(spectral_pipe.psds, hypno_psd_pick)

    win_sec = spectral_pipe.mne_raw.n_times / spectral_pipe.sf / 900
    spectral_pipe.plot_hypnospectrogram(
        picks=hypno_psd_pick,
        win_sec=win_sec,
        freq_range=(0, 25),
        cmap="Spectral_r",
        overlap=True,
        axis=hypno_axes,
        reject_by_annotation="NaN" if rba else None,
    )
    r = max_psd - min_psd
    spectral_pipe.plot_psds(
        picks=hypno_psd_pick,
        psd_range=(min_psd - 0.1 * r, max_psd + 0.1 * r),
        freq_range=(0, 25),
        dB=True,
        xscale="linear",
        axis=psd_axes,
        legend_args=dict(loc="upper right", fontsize="medium"),
    )
    psd_axes.axvspan(0, spectral_pipe.mne_raw.info["highpass"], alpha=0.3, color="gray")
    return min_psd, max_psd


def _plot_dashboard_topographies(
    spectral_pipe, reference, sleep_stages, topo_axes, topo_lims
):
    if len(sleep_stages) == 1:
        stages = ["All"] * 4
        topo_axes[0, 0].set_title(f"Alpha (8-12 Hz)")
        topo_axes[0, 1].set_title(f"Sigma (12-15 Hz)")
        topo_axes[1, 0].set_title(f"Delta (0.5-4 Hz)")
        topo_axes[1, 1].set_title(f"Theta (4-8 Hz)")
    else:
        stages = ["Wake", "N2", "N3", "REM"]
        topo_axes[0, 0].set_title(f"{stages[0]}, Alpha (8-12 Hz)")
        topo_axes[0, 1].set_title(f"{stages[1]}, Sigma (12-15 Hz)")
        topo_axes[1, 0].set_title(f"{stages[2]}, Delta (0.5-4 Hz)")
        topo_axes[1, 1].set_title(f"{stages[3]}, Theta (4-8 Hz)")

    if reference != "average":
        spectral_pipe.compute_psd(
            sleep_stages=sleep_stages,
            reference="average",
            fmin=0,
            fmax=25,
            save=False,
            picks="eeg",
            reject_by_annotation=True,
            verbose=False,
            n_fft=2048,
            n_per_seg=2048,
            n_overlap=1024,
            window="hamming",
        )

    spectral_pipe.plot_topomap(
        stage=stages[0],
        band={"Alpha": (8, 12)},
        dB=False,
        axis=topo_axes[0, 0],
        topomap_args=dict(cmap="plasma", vlim=topo_lims[0]),
        cbar_args=None,
    )

    spectral_pipe.plot_topomap(
        stage=stages[1],
        band={"Sigma": (12, 15)},
        dB=False,
        axis=topo_axes[0, 1],
        topomap_args=dict(cmap="plasma", vlim=topo_lims[1]),
        cbar_args=None,
    )

    spectral_pipe.plot_topomap(
        stage=stages[2],
        band={"Delta": (0.5, 4)},
        dB=False,
        axis=topo_axes[1, 0],
        topomap_args=dict(cmap="plasma", vlim=topo_lims[2]),
        cbar_args=None,
    )

    spectral_pipe.plot_topomap(
        stage=stages[3],
        band={"Theta": (4, 8)},
        dB=False,
        axis=topo_axes[1, 1],
        topomap_args=dict(cmap="plasma", vlim=topo_lims[3]),
        cbar_args=None,
    )


def _check_pipe_folders(output_dir):
    pipes = ["CleaningPipe", "ICAPipe", "SpectralPipe"]
    p = Path(output_dir)
    children = [f.name for f in p.iterdir()]
    return {pipe: pipe in children for pipe in pipes}


def create_dashboard(
    subject_code: str | os.PathLike,
    path_to_eeg: str | os.PathLike | None = None,
    hypnogram: str | os.PathLike | None = None,
    hypno_freq: float | None = None,
    predict_hypno_args: dict | None = None,
    output_dir: str | os.PathLike | None = None,
    reference: Iterable[str] | str | None = None,
    hypno_psd_pick: Iterable[str] | str = ["E101"],
    resampling_freq: float | None = None,
    bandpass_filter_freqs: Iterable[float | None] = None,
    path_to_ica_fif: str | os.PathLike = None,
    path_to_bad_channels: str | os.PathLike | None = None,
    path_to_annotations: str | os.PathLike | None = None,
    power_colorbar_limits: Sequence[tuple[float, float]] | None = None,
    prec_pipe: ICAPipe | CleaningPipe | None = None,
):
    """Applies cleaning, runs psd analyses and plots them on the dashboard.
    Can accept raw, resampled, filtered or cleaned (annotated) recording,
    but not after ica component exclusion.
    If annotated recording with already interpolated channels is provided -
    the interpolated channels will be extracted from mne_raw.info['description'].

    Args:
        subject_code: Subject code.
        path_to_eeg: Path to the raw mff file.
        hypnogram: Either path to the yasa-style hypnogram or
            'predict' for hypnogram prediction using YASA.
            If 'predict', make sure to pass predict_hypno_args.
            Defaults to None.
        hypno_freq: Sampling rate of the hypnogram in Hz.
        predict_hypno_args: dict containing 'eeg_name', 'eog_name',
            'emg_name', 'ref_name' and 'save' keys. First four should be
            channel names according to YASA's suggestions, 'save' is a bool
            for whether to save predicted hypnogram as a file. Defaults to None.
        output_dir: Directory to save the dashboard image in.
        reference: Reference to apply as accepts
            :py:meth:`mne:mne.io.Raw.set_eeg_reference`.
            Defaults to None.
        hypno_psd_pick: Channel to compute spectrogram and PSD plots for.
            Defaults to ['E101'].
        resampling_freq: New frequency in Hz. Defaults to None.
        bandpass_filter_freqs: Lower and upper bounds of the filter.
            Defaults to None.
        path_to_ica_fif: Path to ica components file. Defaults to None.
        path_to_bad_channels: Path to bad_channels.txt.
        path_to_annotations: Path to annotations.
        power_colorbar_limits: Power limits for topography plots.
            If None - will be adaptive. Defaults to None.
        prec_pipe: A pipe object from which to build the dashboard.
            If of type ICAPipe, the components should be marked for exclusion,
            but not applied. Defaults to None.
    """
    fig = plt.figure(layout="constrained", figsize=(1600 / 96, 1200 / 96), dpi=96)
    fig.suptitle(f"Dashboard <{subject_code}>")
    grid_spec = fig.add_gridspec(5, 4)
    is_adaptive_topo = not power_colorbar_limits
    if bandpass_filter_freqs is None:
        bandpass_filter_freqs = [None, None]
    if power_colorbar_limits is None:
        power_colorbar_limits = [(None, None)] * 4

    predict_hypno_args = _validate_hypno_args(hypnogram, predict_hypno_args)

    pipe_folders = _check_pipe_folders(
        output_dir=prec_pipe.output_dir if prec_pipe else output_dir
    )
    pipe = get_cleaning_pipe(output_dir, path_to_eeg, prec_pipe)
    bads, fmax, fmin, notch_freqs, sfreq = _filter_and_manage_bads(
        bandpass_filter_freqs, path_to_bad_channels, pipe, resampling_freq
    )
    if reference:
        pipe.set_eeg_reference(ref_channels=reference)
    spectral_pipe, sleep_stages = _init_spectral_pipe(
        pipe, hypnogram, hypno_freq, predict_hypno_args
    )

    picks_str_repr = (
        hypno_psd_pick
        if isinstance(hypno_psd_pick, str)
        else ", ".join(str(x) for x in hypno_psd_pick)
    )
    try:
        max_psd, min_psd = _plot_before_parts(
            fig, grid_spec, hypno_psd_pick, picks_str_repr, spectral_pipe, sleep_stages
        )
    except:
        logger.error(
            "Failed to plot 'before' graph. It will be missing from the dashboard"
        )

    if path_to_annotations is not None:
        pipe.read_annotations(path=path_to_annotations)

    is_ica, pipe = _get_ica_pipe(path_to_ica_fif, pipe, prec_pipe)
    try:
        psd_after = _plot_dashboard_info(
            bads,
            fig,
            fmax,
            fmin,
            grid_spec,
            is_adaptive_topo,
            is_ica,
            notch_freqs,
            pipe,
            reference,
            sfreq,
        )
        spectral_pipe.mne_raw = pipe.mne_raw
        _plot_after_dashboard(
            fig,
            grid_spec,
            hypno_psd_pick,
            max_psd,
            min_psd,
            picks_str_repr,
            psd_after,
            sleep_stages,
            spectral_pipe,
        )
    except:
        logger.error(
            "Failed to plot 'after' graph. It will be missing from the dashboard"
        )

    topo_subfig = fig.add_subfigure(grid_spec[0:2, 2:4])
    topo_axes = topo_subfig.subplots(2, 2)
    try:
        _plot_dashboard_topographies(
            spectral_pipe, reference, sleep_stages, topo_axes, power_colorbar_limits
        )
    except:
        logger.error("Failed to plot topography. It will be missing from the dashboard")

    for pipe_name, is_existed in pipe_folders.items():
        if not is_existed:
            try:
                os.rmdir(pipe.output_dir / pipe_name)
            except:
                pass
    fig.savefig(pipe.output_dir / f"dashboard_{subject_code}.png")
    return fig


def _filter_and_manage_bads(
    bandpass_filter_freqs, path_to_bad_channels, pipe, resampling_freq
):
    sfreq, fmin, fmax, notch_freqs = _filter(
        pipe,
        resampling_freq,
        bandpass_filter_freqs[0],
        bandpass_filter_freqs[1],
    )

    if path_to_bad_channels is not None:
        pipe.read_bad_channels(path=path_to_bad_channels)
        pipe.interpolate_bads(reset_bads=True)

    bads = []
    mne_info = pipe.mne_raw.info
    if path_to_bad_channels is not None:
        bads = mne_info["bads"]
    elif mne_info and "description" in mne_info and mne_info["description"] is not None:
        bads = literal_eval(mne_info["description"])
    return bads, fmax, fmin, notch_freqs, sfreq


def _plot_after_dashboard(
    fig,
    grid_spec,
    hypno_psd_pick,
    max_psd,
    min_psd,
    picks_str_repr,
    psd_after,
    sleep_stages,
    spectral_pipe,
):
    hypno_after_axes = fig.add_subplot(grid_spec[2:3, 2:4])
    hypno_after_axes.set_title(
        f"Spectra after rejecting bad data spans ({picks_str_repr})"
    )
    _hypno_psd(
        spectral_pipe,
        sleep_stages,
        hypno_psd_pick,
        hypno_after_axes,
        psd_after,
        min_psd,
        max_psd,
        rba=True,
    )
    psd_after.get_legend().remove()
    hypno_after_axes.yaxis.set_label_coords(-0.05, 0.5)
    psd_after.yaxis.set_label_coords(-0.05, 0.5)


def _plot_dashboard_info(
    bads,
    fig,
    fmax,
    fmin,
    grid_spec,
    is_adaptive_topo,
    is_ica,
    notch_freqs,
    pipe,
    reference,
    sfreq,
):
    if len(bads) == 0:
        bads = None
    interpolated = _picks_to_idx(pipe.mne_raw.info, bads)
    cmap = LinearSegmentedColormap.from_list("", ["red", "red"])
    info_subfig = fig.add_subfigure(grid_spec[0:2, 0:2])
    info_axes = info_subfig.subplots(1, 2)
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
        100
        * len(interpolated)
        / len(pipe.mne_raw.copy().pick(picks="eeg").info["ch_names"]),
        2,
    )

    info_txt: str = "\n\n".join(
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
        info_txt,
        transform=info_axes[1].transAxes,
        fontsize="large",
        verticalalignment="center",
        horizontalalignment="left",
    )
    psd_after = fig.add_subplot(grid_spec[3:5, 2:4])
    return psd_after


def _plot_before_parts(
    fig, grid_spec, hypno_psd_pick, picks_str_repr, s_pipe, sleep_stages
):
    psd_before_axes = fig.add_subplot(grid_spec[3:5, 0:2])
    hypno_before_axes = fig.add_subplot(grid_spec[2:3, 0:2])
    hypno_before_axes.set_title(
        f"Spectra after interpolating bad channels ({picks_str_repr})"
    )
    hypno_before_axes.yaxis.set_label_coords(-0.05, 0.5)

    min_psd, max_psd = _hypno_psd(
        s_pipe,
        sleep_stages,
        hypno_psd_pick,
        hypno_before_axes,
        psd_before_axes,
        min_psd=None,
        max_psd=None,
        rba=False,
    )
    psd_before_axes.yaxis.set_label_coords(-0.05, 0.5)
    return max_psd, min_psd


def _get_ica_pipe(path_to_ica_fif, pipe, prec_pipe):
    is_ica = False
    if path_to_ica_fif:
        is_ica = True
        pipe = ICAPipe(prec_pipe=pipe, path_to_ica=path_to_ica_fif)
        pipe.apply()
    elif isinstance(prec_pipe, ICAPipe):
        is_ica = True
        pipe = prec_pipe
        pipe.apply()
    return is_ica, pipe


def get_cleaning_pipe(output_dir, path_to_eeg, prec_pipe):
    if isinstance(prec_pipe, CleaningPipe):
        pipe = prec_pipe
    elif isinstance(prec_pipe, ICAPipe):
        pipe = CleaningPipe(prec_pipe=prec_pipe)
    elif prec_pipe is None:
        pipe = CleaningPipe(path_to_eeg=path_to_eeg, output_dir=output_dir)
    else:
        raise TypeError("prec_pipe expected to be CleaningPipe or ICAPipe")
    return pipe


def _validate_hypno_args(hypnogram, predict_hypno_args):
    predict_params = {"eeg_name", "eog_name", "emg_name", "ref_name"}
    predict_hypno_args = predict_hypno_args or dict()
    if hypnogram == "predict":
        if set(predict_hypno_args.keys()) < predict_params:
            raise ValueError(
                "predict_hypno_args should include all of the keys: 'eeg_name', 'eog_name', 'emg_name', 'ref_name'."
            )
    return predict_hypno_args
