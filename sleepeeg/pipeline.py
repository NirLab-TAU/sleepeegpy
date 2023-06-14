"""This module contains and describes pipe elements for sleep eeg analysis.
"""

from attrs import define, field
from loguru import logger
from typing import TypeVar
from pathlib import Path
from collections.abc import Iterable
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


from .base import (
    BasePipe,
    BaseHypnoPipe,
    BaseEventPipe,
    BaseTopomap,
    SpectrumPlots,
    SleepSpectrum,
    logger_wraps,
)

# For type annotation of pipe elements.
BasePipeType = TypeVar("BasePipeType", bound="BasePipe")


@define(kw_only=True)
class CleaningPipe(BasePipe):
    """The cleaning pipeline element.

    Contains resampling function, band and notch filters,
    browser for manual selection of bad channels
    and bad data spans.
    """

    @logger_wraps()
    def resample(self, sfreq: float = 250, save: bool = False, **resample_kwargs):
        """A wrapper for :py:meth:`mne:mne.io.Raw.resample`
        with an additional option to save the resampled data.

        Args:
            save: Whether to save a resampled data to a fif file. Defaults to False.
            **resample_kwargs: Arguments passed to :py:meth:`mne:mne.io.Raw.resample`.
        """
        self.mne_raw.resample(sfreq=sfreq, **resample_kwargs)
        if save:
            self.save_raw(
                "_".join(
                    filter(
                        None,
                        [
                            "resampled",
                            str(sfreq) + "hz",
                            "raw.fif",
                        ],
                    )
                )
            )

    @logger_wraps()
    def filter(
        self, l_freq: float | None = 0.3, h_freq: float | None = None, **filter_kwargs
    ):
        """A wrapper for :py:meth:`mne:mne.io.Raw.filter`.

        Args:
            **filter_kwargs: Arguments passed to :py:meth:`mne:mne.io.Raw.filter`.
        """
        self.mne_raw.load_data().filter(l_freq=l_freq, h_freq=h_freq, **filter_kwargs)

    @logger_wraps()
    def notch(self, freqs: str | Iterable[float] = "50s", **notch_kwargs):
        """A wrapper for :py:meth:`mne:mne.io.Raw.notch_filter`.

        Args:
            **notch_kwargs: Arguments passed to :py:meth:`mne:mne.io.Raw.notch_filter`.
        """
        if freqs == "50s":
            freqs = np.arange(50, int(self.sf / 2), 50)
        elif freqs == "60s":
            freqs = np.arange(50, int(self.sf / 2), 50)
        self.mne_raw.load_data().notch_filter(freqs=freqs, **notch_kwargs)

    def read_bad_channels(self, path: str | None = None):
        """Imports bad channels from file to mne raw object.

        Args:
            path: Path to the txt file with bad channel name per row. Defaults to None.
        """
        p = (
            Path(path)
            if path
            else self.output_dir / self.__class__.__name__ / "bad_channels.txt"
        )
        with open(p, "r") as f:
            self.mne_raw.info["bads"] = list(filter(None, f.read().split("\n")))

    def read_annotations(self, path: str | None = None):
        """Imports annotations from file to mne raw object

        Args:
            path: Path to txt file with mne-style annotations. Defaults to None.
        """
        from mne import read_annotations

        p = (
            Path(path)
            if path
            else self.output_dir / self.__class__.__name__ / "annotations.txt"
        )
        self.mne_raw.set_annotations(read_annotations(p))

    @logger_wraps()
    def interpolate_bads(self, **interp_kwargs):
        """A wrapper for :py:meth:`mne:mne.io.Raw.interpolate_bads`

        Args:
            **interp_kwargs: Arguments passed to :py:meth:`mne:mne.io.Raw.interpolate_bads`.
        """
        bads = self.mne_raw.info["bads"]
        self.mne_raw.interpolate_bads(**interp_kwargs)
        logger.info(f"Interpolated channels: {bads}")


@define(kw_only=True)
class ICAPipe(BasePipe):
    """The ICA pipeline element.

    Contains ica fitting, plotting multiple ica plots,
    selecting ica exclusion components and
    its application to the raw data.
    More at :py:class:`mne:mne.preprocessing.ICA`.
    """

    mne_ica: mne.preprocessing.ICA = field()
    """Instance of :py:class:`mne:mne.preprocessing.ICA`.
    """

    def __init__(
        self,
        prec_pipe: BasePipeType | None = None,
        path_to_eeg: str = "",
        output_dir: str = "",
        method: str = "fastica",
        n_components: int | float | None = None,
        fit_params: dict | None = None,
        path_to_ica: str | None = None,
        **ica_kwargs,
    ):
        """

        Args:
            prec_pipe: Preceding pipe that hands over mne_raw attribute. Defaults to None.
            path_to_eeg: Can be any file type supported by :py:func:`mne:mne.io.read_raw`. Defaults to None.
            output_dir: Path to the directory where the output will be saved. Defaults to None.
            method: The ICA method to use in the fit method. Defaults to 'fastica'.
            n_components: Number of principal components (from the pre-whitening PCA step)
                that are passed to the ICA algorithm during fitting:
                read more at :py:class:`mne:mne.preprocessing.ICA`. Defaults to None.
            fit_params:
                Additional parameters passed to the ICA estimator as specified by method.
                Allowed entries are determined by the various algorithm implementations:
                see `FastICA <https://scikit-learn.org/stable/modules/generated/sklearn.
                decomposition. FastICA.html#sklearn.decomposition.FastICA>`_,
                `picard <https://pierreablin.github.io/picard/generated/picard.picard.html#picard.picard>`_ and
                `infomax <https://mne.tools/stable/generated/mne.preprocessing.infomax.html#mne.preprocessing.infomax>`_.
                Defaults to None.
            path_to_ica: Path to the saved -ica.fif file you want to continue work with.
            **ica_kwargs: Arguments passed to :py:class:`mne:mne.preprocessing.ICA`.
        """
        if path_to_ica is not None:
            ica = mne.preprocessing.read_ica(path_to_ica)
        else:
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                method=method,
                fit_params=fit_params,
                **ica_kwargs,
            )
        if prec_pipe is not None:
            self.__attrs_init__(prec_pipe=prec_pipe, mne_ica=ica)
        else:
            self.__attrs_init__(
                path_to_eeg=path_to_eeg,
                output_dir=output_dir,
                mne_ica=ica,
            )
        self.mne_raw.load_data()

    @logger_wraps()
    def fit(self, filter_kwargs: dict = None, **fit_kwargs):
        """Highpass-filters (1 Hz) a copy of the mne_raw object
        and then runs :py:meth:`mne:mne.preprocessing.ICA.fit`.

        Args:
            filter_args: Arguments passed to :py:meth:`mne:mne.io.Raw.filter`.
            **fit_kwargs: Arguments passed to :py:meth:`mne:mne.preprocessing.ICA.fit`.
        """
        filter_kwargs = filter_kwargs or dict()
        filter_kwargs.setdefault("l_freq", 1.0)
        filter_kwargs.setdefault("h_freq", None)
        if self.mne_raw.info["highpass"] < 1.0:
            filtered_raw = self.mne_raw.copy()
            filtered_raw.filter(**filter_kwargs)
        else:
            filtered_raw = self.mne_raw
        self.mne_ica.fit(filtered_raw, **fit_kwargs)

    def plot_sources(self, **kwargs):
        """A wrapper for :py:meth:`mne:mne.preprocessing.ICA.plot_sources`."""
        self.mne_ica.plot_sources(inst=self.mne_raw, **kwargs)

    def plot_components(self, **kwargs):
        """A wrapper for :py:meth:`mne:mne.preprocessing.ICA.plot_components`."""
        self.mne_ica.plot_components(inst=self.mne_raw, **kwargs)

    def plot_properties(self, picks=None, **kwargs):
        """A wrapper for :py:meth:`mne:mne.preprocessing.ICA.plot_properties`."""
        self.mne_ica.plot_properties(self.mne_raw, picks=picks, **kwargs)

    @logger_wraps()
    def apply(self, exclude=None, **kwargs):
        """A wrapper for :py:meth:`mne:mne.preprocessing.ICA.apply`."""
        logger.info(
            f"Excluded ICA components: {list(set((exclude or [])+(self.mne_ica.exclude or [])))}"
        )
        self.mne_ica.apply(self.mne_raw, exclude=exclude, **kwargs)

    @logger_wraps()
    def save_ica(self, fname: str = "data-ica.fif", overwrite: bool = False):
        """A wrapper for :py:meth:`mne:mne.preprocessing.ICA.save`.

        Args:
            fname: filename for the ica file being saved. Defaults to "data-ica.fif".
            overwrite: Whether to overwrite the file. Defaults to False.
        """
        fif_folder = self.output_dir / self.__class__.__name__
        fif_folder.mkdir(exist_ok=True)
        self.mne_ica.save(fif_folder / fname, overwrite=overwrite)


@define(kw_only=True)
class SpectralPipe(BaseHypnoPipe, SpectrumPlots):
    """The spectral analyses pipeline element.

    Contains methods for computing and plotting PSD,
    spectrogram and topomaps per sleep stage.
    """

    psds: dict = field(init=False, factory=dict)
    """Instances of :class:`.SleepSpectrum` per sleep stage.
    """

    @logger_wraps()
    def compute_psds_per_stage(
        self,
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        reference: Iterable[str] | str | None = None,
        method: str = "welch",
        fmin: float = 0,
        fmax: float = 60,
        picks: str | Iterable[str] = "eeg",
        reject_by_annotation: bool = True,
        save: bool = False,
        overwrite: bool = False,
        n_jobs: bool = -1,
        verbose: bool = False,
        **psd_kwargs,
    ):
        """For each sleep stage creates a :class:`.SleepSpectrum` object.

        Args:
            sleep_stages: Sleep stages mapping in hypnogram.
                Defaults to {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.
            reference: Which eeg reference to compute PSD with.
                If None, the reference isn't changed. Defaults to None.
            method: Spectral estimation method.. Defaults to "welch".
            fmin: Lower frequency bound. Defaults to 0.
            fmax: Upper frequency bound. Defaults to 60.
            picks: Channels to compute spectra for. Refer to :py:meth:`mne:mne.io.Raw.pick`.
                Defaults to "eeg".
            reject_by_annotation: Whether to not use the annotations for the spectra computation.
                Defaults to True.
            save: Whether to save the spectra in .h5 files. Defaults to False.
            overwrite: Whether to overwrite the file. Defaults to False.
            n_jobs: _description_. Defaults to -1.
            verbose: _description_. Defaults to False.
            **psd_kwargs: Additional arguments passed to :py:func:`mne:mne.time_frequency.psd_array_welch`
                or :py:func:`mne:mne.time_frequency.psd_array_multitaper`.
        """
        inst = self.mne_raw.copy().load_data()
        if reference is not None:
            inst.set_eeg_reference(ref_channels=reference)
        for stage, stage_idx in sleep_stages.items():
            self.psds[stage] = SleepSpectrum(
                inst,
                hypno=self.hypno_up,
                stage_idx=stage_idx,
                method=method,
                fmin=fmin,
                fmax=fmax,
                tmin=None,
                tmax=None,
                picks=picks,
                proj=False,
                reject_by_annotation=reject_by_annotation,
                n_jobs=n_jobs,
                verbose=verbose,
                **psd_kwargs,
            )
        if save:
            import re

            for stage, spectrum in self.psds.items():
                stage = re.sub(r"[^\w\s-]", "_", stage)
                spectrum.save(
                    self.output_dir / self.__class__.__name__ / f"{stage}-psd.h5",
                    overwrite=overwrite,
                )

    @logger_wraps()
    def read_spectra(self, dirpath: str | None = None):
        """Loads spectra stored in hdf5 files.

        Filenames should end with {sleep_stage}-psd.h5

        Args:
            dirpath: Path to the directory containing hdf5 files. Defaults to None.
        """
        from mne.time_frequency import read_spectrum
        from pathlib import Path
        import re

        r = f"(.+)(?:-psd.h5)"
        self.tfrs = dict()
        dirpath = (
            Path(dirpath) if dirpath else self.output_dir / self.__class__.__name__
        )
        for p in dirpath.glob("*psd.h5"):
            m = re.search(r, str(p))
            if m:
                self.psds[m.groups()[0]] = read_spectrum(p)

    @logger_wraps()
    def plot_hypnospectrogram(
        self,
        picks: str | Iterable[str] = ("E101",),
        win_sec: float = 30,
        trimperc: float = 2.5,
        freq_range: tuple = (0, 40),
        cmap: str = "Spectral_r",
        overlap: bool = False,
        save: bool = False,
        axis: plt.Axes = None,
    ):
        """Plots hypnogram and spectrogram.

        Adapted from yasa.

        Args:
            picks: Channels to compute the spectrogram on. Defaults to ("E101",).
            win_sec: The length of the sliding window, in seconds, used for multitaper PSD calculation. Defaults to 30.
            trimperc: The amount of data to trim on both ends of the distribution when normalizing the colormap. Defaults to 2.5.
            freq_range: Range of x axis on spectrogram plot. Defaults to (0, 40).
            cmap: Matplotlib colormap. :std:doc:`mpl:tutorials/colors/colormaps`. Defaults to "Spectral_r".
            overlap: Whether to plot hypnogram over the spectrogram or on top of it. Defaults to False.
            save: Whether to save the figure. Defaults to False.
            axis: Instance of :py:class:`mpl:matplotlib.axes.Axes`.
                Defaults to None.
        """
        # Import data from the raw mne file.
        data = self.mne_raw.get_data(picks, units="uV", reject_by_annotation="NaN")[0]
        # Create a plot figure
        if overlap or not self.hypno_up.any():
            if axis is None:
                fig, axis = plt.subplots(nrows=1, figsize=(12, 4))

            im = self.__plot_spectrogram(
                data,
                self.sf,
                win_sec,
                freq_range[0],
                freq_range[1],
                trimperc,
                cmap,
                axis,
            )
            if self.hypno_up.any():
                ax_hypno = axis.twinx()
                self.__plot_hypnogram(self.sf, self.hypno_up, ax_hypno)
            # Add colorbar
            cbar = plt.colorbar(im, ax=axis, shrink=0.95, fraction=0.1, aspect=25)
            cbar.ax.set_ylabel(r"$\mu V^{2}/Hz$ (dB)", rotation=90)
            return None if axis is not None else fig
        else:
            if axis is None:
                fig, (ax0, ax1) = plt.subplots(
                    nrows=2, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 2]}
                )
                plt.subplots_adjust(hspace=0.1)
            else:
                ax0 = axis[0]
                ax1 = axis[1]

            if self.hypno_up.any():
                # Hypnogram (top axis)
                self.__plot_hypnogram(self.sf, self.hypno_up, ax0)
            # Spectrogram (bottom axis)
            self.__plot_spectrogram(
                data,
                self.sf,
                win_sec,
                freq_range[0],
                freq_range[1],
                trimperc,
                cmap,
                ax1,
            )

        # Save the figure if 'save' set to True
        if save and fig:
            fig.savefig(
                self.output_dir / self.__class__.__name__ / f"spectrogram.png",
                bbox_inches="tight",
            )

    @staticmethod
    def __plot_hypnogram(sf, hypno, ax0):
        """Adapted from :py:func:`yasa:yasa.plot_hypnogram`."""
        from pandas import Series

        hypno = np.asarray(hypno).astype(int)
        t_hyp = np.arange(hypno.size) / (sf * 3600)
        # Make sure that REM is displayed after Wake
        hypno = Series(hypno).map({-2: -2, -1: -1, 0: 0, 1: 2, 2: 3, 3: 4, 4: 1}).values
        # Hypnogram (top axis)
        ax0.step(t_hyp, -1 * hypno, color="k")
        if -2 in hypno and -1 in hypno:
            # Both Unscored and Artefacts are present
            ax0.set_yticks([2, 1, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(["Uns", "Art", "W", "R", "N1", "N2", "N3"])
            ax0.set_ylim(-4.5, 2.5)
        elif -2 in hypno and -1 not in hypno:
            # Only Unscored are present
            ax0.set_yticks([2, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(["Uns", "W", "R", "N1", "N2", "N3"])
            ax0.set_ylim(-4.5, 2.5)

        elif -2 not in hypno and -1 in hypno:
            # Only Artefacts are present
            ax0.set_yticks([1, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(["Art", "W", "R", "N1", "N2", "N3"])
            ax0.set_ylim(-4.5, 1.5)
        else:
            # No artefacts or Unscored
            ax0.set_yticks([0, -1, -2, -3, -4])
            ax0.set_yticklabels(["W", "R", "N1", "N2", "N3"])
            ax0.set_ylim(-4.5, 0.5)
        ax0.set_xlim(0, t_hyp.max())
        # ax0.set_ylabel("Stage")
        ax0.xaxis.set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.spines["top"].set_visible(False)
        return ax0

    @staticmethod
    def __plot_spectrogram(data, sf, win_sec, fmin, fmax, trimperc, cmap, ax):
        """Adapted from :py:func:`yasa:yasa.plot_spectrogram`."""
        from matplotlib.colors import Normalize
        from lspopt import spectrogram_lspopt
        import numpy as np

        # Calculate multi-taper spectrogram
        nperseg = int(win_sec * sf)
        f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
        Sxx = 10 * np.log10(
            Sxx, out=np.full(Sxx.shape, np.nan), where=(Sxx != 0)
        )  # Convert uV^2 / Hz --> dB / Hz

        # Select only relevant frequencies (up to 30 Hz)
        good_freqs = np.logical_and(f >= fmin, f <= fmax)
        Sxx = Sxx[good_freqs, :]
        f = f[good_freqs]
        t /= 3600  # Convert t to hours

        # Normalization
        vmin, vmax = np.nanpercentile(Sxx, [0 + trimperc, 100 - trimperc])
        norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(
            t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto"
        )
        ax.set_xlim(0, t.max())
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [hrs]")
        return im


@define(kw_only=True)
class SpindlesPipe(BaseEventPipe):
    """Spindles detection."""

    @logger_wraps()
    def detect(
        self,
        picks: str | Iterable[str] = ("eeg"),
        reference: Iterable[str] | str = "average",
        include: Iterable[int] = (1, 2, 3),
        freq_sp: Iterable[float] = (12, 15),
        freq_broad: Iterable[float] = (1, 30),
        duration: Iterable[float] = (0.5, 2),
        min_distance: int = 500,
        thresh: dict = {"corr": 0.65, "rel_pow": 0.2, "rms": 1.5},
        multi_only: bool = False,
        remove_outliers: bool = False,
        verbose: bool = False,
        save: bool = False,
    ):
        """A wrapper around :py:func:`yasa:yasa.spindles_detect` with option to save."""
        from yasa import spindles_detect

        self.results = spindles_detect(
            data=self.mne_raw.copy()
            .load_data()
            .set_eeg_reference(ref_channels=reference)
            .pick(picks),
            hypno=self.hypno_up,
            verbose=verbose,
            include=include,
            freq_sp=freq_sp,
            freq_broad=freq_broad,
            duration=duration,
            min_distance=min_distance,
            thresh=thresh,
            multi_only=multi_only,
            remove_outliers=remove_outliers,
        )
        if save:
            self._save_to_csv()


@define(kw_only=True)
class SlowWavesPipe(BaseEventPipe):
    """Slow waves detection."""

    @logger_wraps()
    def detect(
        self,
        picks: str | Iterable[str] = ("eeg"),
        reference: Iterable[str] | str = "average",
        include: Iterable[int] = (1, 2, 3),
        freq_sw: Iterable[float] = (0.3, 1.5),
        dur_neg: Iterable[float] = (0.3, 1.5),
        dur_pos: Iterable[float] = (0.1, 1),
        amp_neg: Iterable[float] = (40, 200),
        amp_pos: Iterable[float] = (10, 150),
        amp_ptp: Iterable[float] = (75, 350),
        coupling: bool = False,
        coupling_params: dict = {"freq_sp": (12, 16), "p": 0.05, "time": 1},
        remove_outliers: bool = False,
        save: bool = False,
    ):
        """A wrapper around :py:func:`yasa:yasa.sw_detect` with option to save."""
        from yasa import sw_detect

        self.results = sw_detect(
            data=self.mne_raw.copy()
            .load_data()
            .set_eeg_reference(ref_channels=reference)
            .pick(picks),
            hypno=self.hypno_up,
            verbose=False,
            include=include,
            freq_sw=freq_sw,
            dur_neg=dur_neg,
            dur_pos=dur_pos,
            amp_neg=amp_neg,
            amp_pos=amp_pos,
            amp_ptp=amp_ptp,
            coupling=coupling,
            coupling_params=coupling_params,
            remove_outliers=remove_outliers,
        )
        if save:
            self._save_to_csv()


@define(kw_only=True)
class RapidEyeMovementsPipe(BaseEventPipe):
    """Rapid eye movements detection."""

    @logger_wraps()
    def detect(
        self,
        loc_chname: str = "E46",
        roc_chname: str = "E238",
        reference: Iterable[str] | str = "average",
        include: int | Iterable[int] = 4,
        freq_rem: Iterable[float] = (0.5, 5),
        duration: Iterable[float] = (0.3, 1.2),
        amplitude: Iterable[float] = (50, 325),
        remove_outliers: bool = False,
        save: bool = False,
    ):
        """A wrapper around :py:func:`yasa:yasa.rem_detect` with option to save."""
        from yasa import rem_detect

        referenced = (
            self.mne_raw.copy().load_data().set_eeg_reference(ref_channels=reference)
        )
        loc = referenced.get_data([loc_chname], units="uV", reject_by_annotation="NaN")
        roc = referenced.get_data([roc_chname], units="uV", reject_by_annotation="NaN")
        self.results = rem_detect(
            loc=loc,
            roc=roc,
            sf=self.sf,
            hypno=self.hypno_up,
            verbose=False,
            include=include,
            freq_rem=freq_rem,
            duration=duration,
            amplitude=amplitude,
            remove_outliers=remove_outliers,
        )
        if save:
            self._save_to_csv()

    def plot_topomap(self):
        raise AttributeError(
            "'RapidEyeMovementsPipe' object has no attribute 'plot_topomap'"
        )

    def plot_topomap_collage(self):
        raise AttributeError(
            "'RapidEyeMovementsPipe' object has no attribute 'plot_topomap'"
        )


@define(kw_only=True)
class GrandPipe(SpectrumPlots, BaseTopomap):
    """The pipeline element combining results from multiple subjects.

    Contains methods for computing and plotting combined PSD,
    spectrogram and topomaps, per sleep stage.
    """

    output_dir: Path = field(converter=Path)
    """Path to the directory where the output will be saved."""

    @output_dir.validator
    def _validate_output_dir(self, attr, value):
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / self.__class__.__name__).mkdir(exist_ok=True)
        logger.remove()
        logger.add(self.output_dir / "pipeline.log")

    psds: dict = field(init=False, factory=dict)
    """Instances of :class:`.SleepSpectrum` per sleep stage.
    """

    pipes: Iterable[BaseHypnoPipe] = field()
    """Stores pipeline elements for multiple subjects.
    """

    mne_raw: mne.io.Raw = field(init=False)

    detection_results = field(init=False, factory=lambda: defaultdict(pd.DataFrame))

    @mne_raw.default
    def _get_mne_raw_from_pipe(self):
        return self.pipes[0].mne_raw

    @logger_wraps()
    def compute_psds_per_stage(
        self,
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        reference: Iterable[str] | str | None = None,
        method: str = "welch",
        fmin: float = 0,
        fmax: float = 60,
        average: str = "mean",
        picks: str | Iterable[str] = "eeg",
        reject_by_annotation: bool = True,
        save: bool = False,
        overwrite: bool = False,
        n_jobs: bool = -1,
        verbose: bool = False,
        **psd_kwargs,
    ):
        """For each sleep stage creates a :class:`.SleepSpectrum` object.

        Args:
            sleep_stages: Sleep stages mapping in hypnogram.
                Defaults to {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.
            reference: Which eeg reference to compute PSD with.
                If None, the reference isn't changed. Defaults to None.
            method: Spectral estimation method.. Defaults to "welch".
            fmin: Lower frequency bound. Defaults to 0.
            fmax: Upper frequency bound. Defaults to 60.
            picks: Channels to compute spectra for. Refer to :py:meth:`mne:mne.io.Raw.pick`.
                Defaults to "eeg".
            reject_by_annotation: Whether to not use the annotations for the spectra computation.
                Defaults to True.
            save: Whether to save the spectra in .h5 files. Defaults to False.
            overwrite: Whether to overwrite the file. Defaults to False.
            n_jobs: _description_. Defaults to -1.
            verbose: _description_. Defaults to False.
            **psd_kwargs: Additional arguments passed to :py:func:`mne:mne.time_frequency.psd_array_welch`
                or :py:func:`mne:mne.time_frequency.psd_array_multitaper`.
        """

        avg_func = np.median if average == "median" else np.mean
        psds = defaultdict(list)
        for pipe in self.pipes:
            inst = pipe.mne_raw.copy().load_data()
            if reference is not None:
                inst.set_eeg_reference(ref_channels=reference)
            for stage, stage_idx in sleep_stages.items():
                psds[stage].append(
                    SleepSpectrum(
                        inst,
                        hypno=pipe.hypno_up,
                        stage_idx=stage_idx,
                        method=method,
                        fmin=fmin,
                        fmax=fmax,
                        tmin=None,
                        tmax=None,
                        picks=picks,
                        proj=False,
                        reject_by_annotation=reject_by_annotation,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        **psd_kwargs,
                    )
                )

        for stage, spectra in psds.items():
            self.psds[stage] = avg_func(spectra, axis=0)

        if save:
            import re

            for stage, spectrum in self.psds.items():
                stage = re.sub(r"[^\w\s-]", "_", stage)
                spectrum.save(
                    self.output_dir / self.__class__.__name__ / f"{stage}-psd.h5",
                    overwrite=overwrite,
                )

    @logger_wraps()
    def spindles_detect(
        self,
        picks: str | Iterable[str] = ("eeg"),
        reference: Iterable[str] | str = "average",
        include: Iterable[int] = (1, 2, 3),
        freq_sp: Iterable[float] = (12, 15),
        freq_broad: Iterable[float] = (1, 30),
        duration: Iterable[float] = (0.5, 2),
        min_distance: int = 500,
        thresh: dict = {"corr": 0.65, "rel_pow": 0.2, "rms": 1.5},
        multi_only: bool = False,
        remove_outliers: bool = False,
        average: str = "mean",
        verbose: bool = False,
        save: bool = False,
        get_sync_events_args=None,
    ):
        """A wrapper around :py:func:`yasa:yasa.spindles_detect` with option to save."""
        from yasa import spindles_detect

        get_sync_events_args = get_sync_events_args or dict()
        for pipe_index, pipe in enumerate(self.pipes):
            inst = pipe.mne_raw.copy().load_data()
            if reference is not None:
                inst.set_eeg_reference(ref_channels=reference)
            detection_results = spindles_detect(
                data=inst.pick(picks),
                hypno=pipe.hypno_up,
                verbose=verbose,
                include=include,
                freq_sp=freq_sp,
                freq_broad=freq_broad,
                duration=duration,
                min_distance=min_distance,
                thresh=thresh,
                multi_only=multi_only,
                remove_outliers=remove_outliers,
            )
            self.detection_results["summary"] = pd.concat(
                [self.detection_results["summary"], detection_results.summary()]
            )
            events_signal = detection_results.get_sync_events()
            events_signal["pipe_index"] = pipe_index
            self.detection_results["events"] = pd.concat(
                [self.detection_results["events"], events_signal]
            )
        if save:
            self._save_to_csv()

    @logger_wraps()
    def plot_topomap(
        self,
        prop: str,
        stage: str = "N2",
        aggfunc: str = "mean",
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        axis: plt.axis = None,
        save: bool = False,
        topomap_args: dict = None,
        cbar_args: dict = None,
        subplots_args: dict = None,
    ):
        """Plots topomap for a sleep stage and some property of detected events.

        Args:
            prop: Any event property returned by self.results.summary().
            stage: One of the sleep_stages keys. Defaults to "N2".
            aggfunc: Averaging function, "mean" or "median". Defaults to "mean".
            sleep_stages: Mapping between sleep stages names and their integer representations.
                Defaults to {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.
            axis: Instance of :py:class:`mpl:matplotlib.axes.Axes`.
                Defaults to None.
            save: Whether to save the figure. Defaults to False.
            topomap_args: Arguments passed to :py:func:`mne:mne.viz.plot_topomap`.Defaults to None.
            cbar_args: Arguments passed to :py:func:`mpl:matplotlib.pyplot.colorbar`.Defaults to None.
            subplots_args: Arguments passed to :py:func:`mpl:matplotlib.pyplot.subplots`.Defaults to None.
        """
        from natsort import natsort_keygen
        from more_itertools import collapse
        from seaborn import color_palette

        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        subplots_args = subplots_args or dict()
        topomap_args.setdefault("cmap", color_palette("rocket_r", as_cmap=True))
        cbar_args.setdefault("label", prop)

        grouped_summary = self.detection_results["summary"].groupby(
            ["Channel", "Stage"]
        )
        grouped_summary = (
            grouped_summary.mean() if aggfunc == "mean" else grouped_summary.median()
        )
        grouped_summary = grouped_summary.sort_values(
            "Channel", key=natsort_keygen()
        ).reset_index()

        assert np.isin(
            sleep_stages[stage], grouped_summary["Stage"].unique()
        ).all(), "No such stage in the detected events, was it included in the detect method?"

        is_new_axis = False
        if not axis:
            fig, axis = plt.subplots(**subplots_args)
            is_new_axis = True

        per_stage = grouped_summary.loc[
            grouped_summary["Stage"].isin(collapse([sleep_stages[stage]]))
        ].groupby("Channel")
        per_stage = per_stage.mean() if aggfunc == "mean" else per_stage.median()
        per_stage = per_stage.sort_values("Channel", key=natsort_keygen()).reset_index()

        info = self.mne_raw.copy().pick(list(per_stage["Channel"].unique())).info

        topomap_args.setdefault("vlim", (per_stage[prop].min(), per_stage[prop].max()))
        self._plot_topomap(
            data=per_stage[prop],
            axis=axis,
            info=info,
            topomap_args=topomap_args,
            cbar_args=cbar_args,
        )
        if is_new_axis:
            fig.suptitle(f"{stage} {self.__class__.__name__[:-4]} ({prop})")
        if save and is_new_axis:
            fig.savefig(
                self.output_dir
                / self.__class__.__name__
                / f"topomap_{self.__class__.__name__[:-4].lower()}_{prop.lower()}.png"
            )

    def detect(self):
        raise AttributeError("'GrandPipe' object has no attribute 'detect'")
