"""This module contains and describes pipe elements for sleep eeg analysis.
"""

from attrs import define, field
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mne.io


from collections.abc import Iterable

from sleepeeg.base import BasePipe, BaseHypnoPipe, BaseEventPipe, BaseSpectrum


@define(kw_only=True)
class CleaningPipe(BasePipe):
    """The cleaning pipeline element.

    Contains resampling function, band and notch filters,
    browser for manual selection of bad channels
    and bad data spans.
    """

    def resample(
        self,
        save: bool = False,
        mne_resample_args: dict = None,
    ):
        """A wrapper for
        `mne.io.Raw.resample <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample>`_
        with an additional option to save the resampled data.

        Args:
            save: Whether to save a resampled data to a fif file. Defaults to False.
            mne_resample_args: Arguments passed to `mne.io.Raw.resample <https://mne.tools/stable/
                generated/mne.io.Raw.html#mne.io.Raw.resample>`_. Defaults to None.
        """
        mne_resample_args = mne_resample_args or dict()
        mne_resample_args.setdefault("sfreq", 250)
        self.mne_raw.resample(**mne_resample_args)
        if save:
            self.save_raw(
                "_".join(
                    filter(
                        None,
                        [
                            "resampled",
                            str(mne_resample_args["sfreq"]) + "hz",
                            "raw.fif",
                        ],
                    )
                )
            )

    def filter(
        self,
        mne_filter_args: dict = None,
    ):
        """A wrapper for
        `mne.io.Raw.filter <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter>`_.

        Args:
            mne_filter_args: Arguments passed to `mne.io.Raw.filter <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter>`_.
        """
        mne_filter_args = mne_filter_args or dict()
        mne_filter_args.setdefault("l_freq", 0.3)
        self.mne_raw.load_data()
        self.mne_raw.filter(**mne_filter_args)

    def notch(
        self,
        mne_notch_args: dict = None,
    ):
        """A wrapper for
        `mne.io.Raw.notch_filter <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter>`_.

        Args:
            freqs: Frequencies to filter out from data,
                e.g. np.arange(50, 251, 50) for sampling freq 250 Hz.
                Defaults to None.
            picks: Channels to filter, if None - all channels will be filtered.
                Defaults to "eeg".
            n_jobs: The number of jobs to run in parallel or CUDA. Defaults to "cuda".
        """
        mne_notch_args = mne_notch_args or dict()
        mne_notch_args.setdefault("freqs", np.arange(50, int(self.sf / 2), 50))
        self.mne_raw.notch_filter(**mne_notch_args)

    def read_bad_channels(self, path=None):
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

    def read_annotations(self, path=None):
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


@define(kw_only=True)
class ICAPipe(BasePipe):
    """The ICA pipeline element.

    Contains ica fitting, plotting multiple ica plots,
    selecting ica exclusion components and
    its application to the raw data.
    More at `mne.preprocessing.ICA <https://mne.tools/stable/
    generated/mne.preprocessing.ICA.html#mne-preprocessing-ica>`_.
    """

    n_components: int | float | None = field(default=30)
    """Number of principal components (from the pre-whitening PCA step) 
    that are passed to the ICA algorithm during fitting.
    """

    method: str = field(default="fastica")
    """The ICA method to use in the fit method. 

    Can be 'fastica', 'infomax' or 'picard'
    Use the fit_params argument to set additional parameters.
    """
    fit_params: dict = field(default=None)
    """Additional parameters passed to the ICA estimator as specified by method. 
    Allowed entries are determined by the various algorithm implementations: 
    see `FastICA <https://scikit-learn.org/stable/modules/generated/sklearn.
    decomposition.FastICA.html#sklearn.decomposition.FastICA>`_, 
    `picard <https://pierreablin.github.io/picard/generated/picard.picard.html#picard.picard>`_ and
    `infomax <https://mne.tools/stable/generated/mne.preprocessing.infomax.html#mne.preprocessing.infomax>`_.
    """

    path_to_ica: Path = field(converter=Path, default="/")

    mne_ica: mne.preprocessing.ICA = field()
    """Instance of 
    `mne.preprocessing.ICA <https://mne.tools/stable/
    generated/mne.preprocessing.ICA.html#mne-preprocessing-ica>`_.
    """

    @mne_ica.default
    def _set_mne_ica(self):
        if self.path_to_ica == Path("/"):
            return mne.preprocessing.ICA(
                n_components=self.n_components,
                method=self.method,
                fit_params=self.fit_params,
            )
        return mne.preprocessing.read_ica(self.path_to_ica)

    def __attrs_post_init__(self):
        self.mne_raw.load_data()

    def fit(self, filter_args=None, ica_fit_args=None):
        """Highpass-filters (1Hz) a copy of the mne_raw object
        and then runs `mne.preprocessing.ICA.fit <https://mne.tools/stable/
        generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.fit>`_.
        """
        filter_args = filter_args or dict()
        ica_fit_args = ica_fit_args or dict()
        filter_args.setdefault("l_freq", 1.0)
        if self.mne_raw.info["highpass"] < 1.0:
            filtered_raw = self.mne_raw.copy()
            filtered_raw.filter(**filter_args)
        else:
            filtered_raw = self.mne_raw
        self.mne_ica.fit(filtered_raw, **ica_fit_args)

    def plot_sources(self, **kwargs):
        """A wrapper for `mne.preprocessing.ICA.plot_sources <https://mne.tools/stable/
        generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.plot_sources>`_.
        """
        self.mne_ica.plot_sources(self.mne_raw, block=True, **kwargs)

    def plot_components(self, **kwargs):
        """A wrapper for `mne.preprocessing.ICA.plot_components <https://mne.tools/stable/
        generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.plot_components>`_.
        """
        self.mne_ica.plot_components(inst=self.mne_raw, **kwargs)

    def plot_overlay(self, exclude=None, picks=None, start=10, stop=20, **kwargs):
        """A wrapper for `mne.preprocessing.ICA.plot_overlay <https://mne.tools/stable/
        generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.plot_overlay>`_.
        """
        self.mne_ica.plot_overlay(
            self.mne_raw, exclude=exclude, picks=picks, start=start, stop=stop, **kwargs
        )

    def plot_properties(self, picks=None, **kwargs):
        """A wrapper for `mne.preprocessing.ICA.plot_properties <https://mne.tools/stable/
        generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.plot_properties>`_.
        """
        self.mne_ica.plot_properties(self.mne_raw, picks=picks, **kwargs)

    def apply(self, exclude=None, **kwargs):
        """Remove selected components from the signal.

        A wrapper for `mne.preprocessing.ICA.apply <https://mne.tools/stable/
        generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.apply>`_.
        """
        self.mne_ica.apply(self.mne_raw, exclude=exclude, **kwargs)

    def save_ica(self, fname="data-ica.fif", overwrite=False):
        """A wrapper for `mne.preprocessing.ICA.save <https://mne.tools/stable/
        generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.save>`_.

        Args:
            fname: filename for the ica file being saved. Defaults to "data-ica.fif".
            overwrite: Whether to overwrite the file. Defaults to False.
        """
        fif_folder = self.output_dir / self.__class__.__name__ / "saved_ica"
        fif_folder.mkdir(exist_ok=True)
        self.mne_ica.save(fif_folder / fname, overwrite=overwrite)


@define(kw_only=True)
class SpectralPipe(BaseHypnoPipe, BaseSpectrum):
    """The spectral analyses pipeline element.

    Contains methods for computing and plotting PSD,
    spectrogram and topomaps, per sleep stage.
    """

    def plot_hypnospectrogram(
        self,
        picks: str | Iterable[str] = ("E101",),
        sec_per_seg: float = 4.096,
        freq_range: tuple = (0, 40),
        cmap: str = "inferno",
        overlap: bool = False,
        save: bool = False,
    ):
        """Plots hypnogram and spectrogram.

        Args:
            picks: Channels to calculate spectrogram on, more info at
                `mne.io.Raw.get_data <https://mne.tools/stable/generated/
                mne.io.Raw.html#mne.io.Raw.get_data>`_.
                Defaults to ("E101",).
            sec_per_seg: Segment length in seconds. Defaults to 4.096.
            freq_range: Range of x axis on spectrogram plot. Defaults to (0, 40).
            cmap: Matplotlib `colormap <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_. Defaults to "inferno".
            overlap: Whether to plot hypnogram over the spectrogram or on top of it. Defaults to False.
            save: Whether to save the figure. Defaults to False.
        """
        # Import data from the raw mne file.
        data = self.mne_raw.get_data(picks, units="uV", reject_by_annotation="NaN")[0]
        # Create a plot figure
        fig = self.__plot_hypnospectrogram(
            data,
            self.sf,
            self.hypno_up,
            win_sec=sec_per_seg,
            fmin=freq_range[0],
            fmax=freq_range[1],
            trimperc=0,
            cmap=cmap,
            overlap=overlap,
        )
        # Save the figure if 'save' set to True
        if save:
            fig.savefig(
                self.output_dir / self.__class__.__name__ / f"spectrogram.png",
                bbox_inches="tight",
            )

    def _compute_psd_per_stage(
        self, picks, sleep_stages, avg_ref, dB, method_args=None
    ):
        from collections import defaultdict
        from scipy import signal, ndimage

        assert (
            self.hypno.any()
        ), f"Hypnogram hasn't been provided, can't compute PSD per stage"

        method_args = method_args or dict()
        method_args["axis"] = 1
        if "nperseg" not in method_args:
            method_args["nperseg"] = self.sf * 4.096
        # Import data from the raw mne file.
        self.mne_raw.load_data()
        if avg_ref:
            data = (
                self.mne_raw.copy()
                .set_eeg_reference()
                .get_data(picks=picks, units="uV", reject_by_annotation="NaN")
            )
        else:
            data = self.mne_raw.get_data(
                picks=picks, units="uV", reject_by_annotation="NaN"
            )
        data = np.ma.array(data, mask=np.isnan(data))
        n_samples_total = np.ma.compress_cols(data).shape[1]
        psds = defaultdict(list)
        psd_per_stage = {}
        for stage, index in sleep_stages.items():
            weights = []
            n_samples = 0
            try:
                regions = ndimage.find_objects(
                    ndimage.label(
                        np.logical_or.reduce([self.hypno_up == i for i in index])
                    )[0]
                )
            except TypeError:
                regions = ndimage.find_objects(ndimage.label(self.hypno_up == index)[0])
            for region in regions:
                compressed = np.ma.compress_cols(data[:, region[0]])
                if compressed.size > method_args["nperseg"]:
                    weights.append(compressed.shape[1])
                    freqs, psd = signal.welch(
                        compressed,
                        fs=self.sf,
                        **method_args,
                    )
                    psds[stage].append(psd)
                    n_samples += compressed.shape[1]
            avg = np.average(np.array(psds[stage]), weights=weights, axis=0)
            psd_per_stage[stage] = [
                freqs,
                10 * np.log10(avg) if dB else avg,
                round(n_samples / n_samples_total * 100, 2),
            ]
        return psd_per_stage

    def __plot_hypnospectrogram(
        self, data, sf, hypno, win_sec, fmin, fmax, trimperc, cmap, overlap
    ):
        """
        ?
        """
        # Increase font size while preserving original
        old_fontsize = plt.rcParams["font.size"]
        plt.rcParams.update({"font.size": 18})

        if overlap or not hypno.any():
            fig, ax = plt.subplots(nrows=1, figsize=(12, 4))
            im = self.__plot_spectrogram(
                data, sf, win_sec, fmin, fmax, trimperc, cmap, ax
            )
            if hypno.any():
                ax_hypno = ax.twinx()
                self.__plot_hypnogram(sf, hypno, ax_hypno)
            # Add colorbar
            cbar = fig.colorbar(
                im, ax=ax, shrink=0.95, fraction=0.1, aspect=25, pad=0.1
            )
            cbar.ax.set_ylabel("Log Power (dB / Hz)", rotation=90, labelpad=20)
            # Revert font-size
            plt.rcParams.update({"font.size": old_fontsize})
            return fig

        fig, (ax0, ax1) = plt.subplots(
            nrows=2, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 2]}
        )
        plt.subplots_adjust(hspace=0.1)

        # Hypnogram (top axis)
        self.__plot_hypnogram(sf, hypno, ax0)
        # Spectrogram (bottom axis)
        self.__plot_spectrogram(data, sf, win_sec, fmin, fmax, trimperc, cmap, ax1)
        # Revert font-size
        plt.rcParams.update({"font.size": old_fontsize})
        return fig

    @staticmethod
    def __plot_hypnogram(sf, hypno, ax0):
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
        ax0.set_ylabel("Stage")
        ax0.xaxis.set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.spines["top"].set_visible(False)
        return ax0

    @staticmethod
    def __plot_spectrogram(data, sf, win_sec, fmin, fmax, trimperc, cmap, ax):
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

    def detect(
        self,
        picks: str | Iterable[str] = ("eeg"),
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
        """Wrapper around YASA's `spindles_detect <https://raphaelvallat.com/
        yasa/build/html/generated/yasa.spindles_detect.html>`_.
        """
        from yasa import spindles_detect

        self.results = spindles_detect(
            data=self.mne_raw.copy().load_data().set_eeg_reference().pick(picks),
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

    def detect(
        self,
        picks: str | Iterable[str] = ("eeg"),
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
        """Wrapper around YASA's `sw_detect <https://raphaelvallat.com/yasa/
        build/html/generated/yasa.sw_detect.html>`_.
        """
        from yasa import sw_detect

        self.results = sw_detect(
            data=self.mne_raw.copy().load_data().set_eeg_reference().pick(picks),
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
class REMsPipe(BaseEventPipe):
    """Rapid eye movements detection."""

    def detect(
        self,
        loc_chname: str = "E46",
        roc_chname: str = "E238",
        include: int | Iterable[int] = 4,
        freq_rem: Iterable[float] = (0.5, 5),
        duration: Iterable[float] = (0.3, 1.2),
        amplitude: Iterable[float] = (50, 325),
        remove_outliers: bool = False,
        save: bool = False,
    ):
        """Wrapper around YASA's `rem_detect <https://raphaelvallat.com/yasa/
        build/html/generated/yasa.rem_detect.html>`_.
        """
        from yasa import rem_detect

        referenced = self.mne_raw.copy().load_data().set_eeg_reference()
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

    def plot_average(self, save=False, yasa_args=None):
        yasa_args = yasa_args or dict()
        self.results.plot_average(**yasa_args)
        if save:
            self._save_avg_fig()

    def plot_topomap(self):
        raise AttributeError("'REMsPipe' object has no attribute 'plot_topomap'")

    def plot_topomap_collage(self):
        raise AttributeError("'REMsPipe' object has no attribute 'plot_topomap'")


@define(kw_only=True)
class CombinedPipe(BaseSpectrum):
    """The pipeline element combining results from multiple subjects.

    Contains methods for computing and plotting combined PSD,
    spectrogram and topomaps, per sleep stage.
    """

    pipes: Iterable[BaseHypnoPipe] = field()
    """Stores pipeline elements for multiple subjects.
    """

    mne_raw: mne.io.Raw = field(init=False)

    @mne_raw.default
    def _get_mne_raw_from_pipe(self):
        return self.pipes[0].mne_raw

    def _compute_psd_per_stage(self, picks, sleep_stages, sec_per_seg, avg_ref, dB):
        psd_per_stage = {}
        psds = []
        for pipe in self.pipes:
            psds.append(
                pipe._compute_psd_per_stage(
                    picks=picks,
                    sleep_stages=sleep_stages,
                    sec_per_seg=sec_per_seg,
                    avg_ref=avg_ref,
                    dB=dB,
                )
            )

        for stage in sleep_stages:
            psd_per_stage[stage] = [
                psds[0][stage][0],
                np.sum([psd[stage][1] for psd in psds], axis=0) / len(self.pipes),
                round(sum([psd[stage][2] for psd in psds]) / len(self.pipes), 2),
            ]
        return psd_per_stage
