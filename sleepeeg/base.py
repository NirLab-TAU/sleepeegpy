import errno
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Type, TypeVar

import matplotlib.pyplot as plt
import mne
import numpy as np
from attrs import define, field
from loguru import logger
from tqdm import tqdm

from .utils import logger_wraps

# For type annotation of pipe elements.
BasePipeType = TypeVar("BasePipeType", bound="BasePipe")


@define(kw_only=True, slots=False)
class BasePipe(ABC):
    """A base class for all per-subject pipeline segments."""

    prec_pipe: Type[BasePipeType] = field(default=None)
    """Preceding pipe that hands over mne_raw object and output_dir."""

    path_to_eeg: Path = field(converter=Path)
    """Can be any eeg file type supported by :py:func:`mne:mne.io.read_raw`.
    """

    @path_to_eeg.default
    def _set_path_to_eeg(self):
        if self.prec_pipe:
            return self.prec_pipe.path_to_eeg
        raise TypeError("Provide either 'pipe' or 'path_to_eeg' arguments")

    @path_to_eeg.validator
    def _validate_path_to_eeg(self, attr, value):
        if not value.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), value)

    output_dir: Path = field(converter=Path)
    """Path to the directory where the output will be saved."""

    @output_dir.default
    def _set_output_dir(self):
        if self.prec_pipe:
            return self.prec_pipe.output_dir
        raise TypeError("missing 1 required keyword-only argument: 'output_dir'")

    @output_dir.validator
    def _validate_output_dir(self, attr, value):
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / self.__class__.__name__).mkdir(exist_ok=True)
        # Duplicate logging into file.
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(self.output_dir / "pipeline.log", level="TRACE")

    mne_raw: mne.io.Raw = field(init=False)
    """An instanse of :py:class:`mne:mne.io.Raw`.
    """

    @mne_raw.default
    def _read_mne_raw(self):
        if self.prec_pipe:
            return self.prec_pipe.mne_raw
        return mne.io.read_raw(self.path_to_eeg)

    @property
    def sf(self):
        """A wrapper for :py:class:`raw.info["sfreq"] <mne:mne.Info>`.

        Returns:
            float: sampling frequency
        """
        return self.mne_raw.info["sfreq"]

    @property
    def bad_data_percent(self):
        """Calculates percent of data segments annotated as BAD.

        Returns:
            float: percent of bad data spans in raw data
        """
        df = self.mne_raw.annotations.to_data_frame()
        return round(
            100
            * (
                df[df.description.str.contains("bad", case=False)].duration.sum()
                * self.sf
            )
            / self.mne_raw.n_times,
            2,
        )

    def _savefig(self, fname, fig=None, **kwargs):
        if fig is None:
            plt.savefig(self.output_dir / self.__class__.__name__ / fname, **kwargs)
        else:
            fig.savefig(self.output_dir / self.__class__.__name__ / fname, **kwargs)

    def plot(
        self,
        save_annotations: bool = False,
        save_bad_channels: bool = False,
        overwrite: bool = False,
        **kwargs,
    ):
        """A wrapper for :py:meth:`mne:mne.io.Raw.plot`.

        Args:
            save_annotations: Whether to save annotations as txt. Defaults to False.
            save_bad_channels: Whether to save bad channels as txt. Defaults to False.
            overwrite: Whether to overwrite annotations and bad_channels files if exist.
                Defaults to False.
            **kwargs: Arguments passed to :py:meth:`mne:mne.io.Raw.plot`.
        """
        kwargs.setdefault("theme", "dark")
        kwargs.setdefault("bad_color", "r")
        if save_annotations or save_bad_channels:
            kwargs["block"] = True

        self.mne_raw.plot(**kwargs)

        if save_annotations:
            self.save_annotations(overwrite=overwrite)

        if save_bad_channels:
            self.save_bad_channels(overwrite=overwrite)

    def save_bad_channels(self, overwrite=False):
        """Adds bad channels from info["bads"] to the "bad_channels.txt" file.

        Args:
            overwrite: Whether to overwrite the file if exists.
                If False will add unique new channels to the file.
                Defaults to False.
        """
        new_bads = self.mne_raw.info["bads"]

        if new_bads:
            from natsort import natsorted

            fpath = self.output_dir / self.__class__.__name__ / "bad_channels.txt"
            old_bads = []
            if fpath.exists():
                with open(fpath, "r") as f:
                    old_bads = f.read().split()

            with open(fpath, "w") as f:
                bads = (
                    natsorted(new_bads)
                    if overwrite
                    else natsorted(set(old_bads + new_bads))
                )
                for bad in bads:
                    f.write(f"{bad}\n")

    def save_annotations(self, overwrite=False):
        """Writes annotations to "annotations.txt" file.

        Args:
            overwrite: Whether to overwrite the file if exists.
                If False and the file exists will throw an exception.
                Defaults to False.
        """
        self.mne_raw.annotations.save(
            self.output_dir / self.__class__.__name__ / "annotations.txt",
            overwrite=overwrite,
        )

    def plot_sensors(
        self, legend: Iterable[str] = None, legend_args: dict = None, **kwargs
    ):
        """A wrapper for :py:func:`mne:mne.viz.plot_sensors` with a legend.

        Args:
            legend: ch_groups names to connect to colors. Defaults to None.
            legend_args: Arguments passed to :py:meth:`mpl:matplotlib.axes.Axes.legend`.
                Defaults to None.
            **kwargs: Arguments passed to :py:func:`mne:mne.viz.plot_sensors`.
        """
        import matplotlib.patches as mpatches

        legend_args = legend_args or dict()

        ch_groups = kwargs.pop("ch_groups", None)
        axes = kwargs.pop("axes", None)
        kwargs.setdefault("show", False)

        fig = mne.viz.plot_sensors(
            self.mne_raw.info, ch_groups=ch_groups, axes=axes, **kwargs
        )
        if axes is None:
            axes = fig.axes[0]
        if legend:
            if not len(legend) == len(ch_groups):
                raise ValueError(
                    "Length of the legend and of the ch_groups should be equal"
                )

            patches = []
            colors = np.linspace(0, 1, len(ch_groups))
            color_vals = [plt.cm.jet(colors[i]) for i in range(len(ch_groups))]
            for i, color in enumerate(color_vals):
                if legend[i]:
                    patches.append(mpatches.Patch(color=color, label=legend[i]))
            if self.mne_raw.info["bads"]:
                patches.append(mpatches.Patch(color="red", label="Bad"))
            axes.legend(handles=patches, **legend_args)
        return fig

    @logger_wraps()
    def save_raw(self, fname: str, **kwargs):
        """A wrapper for :py:meth:`mne:mne.io.Raw.save`.

        Args:
            fname: Filename for the fif file being saved.
            **kwargs: Arguments passed to :py:meth:`mne:mne.io.Raw.save`.
        """
        fif_folder = self.output_dir / self.__class__.__name__
        self.mne_raw.save(fif_folder / fname, **kwargs)

    @logger_wraps()
    def set_eeg_reference(self, ref_channels="average", projection=False, **kwargs):
        """A wrapper for :py:meth:`mne:mne.io.Raw.set_eeg_reference`.

        Args:
            ref_channels: :py:meth:`ref_channels <mne:mne.io.Raw.set_eeg_reference>`. Defaults to 'average'.
            projection: :py:meth:`projection <mne:mne.io.Raw.set_eeg_reference>`. Defaults to False.
            **kwargs: Additional arguments passed to :py:meth:`mne:mne.io.Raw.set_eeg_reference`.
        """
        if not projection:
            logger.info(f"{ref_channels} reference has been applied")
        self.mne_raw.load_data().set_eeg_reference(
            ref_channels=ref_channels, projection=projection, **kwargs
        )


@define(kw_only=True, slots=False)
class BaseHypnoPipe(BasePipe, ABC):
    """A base class for the sleep-stage-analysis pipeline segments."""

    path_to_hypno: Path = field(
        converter=lambda x: Path(x) if x else None, default=None
    )
    """Path to hypnogram. Must be text file with every 
    row being int representing sleep stage for the epoch.
    """

    @path_to_hypno.validator
    def _validate_path_to_hypno(self, attr, value):
        if value is not None and not value.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), value)

    hypno_freq: float = field(converter=float)
    """Sampling rate of the hypnogram in Hz.

    E.g., 1/30 means 1 sample per 30 secs epoch,
    250 means 1 sample per 1/250 sec epoch.
    """

    @hypno_freq.default
    def _get_hypno_freq(self):
        if self.prec_pipe and isinstance(self.prec_pipe, BaseHypnoPipe):
            return self.prec_pipe.hypno_freq
        return 1

    hypno: np.ndarray = field()
    """ Hypnogram with sampling frequency hypno_freq
    with int representing sleep stage.
    """

    @hypno.default
    def _import_hypno(self):
        if isinstance(self.prec_pipe, BaseHypnoPipe):
            return self.prec_pipe.hypno
        if self.path_to_hypno is None:
            return None
        return np.loadtxt(self.path_to_hypno)

    hypno_up: np.array = field(init=False)
    """ Hypnogram upsampled to the sampling frequency of the raw data.
    """

    @hypno_up.default
    def _set_hypno_up(self):
        return self.hypno

    def __attrs_post_init__(self):
        if self.hypno is not None:
            self._upsample_hypno()

    def _upsample_hypno(self):
        """Adapted from YASA.
        Upsamples the hypnogram to the data's sampling frequency.
        Crops or pads the hypnogram if needed."""
        repeats = self.sf / self.hypno_freq
        if self.hypno_freq > self.sf:
            raise ValueError(
                "Sampling frequency of hypnogram must be smaller than that of eeg signal."
            )
        if not repeats.is_integer():
            raise ValueError("sf_hypno / sf_data must be a whole number.")
        hypno_up = np.repeat(np.asarray(self.hypno), repeats)

        # Fit to data
        npts_hyp = hypno_up.size
        npts_data = max(self.mne_raw.times.shape)  # Support for 2D data
        npts_diff = abs(npts_data - npts_hyp)

        if npts_hyp < npts_data:
            # Hypnogram is shorter than data
            logger.warning(
                "Hypnogram is SHORTER than data by {} seconds. "
                "Padding hypnogram with last value to match data.size.",
                round(npts_diff / self.sf, 2),
            )
            hypno_up = np.pad(hypno_up, (0, npts_diff), mode="edge")
        elif npts_hyp > npts_data:
            logger.warning(
                "Hypnogram is LONGER than data by {} seconds. "
                "Cropping hypnogram to match data.size.",
                round(npts_diff / self.sf, 2),
            )
            hypno_up = hypno_up[0:npts_data]
        self.hypno_up = hypno_up

    @logger_wraps()
    def predict_hypno(
        self,
        eeg_name: str = "E183",
        eog_name: str = "E252",
        emg_name: str = "E247",
        ref_name: str = "E26",
        save: bool = True,
    ):
        """Runs YASA's automatic sleep staging.

        Args:
            eeg_name: Preferentially a central electrode. Defaults to "E183".
            eog_name: Preferentially, the left LOC channel. Defaults to "E252".
            emg_name: Preferentially a chin electrode. Defaults to "E247".
            ref_name: Reference channel, preferentially a mastoid. Defaults to "E26".
            save: Whether to save the hypnogram to file. Defaults to True.
        """
        from yasa import SleepStaging, hypno_str_to_int

        sls = SleepStaging(
            self.mne_raw.copy().load_data().set_eeg_reference(ref_channels=[ref_name]),
            eeg_name=eeg_name,
            eog_name=eog_name,
            emg_name=emg_name,
        )
        hypno = sls.predict()

        # Set hypno attributes according to the predicted hypnogram.
        self.hypno = hypno_str_to_int(hypno)
        self.hypno_freq = 1 / 30
        self._upsample_hypno()
        sls.plot_predict_proba()
        if save:
            np.savetxt(
                self.output_dir / self.__class__.__name__ / "predicted_hypno.txt",
                self.hypno,
                fmt="%d",
            )
            self._savefig("predicted_hypno_probabilities.png")

    def sleep_stats(self, save: bool = False):
        """A wrapper for :py:func:`yasa:yasa.sleep_statistics`.

        Args:
            save: Whether to save the stats to csv. Defaults to False.
        """

        from csv import DictWriter

        from yasa import sleep_statistics

        if self.hypno is None:
            raise ValueError("There is no hypnogram to get stats from.")
        stats = sleep_statistics(self.hypno, self.hypno_freq)
        if save:
            with open(
                self.output_dir / self.__class__.__name__ / "sleep_stats.csv",
                "w",
                newline="",
            ) as csv_file:
                w = DictWriter(csv_file, stats.keys())
                w.writeheader()
                w.writerow(stats)
            return
        return stats


@define(kw_only=True, slots=False)
class BaseEventPipe(BaseHypnoPipe, ABC):
    """A base class for event detection."""

    results = field(init=False)
    """Event detection results as returned by YASA's event detection methods. 
    Depending on the child class can be instance of either 
    :py:class:`yasa:yasa.SpindlesResults`, :py:class:`yasa:yasa.SWResults` or :py:class:`yasa:yasa.REMResults` classes. 
    """

    tfrs: dict = field(init=False)
    """Instances of :py:class:`mne:mne.time_frequency.AverageTFR` per sleep stage.
    """

    @abstractmethod
    def detect():
        """Each event class should contain the detection method"""
        pass

    def _save_to_csv(self):
        self.results.summary().to_csv(
            self.output_dir
            / self.__class__.__name__
            / f"{self.__class__.__name__[:-4].lower()}.csv",
            index=False,
        )

    @logger_wraps()
    def plot_average(self, save: bool = False, **kwargs):
        """Plot average of the detected event.

        Args:
            save: Whether to save the figure to file. Defaults to False.
            **kwargs: Arguments passed to the YASA's plot_average().
        """
        self.results.plot_average(**kwargs)
        if save:
            self._savefig(f"{self.__class__.__name__[:-4].lower()}_avg.png")

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
        from more_itertools import collapse
        from natsort import natsort_keygen
        from seaborn import color_palette
        from .utils import plot_topomap

        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        subplots_args = subplots_args or dict()
        topomap_args.setdefault("cmap", color_palette("rocket_r", as_cmap=True))
        cbar_args.setdefault("label", prop)

        # Get detection results df and sort it naturally by channel name.
        grouped_summary = (
            self.results.summary(grp_chan=True, grp_stage=True, aggfunc=aggfunc)
            .sort_values("Channel", key=natsort_keygen())
            .reset_index()
        )
        assert np.isin(
            sleep_stages[stage], grouped_summary["Stage"].unique()
        ).all(), "No such stage in the detected events, was it included in the detect method?"

        if not axis:
            fig, axis = plt.subplots(**subplots_args)

        # Group by channel, average (median) per stage and sort channels again.
        per_stage = grouped_summary.loc[
            grouped_summary["Stage"].isin(collapse([sleep_stages[stage]]))
        ].groupby("Channel")
        per_stage = per_stage.mean() if aggfunc == "mean" else per_stage.median()
        per_stage = per_stage.sort_values("Channel", key=natsort_keygen()).reset_index()

        # Create info with montage containing only channels where events were detected.
        info = self.mne_raw.copy().pick(list(per_stage["Channel"].unique())).info

        topomap_args.setdefault("vlim", (per_stage[prop].min(), per_stage[prop].max()))
        plot_topomap(
            data=per_stage[prop],
            axis=axis,
            info=info,
            topomap_args=topomap_args,
            cbar_args=cbar_args,
        )
        if fig in locals():
            fig.suptitle(f"{stage} {self.__class__.__name__[:-4]} ({prop})")
            if save:
                self._savefig(
                    f"topomap_{self.__class__.__name__[:-4].lower()}_{prop.lower()}.png",
                    fig,
                )

    @logger_wraps()
    def plot_topomap_collage(
        self,
        props: Iterable[str],
        aggfunc: str = "mean",
        stages_to_plot: tuple = "all",
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        low_percentile: float = 5,
        high_percentile: float = 95,
        save: bool = False,
        topomap_args: dict = None,
        cbar_args: dict = None,
        figure_args: dict = None,
    ):
        """Plots topomap collage for multiple sleep stages and event properties.

        Args:
            props: Properties from the self.results.summary() to generate topomaps for.
            aggfunc: Averaging function, "mean" or "median". Defaults to "mean".
            stages_to_plot: stages_to_plot: Tuple of strings representing names from sleep_stages,
                e.g., ("REM", "N1"). If set to "all" plots every stage provided in sleep_stages.
                Defaults to "all".
            sleep_stages: Mapping between sleep stages names and their integer representations.
                Defaults to {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.
            low_percentile: Set min color value by percentile of the property data.
                Defaults to 5.
            high_percentile: Set max color value by percentile of the property data.
                Defaults to 95.
            save: Whether to save the figure. Defaults to False.
            topomap_args: Arguments passed to :py:func:`mne:mne.viz.plot_topomap`.
                Defaults to None.
            cbar_args: Arguments passed to :py:func:`mpl:matplotlib.pyplot.colorbar`.
                Defaults to None.
            figure_args: Arguments passed to :py:func:`mpl:matplotlib.pyplot.figure`.
                Defaults to None.

        """
        from more_itertools import collapse
        from natsort import natsort_keygen
        from .utils import plot_topomap

        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        figure_args = figure_args or dict()
        topomap_args.setdefault("cmap", "plasma")
        topomap_args.setdefault("vlim", [None, None])

        if stages_to_plot == "all":
            # Get all stages the events were detected for.
            stages_to_plot = {
                k: v
                for k, v in sleep_stages.items()
                if v in self.results.summary()["Stage"].unique()
            }
        n_rows = len(stages_to_plot)
        n_cols = len(props)

        figure_args.setdefault("figsize", (n_cols * 4, n_rows * 4))
        figure_args.setdefault("layout", "constrained")
        fig = plt.figure(**figure_args)

        subfigs = fig.subfigures(n_rows, 1)

        # Get detection results df and sort it naturally by channel name.
        grouped_summary = (
            self.results.summary(grp_chan=True, grp_stage=True, aggfunc=aggfunc)
            .sort_values("Channel", key=natsort_keygen())
            .reset_index()
        )

        if low_percentile:
            perc_high = dict()
        if high_percentile:
            perc_low = dict()

        data_per_stage_per_prop = np.empty(
            (len(stages_to_plot), len(props)), dtype=object
        )
        info = np.empty((len(stages_to_plot), len(props)), dtype=object)

        for col_index, prop in enumerate(props):
            for_perc = []
            for row_index, stage in enumerate(stages_to_plot):
                # Group by channel, average (median) per stage and sort channels again.
                per_stage = grouped_summary.loc[
                    grouped_summary["Stage"].isin(collapse([sleep_stages[stage]]))
                ].groupby("Channel")
                per_stage = (
                    per_stage.mean() if aggfunc == "mean" else per_stage.median()
                )
                per_stage = per_stage.sort_values(
                    "Channel", key=natsort_keygen()
                ).reset_index()

                # Create info with montage containing only channels where events were detected.
                info[row_index, col_index] = (
                    self.mne_raw.copy().pick(list(per_stage["Channel"].unique())).info
                )
                data = per_stage[prop]
                for_perc.append(data.to_numpy())
                data_per_stage_per_prop[row_index, col_index] = data
            if low_percentile:
                perc_low[prop] = np.percentile(np.concatenate(for_perc), low_percentile)
            if high_percentile:
                perc_high[prop] = np.percentile(
                    np.concatenate(for_perc), high_percentile
                )

        for row_index, stage in enumerate(stages_to_plot):
            axes = subfigs[row_index].subplots(1, n_cols)

            for col_index, prop in enumerate(props):
                if low_percentile:
                    topomap_args["vlim"][0] = perc_low[prop]
                if high_percentile:
                    topomap_args["vlim"][1] = perc_high[prop]
                plot_topomap(
                    data=data_per_stage_per_prop[row_index, col_index],
                    axis=axes[col_index],
                    info=info[row_index, col_index],
                    topomap_args=topomap_args,
                    cbar_args=cbar_args,
                )
                axes[col_index].set_title(f"{prop}")
            n_spindles = int(
                self.results.summary(grp_chan=False, grp_stage=True).loc[
                    sleep_stages[stage]
                ]["Count"]
            )
            subfigs[row_index].suptitle(
                f"{stage}, n={n_spindles}",
                fontsize="xx-large",
            )

        fig.suptitle(f"{self.__class__.__name__[:-4]}", fontsize="xx-large")
        if save:
            self._savefig(
                f"topomap_{self.__class__.__name__[:-4].lower()}_collage.png", fig
            )

    @logger_wraps()
    def apply_tfr(
        self,
        freqs: Iterable[float],
        n_freqs: int,
        time_before: float,
        time_after: float,
        method: str = "morlet",
        save: bool = False,
        **tfr_kwargs,
    ):
        """Transforms the events signal to time-frequency representation.

        Args:
            freqs: Lower and upper bounds of frequencies of interest in Hz, e.g., (10,20).
            n_freqs: Frequency resolution in TFR.
            time_before: Seconds before the event peak to get from the real data.
            time_after: Seconds after the event peak to get from the real data
            method: TFR transform method. Defaults to "morlet".
            save: Whether to save the TFRs to file. Defaults to False.
            **tfr_kwargs: Arguments passed to :py:func:`mne:mne.time_frequency.tfr_array_morlet`
                or :py:func:`mne:mne.time_frequency.tfr_array_multitaper`.
        """
        if not self.results:
            raise AttributeError("Run the detect method first")
        if not (method == "morlet" or method == "multitaper"):
            raise ValueError("the 'method' argument should be 'morlet' or 'multitaper'")

        from natsort import natsorted

        sleep_stages = {
            -2: "Unscored",
            -1: "Art",
            0: "Wake",
            1: "N1",
            2: "N2",
            3: "N3",
            4: "REM",
        }

        tfr_kwargs.setdefault("n_jobs", -1)
        tfr_kwargs.setdefault("verbose", "error")
        tfr_kwargs["output"] = "avg_power"

        freqs = np.linspace(freqs[0], freqs[1], n_freqs)
        df_raw = self.results.get_sync_events(
            time_before=time_before, time_after=time_after
        )[["Event", "Amplitude", "Channel", "Stage"]]

        self.tfrs = {}

        for stage in df_raw["Stage"].unique():
            df = df_raw[df_raw["Stage"] == stage]
            # Group amplitudes by channel and events
            df = df.groupby(["Channel", "Event"], group_keys=True).apply(
                lambda x: x["Amplitude"]
            )

            # Create dict for every channel and values with array of shape (n_events, 1, n_event_times)
            for_tfrs = {
                channel: np.expand_dims(
                    np.array(
                        events_df.groupby(level=1)
                        .apply(lambda x: x.to_numpy())
                        .tolist()
                    ),
                    axis=1,
                )
                for channel, events_df in df.groupby(level=0)
            }

            # Calculate tfrs
            if method == "morlet":
                tfrs = {
                    channel: mne.time_frequency.tfr_array_morlet(
                        v,
                        self.sf,
                        freqs,
                        **tfr_kwargs,
                    )
                    for channel, v in tqdm(for_tfrs.items())
                }
            elif method == "multitaper":
                tfrs = {
                    channel: mne.time_frequency.tfr_array_multitaper(
                        v,
                        self.sf,
                        freqs,
                        **tfr_kwargs,
                    )
                    for channel, v in tqdm(for_tfrs.items())
                }
            # Sort and combine
            data = np.squeeze(
                np.array([tfr for channel, tfr in natsorted(tfrs.items())])
            )

            # Matrix should be 3D, so if there was only one channel - need to expand.
            if data.ndim == 2:
                data = np.expand_dims(data, axis=0)
            self.tfrs[sleep_stages[stage]] = mne.time_frequency.AverageTFR(
                info=self.mne_raw.copy().pick(list(tfrs.keys())).info,
                data=data,
                times=np.linspace(
                    -time_before,
                    time_after,
                    int((time_before + time_after) * self.sf + 1),
                ),
                freqs=freqs,
                nave=np.mean([arr.shape[0] for arr in for_tfrs.values()], dtype=int),
                method=method,
            )
        if save:
            for stage, tfr in self.tfrs.items():
                tfr.save(
                    self.output_dir
                    / self.__class__.__name__
                    / f"{self.__class__.__name__[:-4].lower()}_{stage}-tfr.h5"
                )

    @logger_wraps()
    def read_tfrs(self, dirpath: str | None = None):
        """Loads TFRs stored in hdf5 files.

        Filenames should end with {type_of_event}_{sleep_stage}-tfr.h5

        Args:
            dirpath: Path to the directory containing hdf5 files. Defaults to None.
        """
        import re

        r = f"{self.__class__.__name__[:-4].lower()}_(.+)(?:-tfr.h5)"
        self.tfrs = dict()
        dirpath = (
            Path(dirpath) if dirpath else self.output_dir / self.__class__.__name__
        )
        for p in dirpath.glob("*tfr.h5"):
            m = re.search(r, str(p))
            if m:
                self.tfrs[m.groups()[0]] = mne.time_frequency.read_tfrs(p)[0]


@define(kw_only=True, slots=False)
class SpectrumPlots(ABC):
    """Plotting spectral data."""

    psds: dict = field(init=False, factory=dict)
    """Instances of :class:`.SleepSpectrum` per sleep stage.
    """

    @logger_wraps()
    def plot_psds(
        self,
        picks: Iterable[str] | str,
        psd_range: tuple = (-40, 60),
        freq_range: tuple = (0, 60),
        dB=True,
        xscale: str = "linear",
        axis: plt.axis = None,
        plot_sensors: bool = False,
        save: bool = False,
        legend_args: dict = None,
        **subplots_kw,
    ):
        """Plot PSD per sleep stage.

        Args:
            picks: Channels to plot PSDs for. Refer to :py:meth:`mne:mne.io.Raw.pick`.
            psd_range: Range of y axis on PSD plot. Defaults to (-40, 60).
            freq_range: Range of x axis on PSD plot. Defaults to (0, 40).
            dB: Whether transform PSD to dB. Defaults to True.
            xscale: Scale of the X axis, check available values at
                :py:meth:`mpl:matplotlib.axes.Axes.set_xscale`. Defaults to "linear".
            axis: Instance of :py:class:`mpl:matplotlib.axes.Axes`.
                Defaults to None.
            plot_sensors: Whether to plot sensor map showing which channels were used for
                computing PSD. Defaults to False.
            save: Whether to save the figure. Defaults to False.
            **subplots_kw: Arguments passed to the :py:func:`mpl:matplotlib.pyplot.subplots`.
                Have no effect if axis is provided.Defaults to None.
        """
        from mne.io.pick import _picks_to_idx

        legend_args = legend_args or dict()

        is_new_axis = False

        if not axis:
            fig, axis = plt.subplots(**subplots_kw)
            is_new_axis = True

        for stage, spectrum in self.psds.items():
            psds = np.mean(
                spectrum._data[_picks_to_idx(spectrum.info, picks=picks), :], axis=0
            )
            psds = 10 * np.log10(10**12 * psds) if dB else 10**12 * psds
            axis.plot(
                spectrum._freqs,
                psds,
                label=f"{stage} ({spectrum.info['description']}%)",
            )

        axis.set_xlim(freq_range)
        axis.set_ylim(psd_range)
        axis.set_xscale(xscale)
        units = r"$\mu V^{2}/Hz$ (dB)" if dB else r"$\mu V^{2}/Hz$"
        axis.set_ylabel(f"Power ({units})")
        xlabel = (
            "Frequency (Hz)"
            if xscale == "linear"
            else f"{xscale} frequency [Hz]".capitalize()
        )
        axis.set_xlabel(xlabel)
        axis.legend(**legend_args)

        if plot_sensors:
            from ast import literal_eval

            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            # color = "cyan"
            axins = inset_axes(axis, width="30%", height="30%", loc="lower left")
            psd_channels = _picks_to_idx(self.mne_raw.info, picks)

            try:
                interpolated = literal_eval(self.mne_raw.info["description"])
            except:
                interpolated = None

            interpolated = (
                _picks_to_idx(self.mne_raw.info, interpolated) if interpolated else None
            )

            ch_groups = {
                k: v
                for k, v in {"PSD": psd_channels, "Interpolated": interpolated}.items()
                if v is not None
            }
            self.plot_sensors(
                legend=list(ch_groups.keys()),
                axes=axins,
                legend_args=dict(
                    loc="lower left", bbox_to_anchor=(1, 0), fontsize="x-small"
                ),
                ch_groups=list(ch_groups.values()),
                pointsize=7,
                linewidth=0.7,
            )

        # Save the figure if 'save' set to True and no axis has been passed.
        if save and is_new_axis:
            self._savefig(f"psd.png", fig)

    @logger_wraps()
    def plot_topomap(
        self,
        stage: str = "REM",
        band: dict = {"Delta": (0, 4)},
        dB: bool = False,
        axis: plt.axis = None,
        save: bool = False,
        topomap_args: dict = None,
        cbar_args: dict = None,
        subplots_args: dict = None,
    ):
        """Plots topomap for a sleep stage and a frequency band.

        Args:
            stage: One of the sleep_stages keys. Defaults to "REM".
            band: Name-value pair - with name=arbitrary name
                and value=(l_freq, h_freq).
                Defaults to {"Delta": (0, 4)}.
            dB: Whether transform PSD to dB. Defaults to False.
            axis: Instance of :py:class:`mpl:matplotlib.axes.Axes`.
                Defaults to None.
            save: Whether to save the figure. Defaults to False.
            topomap_args: Arguments passed to :py:func:`mne:mne.viz.plot_topomap`.Defaults to None.
            cbar_args: Arguments passed to :py:func:`mpl:matplotlib.pyplot.colorbar`.Defaults to None.
            subplots_args: Arguments passed to :py:func:`mpl:matplotlib.pyplot.subplots`.Defaults to None.
        """
        from .utils import plot_topomap

        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        subplots_args = subplots_args or dict()
        topomap_args.setdefault("cmap", "plasma")
        cbar_args.setdefault(
            "label",
            r"$\mu V^{2}/Hz$ (dB)" if dB else r"$\mu V^{2}/Hz$",
        )
        assert stage in self.psds, f"{stage} is not in self.psds"

        is_new_axis = False

        if axis is None:
            fig, axis = plt.subplots(**subplots_args)
            is_new_axis = True

        [(_, b)] = band.items()

        psds = (
            10 * np.log10(10**12 * self.psds[stage]._data)
            if dB
            else 10**12 * self.psds[stage]._data
        )
        psds = np.take(
            psds,
            np.where(
                np.logical_and(
                    self.psds[stage]._freqs >= b[0],
                    self.psds[stage]._freqs <= b[1],
                )
            )[0],
            axis=1,
        ).sum(axis=1)

        plot_topomap(
            data=psds,
            axis=axis,
            info=self.psds[stage].info,
            topomap_args=topomap_args,
            cbar_args=cbar_args,
        )

        if is_new_axis:
            fig.suptitle(f"{stage} ({b[0]}-{b[1]} Hz)")
        if save and is_new_axis:
            self._savefig(f"topomap_psd_{list(band)[0]}.png", fig)

    @logger_wraps()
    def plot_topomap_collage(
        self,
        stages_to_plot: tuple = "all",
        bands: dict = {
            "Delta": (0, 3.99),
            "Theta": (4, 7.99),
            "Alpha": (8, 12.49),
            "SMR": (12.5, 15),
            "Beta": (12.5, 29.99),
            "Gamma": (30, 60),
        },
        dB: bool = False,
        low_percentile: float = 5,
        high_percentile: float = 95,
        save: bool = False,
        topomap_args: dict = None,
        cbar_args: dict = None,
        figure_args: dict = None,
    ):
        """Plots topomap collage for multiple sleep stages and bands.

        Args:
            stages_to_plot: Tuple of strings representing names from sleep_stages,
                e.g., ("REM", "N1").
                If set to "all" plots every stage provided in sleep_stages.
                Defaults to "all".
            bands: Dict of name-value pairs - with name=arbitrary name
                and value=(l_freq, h_freq).
                Defaults to { "Delta": (0, 3.99), "Theta": (4, 7.99), "Alpha": (8, 12.49),
                "SMR": (12.5, 15), "Beta": (12.5, 29.99), "Gamma": (30, 60), }.
            dB: Whether transform PSD to dB. Defaults to False.
            sleep_stages: Mapping between sleep stages names and their integer representations.
                Defaults to {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.
            low_percentile: Set min color value by percentile of the band data.
                Defaults to 5.
            high_percentile: Set max color value by percentile of the band data.
                Defaults to 95.
            save: Whether to save the figure. Defaults to False.
            topomap_args: Arguments passed to :py:func:`mne:mne.viz.plot_topomap`.Defaults to None.
            cbar_args: Arguments passed to :py:func:`mpl:matplotlib.pyplot.colorbar`.Defaults to None.
            figure_args: Arguments passed to :py:func:`mpl:matplotlib.pyplot.figure`.Defaults to None.
        """
        from .utils import plot_topomap

        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        figure_args = figure_args or dict()
        topomap_args.setdefault("cmap", "plasma")
        topomap_args.setdefault("vlim", [None, None])
        cbar_args.setdefault(
            "label",
            r"$\mu V^{2}/Hz$ (dB)" if dB else r"$\mu V^{2}/Hz$",
        )

        if stages_to_plot == "all":
            stages_to_plot = self.psds.keys()
        n_rows = len(stages_to_plot)
        n_cols = len(bands)

        figure_args.setdefault("figsize", (n_cols * 4, n_rows * 4))
        figure_args.setdefault("layout", "constrained")
        fig = plt.figure(**figure_args)
        subfigs = fig.subfigures(n_rows, 1)

        if low_percentile:
            perc_low = dict()

        if high_percentile:
            perc_high = dict()

        psds_per_stage_per_band = np.empty(
            (len(stages_to_plot), len(bands)), dtype=object
        )
        for col_index, (band_key, b) in enumerate(bands.items()):
            for_perc = []
            for row_index, stage in enumerate(self.psds):
                psds = (
                    10 * np.log10(10**12 * self.psds[stage]._data)
                    if dB
                    else 10**12 * self.psds[stage]._data
                )
                psds = np.take(
                    psds,
                    np.where(
                        np.logical_and(
                            self.psds[stage]._freqs >= b[0],
                            self.psds[stage]._freqs <= b[1],
                        )
                    )[0],
                    axis=1,
                ).sum(axis=1)
                for_perc.append(psds)
                psds_per_stage_per_band[row_index, col_index] = psds
            if low_percentile:
                perc_low[band_key] = np.percentile(for_perc, low_percentile)
            if high_percentile:
                perc_high[band_key] = np.percentile(for_perc, high_percentile)

        for row_index, stage in enumerate(stages_to_plot):
            axes = subfigs[row_index].subplots(1, n_cols)

            for col_index, band_key in enumerate(bands):
                if low_percentile:
                    topomap_args["vlim"][0] = perc_low[band_key]
                if high_percentile:
                    topomap_args["vlim"][1] = perc_high[band_key]

                plot_topomap(
                    data=psds_per_stage_per_band[row_index, col_index],
                    axis=axes[col_index],
                    info=self.psds[stage].info,
                    topomap_args=topomap_args,
                    cbar_args=cbar_args,
                )
                axes[col_index].set_title(
                    f"{band_key} ({bands[band_key][0]}-{bands[band_key][1]} Hz)"
                )

            subfigs[row_index].suptitle(
                f"{stage} ({self.psds[stage].info['description']}%)",
                fontsize="xx-large",
            )

        if save:
            self._savefig(f"topomap_psd_collage.png", fig)

    @logger_wraps()
    def save_psds(self, overwrite):
        import re

        for stage, spectrum in self.psds.items():
            stage = re.sub(r"[^\w\s-]", "_", stage)
            spectrum.save(
                self.output_dir / self.__class__.__name__ / f"{stage}-psd.h5",
                overwrite=overwrite,
            )


@define(kw_only=True, slots=False)
class SleepSpectrum(mne.time_frequency.spectrum.BaseSpectrum):
    """Spectral representation of sleep stage data.

    Adapted from `MNE <https://mne.tools/stable/index.html>`_.
    """

    def __init__(
        self,
        inst,
        hypno,
        stage_idx,
        method,
        fmin,
        fmax,
        tmin,
        tmax,
        picks,
        proj,
        reject_by_annotation,
        *,
        n_jobs,
        verbose=None,
        **method_kw,
    ):
        # triage reading from file
        if isinstance(inst, dict):
            self.__setstate__(inst)
            return
        # Get uniform segment length, otherwise avging won't be possible.
        multitaper_segment_len = method_kw.pop("multitaper_segment_len", 2000)
        # do the basic setup
        super().__init__(
            inst,
            method,
            fmin,
            fmax,
            tmin,
            tmax,
            picks,
            proj,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )
        data = self.inst.get_data(
            self._picks,
            reject_by_annotation="NaN" if reject_by_annotation else None,
        )
        # compute the spectra
        self._compute_spectra(
            method,
            data,
            hypno,
            stage_idx,
            fmin,
            fmax,
            n_jobs,
            verbose,
            multitaper_segment_len=multitaper_segment_len,
        )
        # check for correct shape and bad values
        self._check_values()
        del self._shape
        # save memory
        del self.inst

    def __add__(self, other):
        if isinstance(other, self.__class__):
            from copy import deepcopy

            spectrum_cp = deepcopy(self)
            spectrum_cp._data = np.add(self._data, other._data)
            try:
                stage_percent = float(self.info["description"])
                other_stage_percent = float(other.info["description"])
                spectrum_cp.info["description"] = str(
                    stage_percent + other_stage_percent
                )
            except:
                pass
            return spectrum_cp
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            from copy import deepcopy

            spectrum_cp = deepcopy(self)
            spectrum_cp._data = self._data / other
            try:
                spectrum_cp.info["description"] = str(
                    round(float(self.info["description"]) / other, 2)
                )
            except:
                pass
            return spectrum_cp
        else:
            return NotImplemented

    def _compute_spectra(
        self,
        method,
        data,
        hypno,
        stage_idx,
        fmin,
        fmax,
        n_jobs,
        verbose,
        multitaper_segment_len,
    ):
        """
        Weighted average for Welch's PSD.
        Average of uniform sample regions for Multitaper's PSD.
        """
        from more_itertools import collapse
        from scipy import ndimage

        n_samples_total = np.count_nonzero(~np.isnan(data), axis=1)[0]
        psds_list = []
        weights = []
        n_samples = 0
        try:
            regions = collapse(
                ndimage.find_objects(
                    ndimage.label(
                        np.logical_or.reduce([hypno == i for i in stage_idx])
                    )[0]
                )
            )
        except TypeError:
            regions = collapse(
                ndimage.find_objects(ndimage.label(hypno == stage_idx)[0])
            )

        if method == "multitaper":
            from more_itertools import flatten

            ranges = [
                list(range(region.start, region.stop, multitaper_segment_len))
                for region in regions
            ]
            slice_ranges = flatten([list(zip(r[:-1], r[1:])) for r in ranges])
            regions = [slice(z[0], z[1]) for z in slice_ranges]

        for region in regions:
            n_samples_per_region = np.count_nonzero(~np.isnan(data[:, region]), axis=1)[
                0
            ]

            # make the spectra
            psds, freqs = self._psd_func(
                data[:, region],
                self.sfreq,
                fmin=fmin,
                fmax=fmax,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            psds_list.append(psds)
            weights.append(n_samples_per_region)
            n_samples += n_samples_per_region

        # avg_psds = np.average(np.array(psds_list), weights=weights, axis=0)
        masked_data = np.ma.masked_array(
            np.array(psds_list), np.isnan(np.array(psds_list))
        )
        average = np.ma.average(masked_data, weights=weights, axis=0)
        avg_psds = average.filled(np.nan)

        self._data = avg_psds
        self._freqs = freqs
        self.info["description"] = str(round(n_samples / n_samples_total * 100, 2))
        # this is *expected* shape, it gets asserted later in _check_values()
        # (and then deleted afterwards)
        self._shape = (len(self.ch_names), len(self.freqs))
        # we don't need these anymore, and they make save/load harder
        del self._picks
        del self._psd_func
        del self._time_mask
