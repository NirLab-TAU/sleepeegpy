import os
import errno
import inspect

import yaml
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from functools import wraps

from typing import TypeVar, Type
from attrs import define, field

import matplotlib.pyplot as plt
import numpy as np
import mne
from tqdm import tqdm

# For type annotation of pipe elements.
BasePipeType = TypeVar("BasePipeType", bound="BasePipe")


@define(kw_only=True)
class Log:
    PIPES = {
        "CleaningPipe": "cleaning",
        "ICAPipe": "ica",
        "SpectralPipe": "spectral",
        "SpindlesPipe": "spindles",
        "SlowWavesPipe": "slow_waves",
        "REMsPipe": "rems",
        "Dashboard": "dashboard",
    }
    output_dir = field(converter=Path)
    cleaning: dict = field(init=False, factory=dict)
    ica: dict = field(init=False, factory=dict)
    spectral: dict = field(init=False, factory=dict)
    spindles: dict = field(init=False, factory=dict)
    slow_waves: dict = field(init=False, factory=dict)
    rems: dict = field(init=False, factory=dict)
    dashboard: dict = field(init=False, factory=dict)

    def __attrs_post_init__(self):
        self._load()

    def __str__(self):
        from yaml.representer import Representer

        yaml.add_representer(tuple, Representer.represent_list)
        return yaml.dump(self.data, default_flow_style=False)

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    @property
    def data(self):
        return {value: self.__getattribute__(value) for value in self.PIPES.values()}

    def _update(self, key, value):
        the_class = inspect.stack()[1][0].f_locals["self"].__class__.__name__
        x = self.__getattribute__(self.PIPES[the_class])
        x.setdefault(key, []).append(value)
        self.__setattr__(self.PIPES[the_class], x)

    def update(func):
        @wraps(func)
        def wrapper_func(self, *args, **kwargs):
            # Do something before the function.
            func(self, *args, **kwargs)
            # Do something after the function.

            # args_values = args_dict.values()
            all_args = {"args": list(args)} if args else dict()
            all_args |= kwargs
            self.log._update("func: " + func.__name__, all_args)
            self.log.dump()

        return wrapper_func

    def _load(self):
        if "log.json" in [f.name for f in self.output_dir.iterdir() if f.is_file()]:
            with open(self.output_dir / "log.json") as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    self.__setattr__(key, value)

    def dump(self):
        with open(self.output_dir / "log.json", "w") as fp:
            json.dump(
                self.data,
                fp,
                indent=4,
            )


@define(kw_only=True, slots=False)
class BasePipe(ABC):
    """A template class for the pipeline segments."""

    prec_pipe: Type[BasePipeType] = field(default=None, metadata={"my_metadata": 1})
    """Preceding pipe that hands over mne_raw attr."""

    path_to_eeg: Path = field(converter=Path)
    """Can be any file type supported by 
    `mne.io.read_raw() <https://mne.tools/stable/generated/
    mne.io.Raw.html#mne.io.Raw.save>`_.
    """

    @path_to_eeg.default
    def _set_path_to_eeg(self):
        if self.prec_pipe:
            return "/"
        raise TypeError('Provide either "pipe" or "path_to_eeg" arguments')

    @path_to_eeg.validator
    def _validate_path_to_eeg(self, attr, value):
        if not value.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), value)

    output_dir: Path = field(converter=Path)
    """Path to the directory where the output will be saved."""

    @output_dir.default
    def _set_output_dir(self):
        return (
            self.prec_pipe.output_dir if self.prec_pipe else self.path_to_eeg.parents[0]
        )

    @output_dir.validator
    def _validate_output_dir(self, attr, value):
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / self.__class__.__name__).mkdir(exist_ok=True)

    mne_raw: mne.io.Raw = field(init=False)
    """An instanse of  
    `mne.io.Raw <https://mne.tools/stable/generated/
    mne.io.Raw.html#mne-io-raw>`_.
    """

    @mne_raw.default
    def _read_mne_raw(self):
        from mne.io import read_raw

        try:
            return (
                self.prec_pipe.mne_raw if self.prec_pipe else read_raw(self.path_to_eeg)
            )
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    log: Log = field(init=False)

    @log.default
    def _initialize_log(self):
        log = Log(output_dir=self.output_dir)
        log._update("eeg_file", self.path_to_eeg.resolve().as_posix())
        log._update("output_dir", self.output_dir.resolve().as_posix())
        return log

    @property
    def sf(self):
        """A wrapper for
        `mne.Info["sfreq"] <https://mne.tools/stable/generated/
        mne.Info.html#mne-info>`_.

        Returns:
            float: sampling frequency
        """
        return self.mne_raw.info["sfreq"]

    def plot(
        self,
        save_annotations: bool = False,
        save_bad_channels: bool = False,
        overwrite: bool = False,
        **kwargs,
    ):
        """A wrapper for `mne.io.Raw.plot() <https://mne.tools/stable/
        generated/mne.io.Raw.html#mne.io.Raw.plot>`_.

        Args:
            butterfly: Whether to start in butterfly mode. Defaults to False.
            save_annotations: Whether to save annotations as txt. Defaults to False.
            save_bad_channels: Whether to save bad channels as txt. Defaults to False.
            scalings: Scale for amplitude per channel type. Defaults to 'auto'.
            use_opengl: Whether to use OpenGL acceleration. Defaults to None.
            overwrite: Whether to overwrite annotations and bad_channels files if exist.
                Defaults to False.
            **kwargs: Arguments passed to `raw.plot() <https://mne.tools/stable/
                generated/mne.io.Raw.html#mne.io.Raw.plot>`_.
        """
        kwargs.setdefault("theme", "dark")
        kwargs.setdefault("bad_color", "r")
        kwargs.setdefault("scalings", "auto")
        self.mne_raw.plot(**kwargs)

        if save_annotations:
            self.mne_raw.annotations.save(
                self.output_dir / self.__class__.__name__ / "annotations.txt",
                overwrite=overwrite,
            )
        if save_bad_channels:
            if overwrite:
                with open(
                    self.output_dir / self.__class__.__name__ / "bad_channels.txt", "w"
                ) as f:
                    for bad in self.mne_raw.info["bads"]:
                        f.write(f"{bad}\n")
            else:
                with open(
                    self.output_dir / self.__class__.__name__ / "bad_channels.txt", "a+"
                ) as f:
                    f.seek(0)
                    bads = f.read().split()
                    for bad in self.mne_raw.info["bads"]:
                        if bad not in bads:
                            f.write(f"{bad}\n")

    def plot_sensors(self, legend=None, legend_args=None, **kwargs):
        import matplotlib.patches as mpatches

        legend_args = legend_args or dict()

        ch_groups = kwargs.pop("ch_groups", None)
        axes = kwargs.pop("axes", None)
        kwargs.setdefault("show", False)

        fig = mne.viz.plot_sensors(
            self.mne_raw.info, ch_groups=ch_groups, axes=axes, **kwargs
        )
        if legend is not None:
            if not len(legend) == len(ch_groups):
                raise ValueError(
                    "Length of the legend and of the ch_groups should be equal"
                )
            patches = []
            # if legend:
            colors = np.linspace(0, 1, len(ch_groups))
            color_vals = [plt.cm.jet(colors[i]) for i in range(len(ch_groups))]
            for i, color in enumerate(color_vals):
                patches.append(mpatches.Patch(color=color, label=legend[i]))
            if self.mne_raw.info["bads"]:
                patches.append(mpatches.Patch(color="red", label="Bad"))
            axes.legend(handles=patches, **legend_args)
        return fig

    @Log.update
    def save_raw(self, fname: str, **kwargs):
        """A wrapper for `mne.io.Raw.save <https://mne.tools/stable/
        generated/mne.io.Raw.html#mne.io.Raw.save>`_.

        Args:
            fname: Filename for the fif file being saved.
            **kwargs: Arguments passed to raw.save()
        """
        fif_folder = self.output_dir / self.__class__.__name__
        self.mne_raw.save(fif_folder / fname, **kwargs)

    @Log.update
    def set_eeg_reference(self, **kwargs):
        kwargs.setdefault("ref_channels", "average")
        kwargs.setdefault("projection", False)
        if not kwargs["projection"]:
            self.log._update("reference", kwargs["ref_channels"])
        self.mne_raw.load_data().set_eeg_reference(**kwargs)


@define(kw_only=True, slots=False)
class BaseHypnoPipe(BasePipe, ABC):
    """A template class for the sleep stage analysis pipeline segments."""

    path_to_hypno: Path = field(converter=Path)
    """Path to hypnogram. Must be text file with every 
    row being int representing sleep stage for the epoch.
    """

    @path_to_hypno.default
    def _set_path_to_hypno(self):
        return "/"

    @path_to_hypno.validator
    def _validate_path_to_hypno(self, attr, value):
        if not value.exists():
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
        if self.prec_pipe and isinstance(self.prec_pipe, BaseHypnoPipe):
            return self.prec_pipe.hypno
        if self.path_to_hypno == Path("/"):
            return np.empty(0)
        try:
            return np.loadtxt(self.path_to_hypno)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    hypno_up: np.array = field()
    """ Hypnogram upsampled to the sampling frequency of the raw data.
    """

    @hypno_up.default
    def _set_hypno_up(self):
        return self.hypno

    def __attrs_post_init__(self):
        if self.hypno.any():
            self.__upsample_hypno()
        self.log._update("hypno_file", self.path_to_hypno.resolve().as_posix())
        self.log._update("hypno_freq", self.hypno_freq)

    def __upsample_hypno(self):
        from yasa import hypno_upsample_to_data

        self.hypno_up = hypno_upsample_to_data(
            self.hypno, self.hypno_freq, self.mne_raw, verbose=False
        )

    @Log.update
    def predict_hypno(
        self,
        eeg_name: str = "E183",
        eog_name: str = "E252",
        emg_name: str = "E247",
        ref_name: str = "E26",
        save=True,
    ):
        """Runs YASA's automatic sleep staging

        Args:
            eeg_name: Preferentially a central electrode. Defaults to "E183".
            eog_name: Preferentially, the left LOC channel. Defaults to "E252".
            emg_name: Preferentially a chin electrode. Defaults to "E247".
            ref_name: Reference channel, preferentially a mastoid. Defaults to "E26".
            save: Whether to save the hypnogram. Defaults to True.
        """
        from yasa import SleepStaging, hypno_str_to_int

        sls = SleepStaging(
            self.mne_raw.copy().load_data().set_eeg_reference(ref_channels=[ref_name]),
            eeg_name=eeg_name,
            eog_name=eog_name,
            emg_name=emg_name,
        )
        hypno = sls.predict()
        self.hypno = hypno_str_to_int(hypno)
        self.hypno_freq = 1 / 30
        self.__upsample_hypno()
        sls.plot_predict_proba()
        if save:
            np.savetxt(
                self.output_dir / self.__class__.__name__ / "predicted_hypno.txt",
                self.hypno,
                fmt="%d",
            )
            plt.savefig(
                self.output_dir
                / self.__class__.__name__
                / "predicted_hypno_probabilities.png"
            )

    def sleep_stats(self, save: bool = False):
        """A wrapper for
        `yasa.sleep_statistics <https://raphaelvallat.com/yasa/build/html/generated/yasa.sleep_statistics.html#yasa-sleep-statistics>`_.

        Args:
            save: Whether to save the stats to csv. Defaults to False.
        """

        from yasa import sleep_statistics
        from csv import DictWriter

        assert self.hypno.any(), "There is no hypnogram to get stats from."
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
class BaseTopomap(ABC):
    def _plot_topomap(self, data, axis, info=None, topomap_args=None, cbar_args=None):
        from mne.viz import plot_topomap

        topomap_args = topomap_args or dict()
        topomap_args.setdefault("size", 5)
        topomap_args.setdefault("show", False)
        im, cn = plot_topomap(
            data,
            info if info else self.mne_raw.info,
            axes=axis,
            **topomap_args,
        )

        cbar_args = cbar_args or dict()
        cbar_args.setdefault("shrink", 0.6)
        cbar_args.setdefault("orientation", "vertical")
        plt.colorbar(
            im,
            ax=axis,
            **cbar_args,
        )


@define(kw_only=True, slots=False)
class BaseEventPipe(BaseHypnoPipe, BaseTopomap, ABC):
    """A template class for event detection."""

    results = field(init=False)
    """Event detection results as returned by YASA's event detection methods. 
    Depending on the child class can be instance of either 
    SpindlesResults, SWResults or REMResults classes. 
    """

    tfrs: dict = field(init=False)
    """Instances of mne.time_frequency.AverageTFR per sleep stage.
    """

    @abstractmethod
    def detect():
        pass

    def _save_to_csv(self):
        self.results.summary().to_csv(
            self.output_dir
            / self.__class__.__name__
            / f"{self.__class__.__name__[:-4].lower()}.csv",
            index=False,
        )

    def _save_avg_fig(self):
        plt.savefig(
            self.output_dir
            / self.__class__.__name__
            / f"{self.__class__.__name__[:-4].lower()}_avg.png"
        )

    @Log.update
    def plot_average(self, save: bool = False, **kwargs):
        """Average of YASA's detected event.

        Args:
            save: Whether to save the figure to file. Defaults to False.
            **kwargs: Arguments passed to the YASA's plot_average().
        """
        self.results.plot_average(**kwargs)
        if save:
            self._save_avg_fig()

    @Log.update
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
            axis: Instance of `matplotlib.pyplot.axis <https://matplotlib.org/
                stable/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib-pyplot-axis>`_.
                Defaults to None.
            cmap: Matplotlib `colormap <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
                Defaults to "plasma".
            save: Whether to save the figure. Defaults to False.
        """
        from natsort import natsort_keygen
        from more_itertools import collapse
        from seaborn import color_palette

        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        subplots_args = subplots_args or dict()
        topomap_args.setdefault("cmap", color_palette("rocket_r", as_cmap=True))
        cbar_args.setdefault("label", prop)

        grouped_summary = (
            self.results.summary(grp_chan=True, grp_stage=True, aggfunc=aggfunc)
            .sort_values("Channel", key=natsort_keygen())
            .reset_index()
        )
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

    @Log.update
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
            topomap_args: Arguments passed to `mne.viz.plot_topomap() <https://mne.tools/
                stable/generated/mne.viz.plot_topomap.html>`_. Defaults to None.
            cbar_args: Arguments passed to `plt.colorbar() <https://matplotlib.org/stable
                /api/_as_gen/matplotlib.pyplot.colorbar.html>`_. Defaults to None.
            figure_args: Arguments passed to `plt.figure() <https://matplotlib.org/stable/
                api/_as_gen/matplotlib.pyplot.figure.html>`_. Defaults to None.

        """
        from natsort import natsort_keygen
        from more_itertools import collapse

        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        figure_args = figure_args or dict()
        topomap_args.setdefault("cmap", "plasma")
        topomap_args.setdefault("vlim", [None, None])

        if stages_to_plot == "all":
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
                per_stage = grouped_summary.loc[
                    grouped_summary["Stage"].isin(collapse([sleep_stages[stage]]))
                ].groupby("Channel")
                per_stage = (
                    per_stage.mean() if aggfunc == "mean" else per_stage.median()
                )
                per_stage = per_stage.sort_values(
                    "Channel", key=natsort_keygen()
                ).reset_index()

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
                self._plot_topomap(
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
                f"{stage}, {n_spindles} {self.__class__.__name__[:-4].lower()}",
                fontsize="xx-large",
            )

        fig.suptitle(f"{self.__class__.__name__[:-4]}", fontsize="xx-large")
        if save:
            fig.savefig(
                self.output_dir
                / self.__class__.__name__
                / f"topomap_{self.__class__.__name__[:-4].lower()}_collage.png"
            )

    @Log.update
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
            freqs: Lower and upper bounds of frequencies of interest in Hz, e.g., (10,20)
            n_freqs: Frequency resolution in TFR.
            time_before: Seconds before the event peak to get from the real data.
            time_after: Seconds after the event peak to get from the real data
            method: TFR transform method. Defaults to "morlet".
            save: Whether to save the TFRs to file. Defaults to False.
            **tfr_kwargs: Arguments passed to `mne.time_frequency.tfr_array_morlet()
                <https://mne.tools/stable/generated/mne.time_frequency.tfr_array_morlet.html>`_
                or `mne.time_frequency.tfr_array_multitaper() <https://mne.tools/stable/
                generated/mne.time_frequency.tfr_array_multitaper.html>`_.
        """
        assert self.results, "Run detect method first"
        assert (
            method == "morlet" or method == "multitaper"
        ), "method should be 'morlet' or 'multitaper'"
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

    @Log.update
    def read_tfrs(self, dirpath=None):
        """Loads TFRs stored in hdf5 files.

        Filenames should end with {type_of_event}_{sleep_stage}-tfr.h5

        Args:
            dirpath: Path to the directory containing hdf5 files. Defaults to None.
        """
        from mne.time_frequency import read_tfrs
        from pathlib import Path
        import re

        r = f"{self.__class__.__name__[:-4].lower()}_(.+)(?:-tfr.h5)"
        self.tfrs = dict()
        dirpath = (
            Path(dirpath) if dirpath else self.output_dir / self.__class__.__name__
        )
        for p in dirpath.glob("*tfr.h5"):
            m = re.search(r, str(p))
            if m:
                self.tfrs[m.groups()[0]] = read_tfrs(p)


@define(kw_only=True, slots=False)
class BaseSpectrum(BaseTopomap, ABC):
    """A template class for the spectral analysis."""

    psds: dict = field(init=False, factory=dict)

    @Log.update
    def plot_psds(
        self,
        picks,
        psd_range: tuple = (-40, 60),
        freq_range: tuple = (0, 60),
        dB=True,
        xscale: str = "linear",
        axis: plt.axis = None,
        plot_sensors: bool = False,
        save: bool = False,
        **subplots_kw,
    ):
        """Plot PSD per sleep stage.

        Args:
            psd_range: Range of y axis on PSD plot. Defaults to (-40, 60).
            freq_range: Range of x axis on PSD plot. Defaults to (0, 40).
            dB:
            xscale: Scale of the X axis, check available values at
                `matplotlib.axes.Axes.set_xscale <https://mne.tools/stable/
                generated/mne.io.Raw.html#mne.io.Raw.get_data>`_.
                Defaults to "linear".
            axis: Instance of `matplotlib.pyplot.axis <https://matplotlib.org/
                stable/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib-pyplot-axis>`_.
                Defaults to None.
            plot_sensors: Whether to plot sensor map showing which channels were used for
                computing PSD. Defaults to False.
            save: Whether to save the figure. Defaults to False.
            **subplots_kw: Arguments passed to the plt.subplots(). Have no effect if axis is provided.
                Defaults to None.
        """
        from mne.io.pick import _picks_to_idx

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
                label=f"{stage} ({spectrum.info.description}%)",
            )

        axis.set_xlim(freq_range)
        axis.set_ylim(psd_range)
        axis.set_xscale(xscale)
        axis.set_title("Welch's PSD")
        units = "[dB/Hz]" if dB else r"[$\mu V^{2}/\sqrt{Hz}$]"
        axis.set_ylabel("PSD " + units)
        axis.set_xlabel(f"{xscale} frequency [Hz]".capitalize())
        axis.legend()

        if plot_sensors:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            from more_itertools import flatten

            # color = "cyan"
            axins = inset_axes(axis, width="30%", height="30%", loc="lower left")
            psd_channels = _picks_to_idx(self.mne_raw.info, picks)
            interpolated = (
                _picks_to_idx(
                    self.mne_raw.info,
                    list(flatten(self.log.cleaning["interpolated"])),
                )
                if "interpolated" in self.log.cleaning
                else None
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
            fig.savefig(self.output_dir / self.__class__.__name__ / f"psd.png")

    @Log.update
    def plot_topomap(
        self,
        stage: str = "REM",
        band: dict = {"Delta": (0, 4)},
        dB: bool = False,
        axis: plt.axis = None,
        fooof: bool = False,
        save: bool = False,
        topomap_args: dict = None,
        cbar_args: dict = None,
        fooof_group_args: dict = None,
        fooof_get_band_peak_fg_args: dict = None,
        subplots_args: dict = None,
    ):
        """Plots topomap for a sleep stage and a frequency band.

        Args:
            stage: One of the sleep_stages keys. Defaults to "REM".
            band: Name-value pair - with name=arbitrary name
                and value=(l_freq, h_freq).
                Defaults to {"Delta": (0, 4)}.
            dB: Whether transform PSD to dB. Defaults to False.
            sleep_stages: Mapping between sleep stages names and their integer representations.
                Defaults to {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.
            axis: Instance of `matplotlib.pyplot.axis <https://matplotlib.org/
                stable/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib-pyplot-axis>`_.
                Defaults to None.
            fooof: Whether to plot parametrised spectra.
                More at `fooof <https://fooof-tools.github.io/fooof/auto_examples/analyses/
                plot_mne_example.html#sphx-glr-auto-examples-analyses-plot-mne-example-py>`_.
                Defaults to False.
            save: Whether to save the figure. Defaults to False.
            psd_method_args: Arguments passed to the PSD method, e.g., welch. Defaults to None.
            topomap_args: Arguments passed to `mne.viz.plot_topomap() <https://mne.tools/
                stable/generated/mne.viz.plot_topomap.html>`_. Defaults to None.
            cbar_args: Arguments passed to `plt.colorbar() <https://matplotlib.org/stable
                /api/_as_gen/matplotlib.pyplot.colorbar.html>`_. Defaults to None.
            fooof_group_args: Arguments passed to `fooof.FOOOFGroup() <https://fooof-tools.github.io/
                fooof/generated/fooof.FOOOFGroup.html#fooof.FOOOFGroup>`_. Defaults to None.
            fooof_get_band_peak_fg_args: Arguments passed to the `fooof.analysis.get_band_peak_fg()
                <https://fooof-tools.github.io/fooof/generated/fooof.analysis.get_band_peak_fg.html>`_.
                Defaults to None.
            subplots_args: Arguments passed to the plt.subplots(). Have no effect if axis is provided.
                Defaults to None.
        """
        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        subplots_args = subplots_args or dict()
        topomap_args.setdefault("cmap", "plasma")
        cbar_args.setdefault(
            "label", "dB/Hz" if dB else r"$\mu V^{2}/\sqrt{Hz}$" if not fooof else None
        )
        assert stage in self.psds, f"{stage} is not in self.psds"

        is_new_axis = False

        if axis is None:
            fig, axis = plt.subplots(**subplots_args)
            is_new_axis = True

        [(_, b)] = band.items()

        if fooof:
            psds = self._fooof(
                band=band,
                stage=stage,
                fooof_group_args=fooof_group_args,
                get_band_peak_fg_args=fooof_get_band_peak_fg_args,
            )

        else:
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

        self._plot_topomap(
            data=psds,
            axis=axis,
            topomap_args=topomap_args,
            cbar_args=cbar_args,
        )
        if is_new_axis:
            fig.suptitle(f"{stage} ({b[0]}-{b[1]} Hz)")
        if save and is_new_axis:
            fig.savefig(
                self.output_dir
                / self.__class__.__name__
                / f"topomap_psd_{list(band)[0]}.png"
            )

    @Log.update
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
        fooof: bool = False,
        low_percentile: float = 5,
        high_percentile: float = 95,
        save: bool = False,
        topomap_args: dict = None,
        cbar_args: dict = None,
        fooof_group_args: dict = None,
        fooof_get_band_peak_fg_args: dict = None,
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
            fooof: Whether to plot parametrised spectra.
                More at `fooof <https://fooof-tools.github.io/fooof/auto_examples/analyses/
                plot_mne_example.html#sphx-glr-auto-examples-analyses-plot-mne-example-py>`_.
                Defaults to False.
            low_percentile: Set min color value by percentile of the band data.
                Defaults to 5.
            high_percentile: Set max color value by percentile of the band data.
                Defaults to 95.
            save: Whether to save the figure. Defaults to False.
            psd_method_args: Arguments passed to the PSD method, e.g., welch. Defaults to None.
            topomap_args: Arguments passed to `mne.viz.plot_topomap() <https://mne.tools/
                stable/generated/mne.viz.plot_topomap.html>`_. Defaults to None.
            cbar_args: Arguments passed to `plt.colorbar() <https://matplotlib.org/stable
                /api/_as_gen/matplotlib.pyplot.colorbar.html>`_. Defaults to None.
            fooof_group_args: Arguments passed to `fooof.FOOOFGroup() <https://fooof-tools.github.io/
                fooof/generated/fooof.FOOOFGroup.html#fooof.FOOOFGroup>`_. Defaults to None.
            fooof_get_band_peak_fg_args: Arguments passed to the `fooof.analysis.get_band_peak_fg()
                <https://fooof-tools.github.io/fooof/generated/fooof.analysis.get_band_peak_fg.html>`_.
                Defaults to None.
            figure_args: Arguments passed to `plt.figure() <https://matplotlib.org/stable/
                api/_as_gen/matplotlib.pyplot.figure.html>`_. Defaults to None.
        """
        topomap_args = topomap_args or dict()
        cbar_args = cbar_args or dict()
        figure_args = figure_args or dict()
        topomap_args.setdefault("cmap", "plasma")
        topomap_args.setdefault("vlim", [None, None])
        cbar_args.setdefault(
            "label", "dB/Hz" if dB else r"$\mu V^{2}/\sqrt{Hz}$" if not fooof else None
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
                if fooof:
                    psds = self._fooof(
                        band={band_key: b},
                        stage=stage,
                        fooof_group_args=fooof_group_args,
                        get_band_peak_fg_args=fooof_get_band_peak_fg_args,
                    )
                else:
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

                self._plot_topomap(
                    data=psds_per_stage_per_band[row_index, col_index],
                    axis=axes[col_index],
                    topomap_args=topomap_args,
                    cbar_args=cbar_args,
                )
                axes[col_index].set_title(
                    f"{band_key} ({bands[band_key][0]}-{bands[band_key][1]} Hz)"
                )

            subfigs[row_index].suptitle(
                f"{stage} ({self.psds[stage].info.description}%)", fontsize="xx-large"
            )

        if save:
            fig.savefig(
                self.output_dir / self.__class__.__name__ / f"topomap_psd_collage.png"
            )

    def _fooof(self, band, stage, fooof_group_args=None, get_band_peak_fg_args=None):
        from fooof import FOOOFGroup
        from fooof.bands import Bands
        from fooof.analysis import get_band_peak_fg

        fooof_group_args = fooof_group_args or dict()
        get_band_peak_fg_args = get_band_peak_fg_args or dict()
        fooof_group_args.setdefault("peak_width_limits", (1, 6))
        fooof_group_args.setdefault("min_peak_height", 0.15)
        fooof_group_args.setdefault("peak_threshold", 2.0)
        fooof_group_args.setdefault("verbose", False)
        # Initialize a FOOOFGroup object, with desired settings
        fg = FOOOFGroup(**fooof_group_args)

        # Define the frequency range to fit
        freq_range = [1, 45]

        fg.fit(
            self.psds[stage]._freqs,
            self.psds[stage]._data,
            freq_range=freq_range,
        )

        # Define frequency bands of interest
        bands = Bands(band)

        # Extract peaks
        peaks = get_band_peak_fg(fg, bands[list(band)[0]], **get_band_peak_fg_args)

        peaks[np.where(np.isnan(peaks))] = 0
        # Extract the power values from the detected peaks
        psds = peaks[:, 1]

        return psds


@define(kw_only=True, slots=False)
class SleepSpectrum(mne.time_frequency.spectrum.BaseSpectrum):
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
            method, data, hypno, stage_idx, fmin, fmax, n_jobs, method_kw, verbose
        )
        # check for correct shape and bad values
        self._check_values()
        del self._shape
        # save memory
        del self.inst

    def _compute_spectra(
        self, method, data, hypno, stage_idx, fmin, fmax, n_jobs, method_kw, verbose
    ):
        from scipy import ndimage
        from more_itertools import collapse

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

            uniform_region_samples = 2000
            ranges = [
                list(range(region.start, region.stop, uniform_region_samples)) for region in regions
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
        self.info.description = str(round(n_samples / n_samples_total * 100, 2))
        # this is *expected* shape, it gets asserted later in _check_values()
        # (and then deleted afterwards)
        self._shape = (len(self.ch_names), len(self.freqs))
        # we don't need these anymore, and they make save/load harder
        del self._picks
        del self._psd_func
        del self._time_mask
