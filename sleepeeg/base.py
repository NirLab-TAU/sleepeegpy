import os
import errno

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from typing import TypeVar, Type
from attrs import define, field

import matplotlib.pyplot as plt
import numpy as np
import mne
from tqdm import tqdm

# For type annotation of pipe elements.
BasePipeType = TypeVar("BasePipeType", bound="BasePipe")


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
        mne_plot_args: dict = None,
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
            mne_plot_args: Arguments passed to `raw.plot() <https://mne.tools/stable/
                generated/mne.io.Raw.html#mne.io.Raw.plot>`_. Defaults to None.
        """
        mne_plot_args = mne_plot_args or dict()
        mne_plot_args.setdefault("theme", "dark")
        mne_plot_args.setdefault("block", True)
        mne_plot_args.setdefault("bad_color", "r")
        mne_plot_args.setdefault("scalings", "auto")
        self.mne_raw.plot(**mne_plot_args)

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

    def _plot_sensors(
        self,
        ch_colors=None,
        kind="topomap",
        ch_type=None,
        title=None,
        show_names=False,
        ch_groups=None,
        to_sphere=True,
        axes=None,
        block=False,
        show=False,
        sphere=None,
        pointsize=7,
        linewidth=1,
    ):
        from mne.viz.evoked import _rgb
        from mne.viz.utils import _plot_sensors
        from mne.defaults import _handle_default
        from mne.io.constants import FIFF
        from mne.io.pick import (
            channel_type,
            channel_indices_by_type,
            pick_channels,
            _DATA_CH_TYPES_SPLIT,
            _contains_ch_type,
        )
        from mne.utils import _check_ch_locs, _check_option, warn
        from mne.transforms import apply_trans

        info = self.mne_raw.info
        _check_option("kind", kind, ["topomap", "3d", "select"])
        if not isinstance(info, mne.io.Info):
            raise TypeError(f"info must be an instance of Info not {type(info)}")
        ch_indices = channel_indices_by_type(info)
        allowed_types = _DATA_CH_TYPES_SPLIT
        if ch_type is None:
            for this_type in allowed_types:
                if _contains_ch_type(info, this_type):
                    ch_type = this_type
                    break
            picks = ch_indices[ch_type]
        elif ch_type == "all":
            picks = list()
            for this_type in allowed_types:
                picks += ch_indices[this_type]
        elif ch_type in allowed_types:
            picks = ch_indices[ch_type]
        else:
            raise ValueError(f"ch_type must be one of {allowed_types} not {ch_type}!")

        if len(picks) == 0:
            raise ValueError(f"Could not find any channels of type {ch_type}.")

        if not _check_ch_locs(info=info, picks=picks):
            raise RuntimeError("No valid channel positions found")

        dev_head_t = info["dev_head_t"]
        chs = [info["chs"][pick] for pick in picks]
        pos = np.empty((len(chs), 3))
        for ci, ch in enumerate(chs):
            pos[ci] = ch["loc"][:3]
            if ch["coord_frame"] == FIFF.FIFFV_COORD_DEVICE:
                if dev_head_t is None:
                    warn(
                        "dev_head_t is None, transforming MEG sensors to head "
                        "coordinate frame using identity transform"
                    )
                    dev_head_t = np.eye(4)
                pos[ci] = apply_trans(dev_head_t, pos[ci])
        del dev_head_t

        ch_names = np.array([ch["ch_name"] for ch in chs])
        bads = [idx for idx, name in enumerate(ch_names) if name in info["bads"]]
        ch_colors = {} if not ch_colors else ch_colors
        if ch_groups is None:
            def_colors = _handle_default("color")
            colors = [
                ch_colors[self.mne_raw.info.ch_names[pick]]
                if self.mne_raw.info.ch_names[pick] in ch_colors
                else def_colors[channel_type(info, pick)]
                for i, pick in enumerate(picks)
            ]
        else:
            if ch_groups in ["position", "selection"]:
                # Avoid circular import
                from mne.channels import (
                    read_vectorview_selection,
                    _SELECTIONS,
                    _EEG_SELECTIONS,
                    _divide_to_regions,
                )

                if ch_groups == "position":
                    ch_groups = _divide_to_regions(info, add_stim=False)
                    ch_groups = list(ch_groups.values())
                else:
                    ch_groups, color_vals = list(), list()
                    for selection in _SELECTIONS + _EEG_SELECTIONS:
                        channels = pick_channels(
                            info["ch_names"],
                            read_vectorview_selection(selection, info=info),
                        )
                        ch_groups.append(channels)
                color_vals = np.ones((len(ch_groups), 4))
                for idx, ch_group in enumerate(ch_groups):
                    color_picks = [
                        np.where(picks == ch)[0][0] for ch in ch_group if ch in picks
                    ]
                    if len(color_picks) == 0:
                        continue
                    x, y, z = pos[color_picks].T
                    color = np.mean(_rgb(x, y, z), axis=0)
                    color_vals[idx, :3] = color  # mean of spatial color
            else:
                import matplotlib.pyplot as plt

                colors = np.linspace(0, 1, len(ch_groups))
                color_vals = [plt.cm.jet(colors[i]) for i in range(len(ch_groups))]
            if not isinstance(ch_groups, (np.ndarray, list)):
                raise ValueError(
                    "ch_groups must be None, 'position', "
                    "'selection', or an array. Got %s." % ch_groups
                )
            colors = np.zeros((len(picks), 4))
            for pick_idx, pick in enumerate(picks):
                for ind, value in enumerate(ch_groups):
                    if pick in value:
                        colors[pick_idx] = color_vals[ind]
                        break
        title = "Sensor positions (%s)" % ch_type if title is None else title
        fig = _plot_sensors(
            pos,
            info,
            picks,
            colors,
            bads,
            ch_names,
            title,
            show_names,
            axes,
            show,
            kind,
            block,
            to_sphere,
            sphere,
            pointsize=pointsize,
            linewidth=linewidth,
        )

        if kind == "select":
            return fig, fig.lasso.selection
        return fig

    def save_raw(self, fname: str):
        """A wrapper for `mne.io.Raw.save <https://mne.tools/stable/
        generated/mne.io.Raw.html#mne.io.Raw.save>`_.

        Args:
            fname: filename for the fif file being saved.
        """
        fif_folder = self.output_dir / self.__class__.__name__
        self.mne_raw.save(fif_folder / fname)


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

    def __upsample_hypno(self):
        from yasa import hypno_upsample_to_data

        self.hypno_up = hypno_upsample_to_data(
            self.hypno, self.hypno_freq, self.mne_raw, verbose=False
        )

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

    def plot_average(self, save: bool = False, yasa_args=None):
        """Average of YASA's detected event.

        Args:
            save: Whether to save the figure to file. Defaults to False.
            yasa_args: Arguments passed to the YASA's plot_average(). Defaults to None.
        """
        yasa_args = yasa_args or dict()
        self.results.plot_average(**yasa_args)
        if save:
            self._save_avg_fig()

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

    def apply_tfr(
        self,
        freqs: Iterable[float],
        n_freqs: int,
        time_before: float,
        time_after: float,
        method: str = "morlet",
        save: bool = False,
        tfr_method_args: dict = None,
    ):
        """Transforms the events signal to time-frequency representation.

        Args:
            freqs: Lower and upper bounds of frequencies of interest in Hz, e.g., (10,20)
            n_freqs: Frequency resolution in TFR.
            time_before: Seconds before the event peak to get from the real data.
            time_after: Seconds after the event peak to get from the real data
            method: TFR transform method. Defaults to "morlet".
            save: Whether to save the TFRs to file. Defaults to False.
            tfr_method_args: Arguments passed to `mne.time_frequency.tfr_array_morlet()
                <https://mne.tools/stable/generated/mne.time_frequency.tfr_array_morlet.html>`_
                or `mne.time_frequency.tfr_array_multitaper() <https://mne.tools/stable/
                generated/mne.time_frequency.tfr_array_multitaper.html>`_. Defaults to None.
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

        tfr_method_args = tfr_method_args or dict()
        tfr_method_args.setdefault("n_jobs", -1)
        tfr_method_args.setdefault("verbose", "error")
        tfr_method_args["output"] = "avg_power"

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
                        **tfr_method_args,
                    )
                    for channel, v in tqdm(for_tfrs.items())
                }
            elif method == "multitaper":
                tfrs = {
                    channel: mne.time_frequency.tfr_array_multitaper(
                        v,
                        self.sf,
                        freqs,
                        **tfr_method_args,
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

    psd_per_stage: dict = field(init=False)
    """ Dictionary of the form sleep_stage:[freqs array with shape (n_freqs), 
    psd array with shape (n_electrodes, n_freqs), 
    sleep_stage percent from the whole unrejected data]
    """

    def plot_psd_per_stage(
        self,
        picks: str | Iterable[str] = ("E101",),
        psd_range: tuple = (-40, 60),
        freq_range: tuple = (0, 40),
        xscale: str = "linear",
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        axis: plt.axis = None,
        plot_sensors: bool = False,
        save: bool = False,
        psd_method_args: dict = None,
        subplots_args: dict = None,
    ):
        """Plot PSD per sleep stage.

        Args:
            picks: Channels to calculate PSD on, more info at
                `mne.io.Raw.get_data <https://mne.tools/stable/generated/
                mne.io.Raw.html#mne.io.Raw.get_data>`_.
                Defaults to ("E101",).
            psd_range: Range of y axis on PSD plot. Defaults to (-40, 60).
            freq_range: Range of x axis on PSD plot. Defaults to (0, 40).
            xscale: Scale of the X axis, check available values at
                `matplotlib.axes.Axes.set_xscale <https://mne.tools/stable/
                generated/mne.io.Raw.html#mne.io.Raw.get_data>`_.
                Defaults to "linear".
            sleep_stages: Mapping between sleep stages names and their integer representations.
                Defaults to {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.
            axis: Instance of `matplotlib.pyplot.axis <https://matplotlib.org/
                stable/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib-pyplot-axis>`_.
                Defaults to None.
            plot_sensors: Whether to plot sensor map showing which channels were used for
                computing PSD. Defaults to False.
            save: Whether to save the figure. Defaults to False.
            psd_method_args: Arguments passed to the PSD method, e.g., welch. Defaults to None.
            subplots_args: Arguments passed to the plt.subplots(). Have no effect if axis is provided.
                Defaults to None.
        """
        subplots_args = subplots_args or dict()
        is_new_axis = False

        if not axis:
            fig, axis = plt.subplots(**subplots_args)
            is_new_axis = True

        psd_per_stage = self._compute_psd_per_stage(
            picks=picks,
            sleep_stages=sleep_stages,
            avg_ref=False,
            dB=True,
            method_args=psd_method_args,
        )

        for stage in sleep_stages:
            axis.plot(
                psd_per_stage[stage][0],
                psd_per_stage[stage][1][0],
                label=f"{stage} ({psd_per_stage[stage][2]}%)",
            )

        axis.set_xlim(freq_range)
        axis.set_ylim(psd_range)
        axis.set_xscale(xscale)
        axis.set_title("Welch's PSD")
        axis.set_ylabel("PSD [dB/Hz]")
        axis.set_xlabel(f"{xscale} frequency [Hz]".capitalize())
        axis.legend()

        if plot_sensors:
            from mne.io.pick import _picks_to_idx
            import matplotlib.patches as mpatches
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            color = "cyan"
            # ax = fig.add_subplot(3, 3, 7)
            axins = inset_axes(axis, width="30%", height="30%", loc="lower left")
            channels = np.array(self.mne_raw.info.ch_names)[
                _picks_to_idx(self.mne_raw.info, picks)
            ].tolist()
            self._plot_sensors(
                ch_colors={ch: color for ch in channels},
                axes=axins,
            )

            patches = []
            patches.append(mpatches.Patch(color=color, label="psd"))
            if self.mne_raw.info["bads"]:
                patches.append(mpatches.Patch(color="red", label="bad"))
            axins.legend(
                handles=patches,
                loc="lower left",
                bbox_to_anchor=(1, 0),
                fontsize="x-small",
            )

        # Save the figure if 'save' set to True and no axis has been passed.
        if save and is_new_axis:
            fig.savefig(self.output_dir / self.__class__.__name__ / f"psd.png")

    def plot_topomap(
        self,
        stage: str = "REM",
        band: dict = {"Delta": (0, 4)},
        dB: bool = False,
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        axis: plt.axis = None,
        fooof: bool = False,
        save: bool = False,
        psd_method_args: dict = None,
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
            "label", "dB/Hz" if dB else r"$\mu V^{2}/Hz$" if not fooof else None
        )
        assert (
            stage in sleep_stages.keys()
        ), f"sleep_stages should contain provided stage"

        is_new_axis = False

        if axis is None:
            fig, axis = plt.subplots(**subplots_args)
            is_new_axis = True

        self.psd_per_stage = self._compute_psd_per_stage(
            picks=["eeg"],
            sleep_stages=sleep_stages,
            avg_ref=True,
            dB=dB,
            method_args=psd_method_args,
        )

        [(_, b)] = band.items()

        if fooof:
            psds = self._fooof(
                band=band,
                stage=stage,
                fooof_group_args=fooof_group_args,
                get_band_peak_fg_args=fooof_get_band_peak_fg_args,
            )

        else:
            psds = np.take(
                self.psd_per_stage[stage][1],
                np.where(
                    np.logical_and(
                        self.psd_per_stage[stage][0] >= b[0],
                        self.psd_per_stage[stage][0] <= b[1],
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
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        fooof: bool = False,
        low_percentile: float = 5,
        high_percentile: float = 95,
        save: bool = False,
        psd_method_args: dict = None,
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
            "label", "dB/Hz" if dB else r"$\mu V^{2}/Hz$" if not fooof else None
        )

        if stages_to_plot == "all":
            stages_to_plot = sleep_stages.keys()
        n_rows = len(stages_to_plot)
        n_cols = len(bands)

        figure_args.setdefault("figsize", (n_cols * 4, n_rows * 4))
        figure_args.setdefault("layout", "constrained")
        fig = plt.figure(**figure_args)
        subfigs = fig.subfigures(n_rows, 1)

        self.psd_per_stage = self._compute_psd_per_stage(
            picks=["eeg"],
            sleep_stages=sleep_stages,
            avg_ref=True,
            dB=dB,
            method_args=psd_method_args,
        )

        if low_percentile:
            perc_low = dict()

        if high_percentile:
            perc_high = dict()

        psds_per_stage_per_band = np.empty(
            (len(stages_to_plot), len(bands)), dtype=object
        )
        for col_index, (band_key, b) in enumerate(bands.items()):
            for_perc = []
            for row_index, stage in enumerate(self.psd_per_stage):
                if fooof:
                    psds = self._fooof(
                        band={band_key: b},
                        stage=stage,
                        fooof_group_args=fooof_group_args,
                        get_band_peak_fg_args=fooof_get_band_peak_fg_args,
                    )
                else:
                    psds = np.take(
                        self.psd_per_stage[stage][1],
                        np.where(
                            np.logical_and(
                                self.psd_per_stage[stage][0] >= b[0],
                                self.psd_per_stage[stage][0] <= b[1],
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
                f"{stage} ({self.psd_per_stage[stage][2]}%)", fontsize="xx-large"
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
            self.psd_per_stage[stage][0],
            self.psd_per_stage[stage][1],
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

    @abstractmethod
    def _compute_psd_per_stage(self):
        pass
