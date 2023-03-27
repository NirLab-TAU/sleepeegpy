"""Bla-bla-bla
"""

import os
import errno
from pathlib import Path
from attrs import define, field
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import mne.io

from typing import TypeVar, Type
from collections.abc import Iterable


# For type annotation of pipe elements.
BasePipeType = TypeVar("BasePipeType", bound="BasePipe")


@define(kw_only=True)
class BasePipe:
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
        butterfly: bool = False,
        save_annotations: bool = False,
        save_bad_channels: bool = False,
        scalings: str | dict = "auto",
        use_opengl: bool = False,
        overwrite: bool = False,
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
        """
        from mne import pick_types

        order = pick_types(self.mne_raw.info, eeg=True) if butterfly else None
        self.mne_raw.plot(
            theme="dark",
            block=True,
            scalings=scalings,
            bad_color="r",
            proj=False,
            order=order,
            butterfly=butterfly,
            use_opengl=use_opengl,
        )
        if save_annotations:
            self.mne_raw.annotations.save(
                self.output_dir / "annotations.txt", overwrite=overwrite
            )
        if save_bad_channels:
            with open(
                self.output_dir / "bad_channels.txt", "w" if overwrite else "x"
            ) as f:
                for bad in self.mne_raw.info["bads"]:
                    f.write(f"{bad}\n")

    def save_raw(self, fname: str):
        """A wrapper for `mne.io.Raw.save <https://mne.tools/stable/
        generated/mne.io.Raw.html#mne.io.Raw.save>`_.

        Args:
            fname: filename for the fif file being saved.
        """
        path_to_resampled = self.output_dir / "saved_raw"
        path_to_resampled.mkdir(exist_ok=True)
        self.mne_raw.save(path_to_resampled / fname)


@define(kw_only=True)
class BaseSpectrum(ABC):
    """A template class for the spectral analysis."""

    def plot_psd_per_stage(
        self,
        picks: str | Iterable[str] = ("E101",),
        sec_per_seg: float = 4.096,
        psd_range: tuple = (-40, 60),
        freq_range: tuple = (0, 40),
        xscale: str = "linear",
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        axis: plt.axis = None,
        save: bool = False,
    ):
        """Plot PSD per sleep stage.

        Args:
            picks: Channels to calculate PSD on, more info at
                `mne.io.Raw.get_data <https://mne.tools/stable/generated/
                mne.io.Raw.html#mne.io.Raw.get_data>`_.
                Defaults to ("E101",).
            sec_per_seg: Welch segment length in seconds. Defaults to 4.096.
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
            save: Whether to save the figure. Defaults to False.
        """

        is_axis = False

        if not axis:
            fig, axis = plt.subplots()
        else:
            is_axis = True

        psd_per_stage = self._compute_psd_per_stage(
            picks=picks,
            sleep_stages=sleep_stages,
            sec_per_seg=sec_per_seg,
            avg_ref=False,
            dB=True,
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
        # Save the figure if 'save' set to True and no axis has been passed.
        if save and not is_axis:
            fig.savefig(self.output_dir / f"psd.png")

    def plot_topomap_per_stage(
        self,
        stage: str = "REM",
        band: dict = {"Delta": (0, 4)},
        sec_per_seg: float = 4.096,
        dB: bool = False,
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        axis: plt.axis = None,
        fooof: bool = False,
        cmap: str = "plasma",
        save: bool = False,
    ):
        """Plots topomap for a sleep stage and a frequency band.

        Args:
            stage: One of the sleep_stages keys. Defaults to "REM".
            band: Name-value pair - with name=arbitrary name
                and value=(l_freq, h_freq).
                Defaults to {"Delta": (0, 4)}.
            sec_per_seg: Welch segment length in seconds. Defaults to 4.096.
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
            cmap: Matplotlib `colormap <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
                Defaults to "plasma".
            save: Whether to save the figure. Defaults to False.
        """
        from mne.viz import plot_topomap

        if not hasattr(self, "psd_per_stage") or stage not in self.psd_per_stage.keys():
            assert (
                stage in sleep_stages.keys()
            ), f"sleep_stages should contain provided stage"

        is_axis = False

        if axis is None:
            fig, axis = plt.subplots()
            is_axis = True

        if not hasattr(self, "psd_per_stage") or stage not in self.psd_per_stage.keys():
            self.psd_per_stage = self._compute_psd_per_stage(
                picks=["eeg"],
                sleep_stages=sleep_stages,
                sec_per_seg=sec_per_seg,
                avg_ref=True,
                dB=dB,
            )

        [(k, b)] = band.items()

        if fooof:
            from fooof import FOOOFGroup
            from fooof.bands import Bands
            from fooof.analysis import get_band_peak_fg

            # Initialize a FOOOFGroup object, with desired settings
            fg = FOOOFGroup(
                peak_width_limits=[1, 6],
                min_peak_height=0.15,
                peak_threshold=2.0,
                max_n_peaks=6,
                verbose=False,
            )

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
            peaks = get_band_peak_fg(fg, bands[list(band)[0]])

            peaks[np.where(np.isnan(peaks))] = 0
            # Extract the power values from the detected peaks
            psds = peaks[:, 1]

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

        im, cn = plot_topomap(
            psds, self.mne_raw.info, size=5, cmap=cmap, axes=axis, show=False
        )

        # divider = make_axes_locatable(axis)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(
            im,
            ax=axis,
            orientation="vertical",
            shrink=0.6,
            label="dB/Hz" if dB else r"$\mu V^{2}/Hz$",
        )

        if is_axis:
            fig.suptitle(f"{stage} ({b[0]}-{b[1]} Hz)")
        if save and is_axis:
            fig.savefig(self.output_dir / f"topomap.png")

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
        sec_per_seg: float = 4.096,
        dB: bool = False,
        sleep_stages: dict = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        fooof: bool = False,
        cmap: str = "plasma",
        save: bool = False,
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
            sec_per_seg: Welch segment length in seconds.. Defaults to 4.096.
            dB: Whether transform PSD to dB. Defaults to False.
            sleep_stages: Mapping between sleep stages names and their integer representations.
                Defaults to {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.
            fooof: Whether to plot parametrised spectra.
                More at `fooof <https://fooof-tools.github.io/fooof/auto_examples/analyses/
                plot_mne_example.html#sphx-glr-auto-examples-analyses-plot-mne-example-py>`_.
                Defaults to False. Defaults to False.
            cmap: Matplotlib `colormap <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
                Defaults to "plasma".
            save: Whether to save the figure. Defaults to False.
        """
        if stages_to_plot == "all":
            stages_to_plot = sleep_stages.keys()
        n_rows = len(stages_to_plot)
        n_cols = len(bands)

        fig = plt.figure(figsize=(n_cols * 4, n_rows * 4), layout="constrained")
        subfigs = fig.subfigures(n_rows, 1)

        for row_index, stage in enumerate(stages_to_plot):
            axes = subfigs[row_index].subplots(1, n_cols)

            for col_index, band_key in enumerate(bands):
                self.plot_topomap_per_stage(
                    stage=stage,
                    band={band_key: bands[band_key]},
                    sec_per_seg=sec_per_seg,
                    dB=dB,
                    sleep_stages=sleep_stages,
                    axis=axes[col_index],
                    cmap=cmap,
                    fooof=fooof,
                )
                axes[col_index].set_title(
                    f"{band_key} ({bands[band_key][0]}-{bands[band_key][1]} Hz)"
                )

            subfigs[row_index].suptitle(
                f"{stage} ({self.psd_per_stage[stage][2]}%)", fontsize="xx-large"
            )

        if save:
            fig.savefig(self.output_dir / f"topomap_collage.png")

    @abstractmethod
    def _compute_psd_per_stage(self):
        pass
