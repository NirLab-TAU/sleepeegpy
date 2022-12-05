""" Pipeline Module
"""
import matplotlib.pyplot as plt
import numpy as np


class Pipe:
    """The main pipeline class"""

    def __init__(
        self,
        subject_code: str,
        path_to_mff: str,
        output_directory: str,
        path_to_hypno=None,
        sf_hypno=1
    ):
        """Constructs an instanse of the class.
        """

        import os
        import errno
        from pathlib import Path
        from mne.io import read_raw_egi

        self.subject = subject_code
        self.output_dir = Path(output_directory)

        # Check that the directory exists.
        if not self.output_dir.is_dir():
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                output_directory
                )

        # Try import mff file, raise exception if something's wrong.
        mff_file = Path(path_to_mff)
        try:
            self.mne_raw = read_raw_egi(mff_file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                path_to_mff) from exc
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

        # Try import hypno file, raise exception if something's wrong.
        try:
            hypno_file = Path(path_to_hypno)
            self.hypno = np.loadtxt(hypno_file)
            self.sf_hypno = sf_hypno
            self.__upsample_hypno()
 
        except TypeError:
            self.hypno = np.empty(0)
            self.hypno_up = np.empty(0)
            self.sf_hypno = None
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                path_to_hypno) from exc
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


    @property
    def sf(self):
        return self.mne_raw.info['sfreq']

    def resample(self, sf=250, chunk_secs=100, picks=()):
        """ Resamples the data """
        self.mne_raw.pick_channels(picks)
        duration = self.mne_raw.n_times/self.sf
        quot, rem = divmod(duration, chunk_secs)
        samples = [self.mne_raw.copy().crop(tmin=i*chunk_secs, tmax=(i+1)*chunk_secs).resample(sf, n_jobs=-1, verbose=False) for i in range(int(quot))]
        samples.append(self.mne_raw.copy().crop(tmin=quot*chunk_secs, tmax=quot*chunk_secs+rem, include_tmax=False).resample(sf, n_jobs=-1, verbose=False))
        self.mne_raw = samples[0].copy()
        self.mne_raw.append(samples[1:])
        self.__upsample_hypno()

    def sleep_stats(self, save_to_csv: bool = False):
        """sleep statistics"""

        from yasa import sleep_statistics
        from csv import DictWriter
        assert self.hypno.any(), 'There is no hypnogram to get stats from.'
        stats = sleep_statistics(self.hypno, self.sf_hypno)
        if save_to_csv:
            with open(self.output_dir/'sleep_stats.csv', 'w', newline='') as csv_file:
                w = DictWriter(csv_file, stats.keys())
                w.writeheader()
                w.writerow(stats)
            return
        return stats


    def plot_hypnospectrogram(
        self,
        electrode_name: str,
        win_sec: float = 4,
        freq_range: tuple = (0, 40),
        cmap: str = 'inferno',
        overlap=False,
        save: bool = False
    ):
        """ Plots yasa's hypnogram and spectrogram.
        """
        # Import data from the raw mne file.
        data = self.mne_raw.get_data([electrode_name], units="uV")[0]
            # Create a plot figure
        fig = self.__plot_hypnospectrogram(
            data,
            self.sf,
            self.hypno_up,
            win_sec=win_sec,
            fmin=freq_range[0],
            fmax=freq_range[1],
            trimperc=0,
            cmap=cmap,
            overlap=overlap)
        # Save the figure if 'save' set to True 
        if save:
            fig.savefig(self.output_dir / f'{self.subject}_spectrogram.png')


    def plot_psd_per_stage(
        self,
        electrode_name: str,
        nperseg: int = 4096,
        psd_range: tuple = (-60, 60),
        freq_range: tuple = (0, 40),
        xscale: str = 'linear',
        sleep_stages: dict = {'Wake' :0, 'N1' :1, 'N2': 2, 'N3': 3, 'REM': 4},
        axis=None,
        save=False
    ):
        """Plots PSDs for multiple sleep stages.
        """

        from scipy import signal

        if not axis:
            fig, axis = plt.subplots()

        # win = sf*1024/250
        # Import data from the raw mne file.
        data = self.mne_raw.get_data([electrode_name], units='uV')[0]
        signal_by_stage = {}

        # For every stage get its signal, 
        # calculate signal's PSD, 
        # transform PSD units to dB,
        # plot it.
        for stage, index in sleep_stages.items():
            signal_by_stage[stage] = np.take(data, np.where(np.in1d(self.hypno_up, index))[0])
            freqs, psd = signal.welch(
                signal_by_stage[stage], self.sf, nperseg=nperseg)
            psd = 10 * np.log10(psd)
            axis.plot(freqs, psd, label=stage)

        axis.set_xlim(freq_range)
        axis.set_ylim(psd_range)
        axis.set_xscale(xscale)
        axis.set_title("Welch's PSD")
        axis.set_ylabel('PSD [dB/Hz]')
        axis.set_xlabel(f'{xscale} frequency [Hz]'.capitalize())
        axis.legend()
        # Save the figure if 'save' set to True
        if save:
            fig.savefig(self.output_dir / f'{self.subject}_psd.png')


    def __upsample_hypno(self):

        from yasa import hypno_upsample_to_data
        self.hypno_up = hypno_upsample_to_data(
                self.hypno, 
                self.sf_hypno, 
                self.mne_raw,
                verbose=False)

    def __plot_hypnospectrogram(
        self,
        data,
        sf,
        hypno,
        win_sec=30,
        fmin=0.5,
        fmax=25,
        trimperc=2.5,
        cmap="Spectral_r",
        vmin=None,
        vmax=None,
        overlap=False
    ):
        """
        ?
        """
        # Increase font size while preserving original
        old_fontsize = plt.rcParams["font.size"]
        plt.rcParams.update({"font.size": 18})

        # Safety checks
        assert isinstance(data, np.ndarray), "Data must be a 1D NumPy array."
        assert isinstance(sf, (int, float)), "sf must be int or float."
        assert data.ndim == 1, "Data must be a 1D (single-channel) NumPy array."
        assert isinstance(win_sec, (int, float)), "win_sec must be int or float."
        assert isinstance(fmin, (int, float)), "fmin must be int or float."
        assert isinstance(fmax, (int, float)), "fmax must be int or float."
        assert fmin < fmax, "fmin must be strictly inferior to fmax."
        assert fmax < sf / 2, "fmax must be less than Nyquist (sf / 2)."
        assert isinstance(vmin, (int, float, type(None))), "vmin must be int, float, or None."
        assert isinstance(vmax, (int, float, type(None))), "vmax must be int, float, or None."
        if vmin is not None:
            assert isinstance(vmax, (int, float)), "vmax must be int or float if vmin is provided"
        if vmax is not None:
            assert isinstance(vmin, (int, float)), "vmin must be int or float if vmax is provided"

        if not hypno.any():
            fig, ax = plt.subplots(nrows=1, figsize=(12, 4))
            im = self.__plot_spectrogram(data, sf, win_sec, fmin, fmax, trimperc, cmap, ax)
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.95, fraction=0.1, aspect=25)
            cbar.ax.set_ylabel("Log Power (dB / Hz)", rotation=270, labelpad=20)
            # Revert font-size
            plt.rcParams.update({"font.size": old_fontsize})
            return fig

        if overlap:
            fig, ax = plt.subplots(nrows=1, figsize=(12, 4))
            im = self.__plot_spectrogram(data, sf, win_sec, fmin, fmax, trimperc, cmap, ax)
            ax_hypno = ax.twinx()
            self.__plot_hypnogram(data, sf, hypno, ax_hypno)
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.95, fraction=0.1, aspect=25, pad=0.1)
            cbar.ax.set_ylabel("Log Power (dB / Hz)", rotation=270, labelpad=20)
            # Revert font-size
            plt.rcParams.update({"font.size": old_fontsize})
            return fig

        fig, (ax0, ax1) = plt.subplots(
            nrows=2, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 2]}
        )
        plt.subplots_adjust(hspace=0.1)

        # Hypnogram (top axis)
        self.__plot_hypnogram(data, sf, hypno, ax0)
        # Spectrogram (bottom axis)
        self.__plot_spectrogram(data, sf, win_sec, fmin, fmax, trimperc, cmap, ax1, vmin=vmin, vmax=vmax)
        # Revert font-size
        plt.rcParams.update({"font.size": old_fontsize})
        return fig

    @staticmethod
    def __plot_hypnogram(data, sf, hypno, ax0):

        from pandas import Series

        hypno = np.asarray(hypno).astype(int)
        assert hypno.ndim == 1, "Hypno must be 1D."
        assert hypno.size == data.size, "Hypno must have the same sf as data."
        t_hyp = np.arange(hypno.size) / (sf * 3600)
        # Make sure that REM is displayed after Wake
        hypno = Series(hypno).map({-2: -2, -1: -1, 0: 0, 1: 2, 2: 3, 3: 4, 4: 1}).values
        hypno_rem = np.ma.masked_not_equal(hypno, 1)
        # Hypnogram (top axis)
        ax0.step(t_hyp, -1 * hypno, color="k")
        ax0.step(t_hyp, -1 * hypno_rem, color="r")
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
    def __plot_spectrogram(data, sf, win_sec, fmin, fmax, trimperc, cmap, ax, vmin=None, vmax=None):
        
        from matplotlib.colors import Normalize
        from lspopt import spectrogram_lspopt
        # Calculate multi-taper spectrogram
        nperseg = int(win_sec * sf)
        assert data.size > 2 * nperseg, "Data length must be at least 2 * win_sec."
        f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
        Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

        # Select only relevant frequencies (up to 30 Hz)
        good_freqs = np.logical_and(f >= fmin, f <= fmax)
        Sxx = Sxx[good_freqs, :]
        f = f[good_freqs]
        t /= 3600  # Convert t to hours

        # Normalization
        if vmin is None:
            vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto")
        ax.set_xlim(0, t.max())
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [hrs]")
        return im
