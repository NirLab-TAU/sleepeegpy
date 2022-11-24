""" Pipeline Module
"""
import os
import errno
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yasa
from mne.io import read_raw_egi
from scipy import signal


class Pipe:
    """The main pipeline class"""

    def __init__(
        self,
        subject_code: str,
        path_to_mff: str,
        output_directory: str,
        path_to_hypno: str,
        sf_hypno: int = 1
    ):
        """Constructs an instanse of the class.
        """

        self.subject = subject_code
        self.output_dir = Path(output_directory)
        self.sf_hypno = sf_hypno

        mff_file = Path(path_to_mff)
        hypno_file = Path(path_to_hypno)

        # Check that the directory exists.
        if not self.output_dir.is_dir():
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                output_directory
                )

        # Try import mff file, raise exception if something's wrong.
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
            self.hypno = np.loadtxt(hypno_file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                path_to_hypno) from exc

    def plot_hypnospectrogram(
        self,
        electrode_name: str,
        win_sec: float = 4,
        freq_range: tuple = (0, 40),
        cmap: str = 'inferno',
        save: bool = False
    ):
        """ Plots yasa's hypnogram and spectrogram.
        """
        # Import data from the raw mne file.
        data = self.mne_raw.get_data([electrode_name], units="uV")[0]
        # Upsample hypnogram according to the data
        hypno_up = yasa.hypno_upsample_to_data(
            self.hypno, self.sf_hypno, data=self.mne_raw)
        # Create a plot figure
        fig = yasa.plot_spectrogram(
            data,
            self.mne_raw.info['sfreq'],
            hypno_up,
            win_sec=win_sec,
            fmin=freq_range[0],
            fmax=freq_range[1],
            trimperc=0,
            cmap=cmap)
        # Save the figure if 'save' set to True 
        if save:
            fig.savefig(self.output_dir / f'{self.subject}_spectrogram.png')
        return fig

    def plot_psd_per_stage(
        self,
        electrode_name: str,
        nperseg: int = 4096,
        psd_range: tuple = (-60, 60),
        freq_range: tuple = (0, 40),
        xscale: str = 'linear',
        sleep_stages: tuple[str] = ('Wake', 'N1', 'N2', 'N3', 'REM'),
        axis=None,
        save=False
    ):
        """Plots PSDs for multiple sleep stages.
        """

        sampling_freq = self.mne_raw.info['sfreq']
        # win = sf*1024/250
        # Import data from the raw mne file.
        data = self.mne_raw.get_data([electrode_name], units='uV')[0]
        # Upsample hypnogram according to the data
        hypno_up = yasa.hypno_upsample_to_data(
            self.hypno, self.sf_hypno, data=self.mne_raw)
        signal_by_stage = {}
        if not axis:
            fig, axis = plt.subplots()
        # For every stage get its signal, 
        # calculate signal's PSD, 
        # transform PSD units to dB and plot it.
        for i, stage in enumerate(sleep_stages):
            signal_by_stage[stage] = np.take(data, np.where(hypno_up == i)[0])
            freqs, psd = signal.welch(
                signal_by_stage[stage], sampling_freq, nperseg=nperseg)
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

