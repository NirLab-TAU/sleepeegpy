""" Pipeline Module
"""
import matplotlib.pyplot as plt
import numpy as np

class _SuperPipe:
    """The template pipeline element"""

    def __init__(
        self,
        path_to_eeg: str = None,
        output_directory: str = None,
        pipe = None
    ):
        """Constructs an instanse of the class.
        """

        if pipe:
            self.mne_raw = pipe.mne_raw
            self.output_dir = pipe.output_dir
            return
        elif not path_to_eeg:
            raise TypeError('Provide either pipe object or path to eeg file')

        import os
        import errno
        from pathlib import Path
        from mne.io import read_raw
        # Try to import eeg file, raise exception if something's wrong.
        eeg_file = Path(path_to_eeg)
        try:
            self.mne_raw = read_raw(eeg_file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                path_to_eeg) from exc
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
        
        if not output_directory:
            self.output_dir = eeg_file.parents[0]
        else:
            self.output_dir = Path(output_directory)
            self.output_dir.mkdir(exist_ok=True)


    def plot(
        self, 
        butterfly=False, 
        save_annotations=False, 
        save_bad_channels=False, 
        scalings={'eeg':100e-6, 'eog':100e-6, 'ecg':1000e-6, 'emg':100e-6, 'resp':5e-6, 'bio':10e-6}):

        from mne import pick_types

        order = pick_types(self.mne_raw.info, eeg=True) if butterfly else None
        self.mne_raw.plot( 
            theme='dark',
            block=True, 
            scalings=scalings,
            bad_color='r',
            proj=False,
            order=order,
            butterfly=butterfly)
        if save_annotations:
            self.mne_raw.annotations.save(self.output_dir/'annotations.txt', overwrite=True)
        if save_bad_channels:
            with open(self.output_dir / 'bad_channels.txt', 'w') as f:
                for bad in self.mne_raw.info['bads']:
                    f.write(f"{bad}\n")
    
    
    @property
    def sf(self):
        return self.mne_raw.info['sfreq']


    def _save_raw(self, fname):
        path_to_resampled = self.output_dir/f'saved_raw'
        path_to_resampled.mkdir(exist_ok=True)
        self.mne_raw.save(path_to_resampled / fname)

class CleaningPipe(_SuperPipe):
    """The cleaning pipeline element"""


    def resample(self, sfreq=250, n_jobs='cuda', save=False):
        """ Resamples and updates the data """
        self.mne_raw.resample(sfreq, n_jobs=n_jobs, verbose='WARNING')
        if save:
            self._save_raw('_'.join(filter(None, ['resampled', str(sfreq)+'hz', 'raw.fif'])))


    def filter(self, l_freq=0.3, h_freq=None, picks=None, n_jobs='cuda', savefig=False):

        from mne.filter import create_filter
        from mne.viz import plot_filter

        self.mne_raw.load_data()
        self.mne_raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, n_jobs=n_jobs)
        sf = self.sf
        data = self.mne_raw.get_data(picks)[0]
        _, axes = plt.subplots(3, 1, figsize=(10, 10))
        h_freq = sf/2-0.1 if not h_freq else h_freq
        l_freq = 0 if not l_freq else l_freq
        ideal_freq = [0, l_freq, l_freq, h_freq, h_freq, sf/2]
        ideal_gain = [0, 0, 1, 1, 0, 0]
        filt = create_filter(data, sf, l_freq=l_freq, h_freq=h_freq, phase='zero-double', method='fir')
        fig = plot_filter(filt, sf, ideal_freq, ideal_gain, flim=(0.001, sf/2), compensate=True, axes=axes)
        if savefig:
            fig.savefig(self.output_dir / 'filter.png')


    def notch(self):
        self.mne_raw.notch_filter(
            freqs=np.arange(50, int(self.sf/2), 50),
            picks='eeg',
            n_jobs='cuda'
        )


    def read_bad_channels(self):
        with open(self.output_dir / 'bad_channels.txt', 'r') as f:
            self.mne_raw.info['bads'] = list(filter(None, f.read().split('\n')))


    def read_annotations(self):
        from mne import read_annotations
        self.mne_raw.set_annotations(read_annotations(self.output_dir / 'annotations.txt'))



class ICAPipe(_SuperPipe):

    def __init__(
        self, 
        path_to_eeg: str = None, 
        output_directory: str = None, 
        n_components: int = None, 
        method='fastica', 
        fit_params=None,
        pipe = None
    ):

        from mne.preprocessing import ICA
        super().__init__(path_to_eeg=path_to_eeg, output_directory=output_directory, pipe=pipe)
        self.mne_ica = ICA(
            n_components=n_components,
            method=method,
            fit_params=fit_params
            )
        self.mne_raw.load_data()


    def fit(self):
        
        if self.mne_raw.info['highpass'] < 1.0:
            filtered_raw = self.mne_raw.copy()
            filtered_raw.filter(l_freq=1.0, h_freq=None, n_jobs='cuda')
        else:
            filtered_raw = self.mne_raw
        self.mne_ica.fit(filtered_raw)


    def plot_sources(self):
        self.mne_ica.plot_sources(self.mne_raw, block=True)

    
    def plot_components(self):
        self.mne_ica.plot_components(inst=self.mne_raw)


    def plot_overlay(self, exclude=None, picks=None):
        self.mne_ica.plot_overlay(self.mne_raw, exclude=exclude, picks=picks)


    def plot_properties(self, picks=None):
        self.mne_ica.plot_properties(self.mne_raw, picks=picks)


    def apply(self, exclude=None):
        self.mne_ica.apply(self.mne_raw, exclude=exclude)


class ResultsPipe(_SuperPipe):
    def __init__(
        self,
        path_to_eeg: str = None,
        output_directory: str = None,
        subject_code: str = None,
        path_to_hypno: str = None,
        sf_hypno: int = 1,
        pipe = None
    ):
        """Constructs an instanse of the class.
        """
        
        import os
        import errno
        from pathlib import Path

        super().__init__(
            path_to_eeg=path_to_eeg,
            output_directory=output_directory,
            pipe=pipe)        
        
        self.subject = subject_code
        
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


    
    def plot_hypnospectrogram(
        self,
        picks: str = ('E101',),
        win_sec: float = 4,
        freq_range: tuple = (0, 40),
        cmap: str = 'inferno',
        overlap=False,
        save: bool = False
    ):
        """ Plots yasa's hypnogram and spectrogram.
        """
        # Import data from the raw mne file.
        data = self.mne_raw.get_data(picks, units="uV", reject_by_annotation='NaN')[0]
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
            fig.savefig(self.output_dir / f'spectrogram.png', bbox_inches = "tight")


    def plot_psd_per_stage(
        self,
        picks: str = ('E101',),
        sec_per_seg: float = 4.096,
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

        isaxis = False
        
        if not axis:
            fig, axis = plt.subplots()
        else:
            isaxis = True

        # win = sf*1024/250
        # Import data from the raw mne file.
        data = self.mne_raw.get_data(picks, units='uV', reject_by_annotation='NaN')[0]
        data = np.ma.array(data, mask=np.isnan(data))
        signal_by_stage = {}

        # For every stage get its signal, 
        # calculate signal's PSD, 
        # transform PSD units to dB,
        # plot it.
        for stage, index in sleep_stages.items():
            signal_by_stage[stage] = np.take(data, np.where(np.in1d(self.hypno_up, index))[0])
            freqs, psd = signal.welch(
                signal_by_stage[stage].compressed(), self.sf, nperseg=self.sf*sec_per_seg)
            psd = 10 * np.log10(psd)
            axis.plot(freqs, psd, label=f'{stage} {int(len(signal_by_stage[stage])/len(data)*100)}%')

        axis.set_xlim(freq_range)
        axis.set_ylim(psd_range)
        axis.set_xscale(xscale)
        axis.set_title("Welch's PSD")
        axis.set_ylabel('PSD [dB/Hz]')
        axis.set_xlabel(f'{xscale} frequency [Hz]'.capitalize())
        axis.legend()
        # Save the figure if 'save' set to True and no axis has been passed.
        if save and not isaxis:
            fig.savefig(self.output_dir / f'psd.png')


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
        overlap=False
    ):
        """
        ?
        """
        # Increase font size while preserving original
        old_fontsize = plt.rcParams["font.size"]
        plt.rcParams.update({"font.size": 18})

        if overlap or not hypno.any():
            fig, ax = plt.subplots(nrows=1, figsize=(12, 4))
            im = self.__plot_spectrogram(data, sf, win_sec, fmin, fmax, trimperc, cmap, ax)
            if hypno.any():
                ax_hypno = ax.twinx()
                self.__plot_hypnogram(sf, hypno, ax_hypno)
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.95, fraction=0.1, aspect=25, pad=0.1)
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
        Sxx = 10 * np.log10(Sxx, out=np.full(Sxx.shape, np.nan), where=(Sxx!=0))  # Convert uV^2 / Hz --> dB / Hz

        # Select only relevant frequencies (up to 30 Hz)
        good_freqs = np.logical_and(f >= fmin, f <= fmax)
        Sxx = Sxx[good_freqs, :]
        f = f[good_freqs]
        t /= 3600  # Convert t to hours
        
        # Normalization
        vmin, vmax = np.nanpercentile(Sxx, [0 + trimperc, 100 - trimperc])
        norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto")
        ax.set_xlim(0, t.max())
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [hrs]")
        return im


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
