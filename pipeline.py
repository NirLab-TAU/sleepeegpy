""" Pipeline Module
"""
from attrs import define, field
from pathlib import Path
from typing import TypeVar, Type
from functools import cached_property
import os
import errno
import matplotlib.pyplot as plt
import numpy as np
import mne.io

_SuperPipeType = TypeVar('_SuperPipeType', bound='_SuperPipe')

@define(kw_only=True)
class _SuperPipe:
    """The template pipeline element"""


    # Preceding pipe that hands over mne_raw attr.
    prec_pipe: Type[_SuperPipeType] = field(default=None)  

    path_to_eeg: Path = field(converter=Path)
    @path_to_eeg.default
    def _set_path_to_eeg(self):
        if self.prec_pipe: return '/'
        raise TypeError('Provide either "pipe" or "path_to_eeg" arguments')
    @path_to_eeg.validator
    def _validate_path_to_eeg(self, attr, value):
        if not value.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), value)
        
    output_dir: Path = field(converter=Path)
    @output_dir.default
    def _set_output_dir(self):
        return self.prec_pipe.output_dir if self.prec_pipe else self.path_to_eeg.parents[0]
    @output_dir.validator
    def _validate_output_dir(self, attr, value):
        self.output_dir.mkdir(exist_ok=True)

    mne_raw: mne.io.Raw = field(init=False)
    @mne_raw.default
    def _read_mne_raw(self):
        from mne.io import read_raw
        try:
            return self.prec_pipe.mne_raw if self.prec_pipe else read_raw(self.path_to_eeg)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


    def plot(
        self, 
        butterfly=False, 
        save_annotations=False, 
        save_bad_channels=False, 
        scalings={'eeg':100e-6, 'eog':100e-6, 'ecg':1000e-6, 'emg':100e-6, 'resp':5e-6, 'bio':10e-6},
        use_opengl=None):

        from mne import pick_types

        order = pick_types(self.mne_raw.info, eeg=True) if butterfly else None
        self.mne_raw.plot( 
            theme='dark',
            block=True, 
            scalings=scalings,
            bad_color='r',
            proj=False,
            order=order,
            butterfly=butterfly,
            use_opengl=use_opengl)
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


@define(kw_only=True)
class CleaningPipe(_SuperPipe):
    """The cleaning pipeline element"""


    def resample(self, sfreq=250, n_jobs='cuda', save=False):
        """ Resamples and updates the data """
        self.mne_raw.resample(sfreq, n_jobs=n_jobs, verbose='WARNING')
        if save:
            self._save_raw('_'.join(filter(None, ['resampled', str(sfreq)+'hz', 'raw.fif'])))


    def filter(self, l_freq=0.3, h_freq=None, picks=None, n_jobs='cuda'):

        self.mne_raw.load_data()
        self.mne_raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, n_jobs=n_jobs)


    def notch(self):
        self.mne_raw.notch_filter(
            freqs=np.arange(50, int(self.sf/2), 50),
            picks='eeg',
            n_jobs='cuda'
        )


    def read_bad_channels(self, path=None):
        p = Path(path) if path else self.output_dir / 'bad_channels.txt'
        with open(p, 'r') as f:
            self.mne_raw.info['bads'] = list(filter(None, f.read().split('\n')))


    def read_annotations(self, path=None):
        from mne import read_annotations
        p = Path(path) if path else self.output_dir / 'annotations.txt'
        self.mne_raw.set_annotations(read_annotations(p))


@define(kw_only=True)
class ICAPipe(_SuperPipe):

    from mne.preprocessing import ICA

    n_components: int = field(default=15)
    method: str = field(default='fastica')
    fit_params: dict = field(default=None)
    mne_ica: ICA = field()
    @mne_ica.default
    def _set_mne_ica(self):
        from mne.preprocessing import ICA
        return ICA(
            n_components=self.n_components,
            method=self.method,
            fit_params=self.fit_params)

    def __attrs_post_init__(self):
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


@define(kw_only=True)
class ResultsPipe(_SuperPipe):

    import numpy as np

    path_to_hypno: Path = field(converter=Path)
    @path_to_hypno.validator
    def _validate_path_to_hypno(self, attr, value):
        if not value.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), value)
    hypno_freq: int = field(converter=int, default=1)
    hypno: np.array = field()
    @hypno.default
    def _import_hypno(self):
        try:
            return np.loadtxt(self.path_to_hypno)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
    hypno_up: np.array = field()
    @hypno_up.default
    def _set_hypno_up(self):
        return self.hypno
    psd_per_stage: np.array = field(init=False)
    def __attrs_post_init__(self):
        self.__upsample_hypno()

    
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

        is_axis = False
        
        if not axis:
            fig, axis = plt.subplots()
        else:
            is_axis = True

        psd_per_stage = self.__compute_psd_per_stage(
            picks=picks, 
            sleep_stages=sleep_stages, 
            sec_per_seg=sec_per_seg, 
            avg_ref=False, 
            dB=True)

        for stage in sleep_stages:
            axis.plot(
                psd_per_stage[stage][0], 
                psd_per_stage[stage][1][0], 
                label=f'{stage} ({psd_per_stage[stage][2]}%)')

        axis.set_xlim(freq_range)
        axis.set_ylim(psd_range)
        axis.set_xscale(xscale)
        axis.set_title("Welch's PSD")
        axis.set_ylabel('PSD [dB/Hz]')
        axis.set_xlabel(f'{xscale} frequency [Hz]'.capitalize())
        axis.legend()
        # Save the figure if 'save' set to True and no axis has been passed.
        if save and not is_axis:
            fig.savefig(self.output_dir / f'psd.png')


    def plot_topomap(
        self, 
        stage: str = 'Wake', 
        bandwidth: tuple = (0, 4), 
        sec_per_seg: float = 4.096, 
        dB: bool = False, 
        sleep_stages: dict = {'Wake' :0, 'N1' :1, 'N2': 2, 'N3': 3, 'REM': 4},
        axis: plt.axis = None,
        save=False
    ):

        from mne.viz import plot_topomap

        is_axis = False
        cmap='plasma'
        
        if axis is None:
            fig, axis = plt.subplots()
            is_axis = True

        if not hasattr(self, 'psd_per_stage'):
            self.psd_per_stage = self.__compute_psd_per_stage(
                picks=['eeg'], 
                sleep_stages=sleep_stages, 
                sec_per_seg=sec_per_seg, 
                avg_ref=True, 
                dB=dB)
            
        psds = np.take(
            self.psd_per_stage[stage][1],
            np.where(np.logical_and(self.psd_per_stage[stage][0]>=bandwidth[0], self.psd_per_stage[stage][0]<=bandwidth[1]))[0],
            axis=1).sum(axis=1)
        
        im, cn = plot_topomap(
            psds, 
            self.mne_raw.info,
            size=5, 
            cmap=cmap,
            axes=axis,
            show=False)
        
        # divider = make_axes_locatable(axis)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(
            im, 
            ax=axis, 
            orientation='vertical',
            shrink=0.6,
            label='dB/Hz' if dB else r'$\mu V^{2}/Hz$')

        if is_axis:
            fig.suptitle(f'{stage} ({bandwidth[0]}-{bandwidth[1]} Hz)')
        if save and not is_axis:
            fig.savefig(self.output_dir / f'topomap.png')
        
    

    def plot_topomap_collage(
        self,
        bands: dict = {'Delta': (0, 3.99), 'Theta': (4, 7.99),
            'Alpha': (8, 12.49), 'SMR': (12.5, 15), 
            'Beta': (12.5, 29.99), 'Gamma': (30, 60)}, 
        sec_per_seg: float = 4.096, 
        dB: bool = False, 
        sleep_stages: dict = {'Wake' :0, 'N1' :1, 'N2': 2, 'N3': 3, 'REM': 4},
        stages_to_plot: tuple = None,
        save=False
    ):
        
        if not stages_to_plot:
            stages_to_plot = sleep_stages.keys()
        n_rows = len(stages_to_plot)
        n_cols = len(bands)

        fig = plt.figure(figsize=(n_cols*4, n_rows*4), layout='constrained')
        subfigs = fig.subfigures(n_rows, 1)
        
        for row_index, stage in enumerate(stages_to_plot):
            axes = subfigs[row_index].subplots(1, n_cols)

            for col_index, band_key in enumerate(bands):
                self.plot_topomap(
                    stage=stage, 
                    bandwidth=bands[band_key], 
                    sec_per_seg=sec_per_seg, 
                    dB=dB, 
                    sleep_stages=sleep_stages,
                    axis=axes[col_index],
                )
                axes[col_index].set_title(f'{band_key} ({bands[band_key][0]}-{bands[band_key][1]} Hz)')

            subfigs[row_index].suptitle(f'{stage} ({self.psd_per_stage[stage][2]}%)', fontsize='xx-large')

        if save:
            fig.savefig(self.output_dir / f'topomap_collage.png')
        

    def __upsample_hypno(self):

        from yasa import hypno_upsample_to_data
        self.hypno_up = hypno_upsample_to_data(
                self.hypno, 
                self.hypno_freq,
                self.mne_raw,
                verbose=False)

    def __compute_psd_per_stage(
        self, 
        picks, 
        sleep_stages, 
        sec_per_seg, 
        avg_ref, 
        dB):
        
        from scipy import signal
        # Import data from the raw mne file.
        self.mne_raw.load_data()
        if avg_ref:
            data = self.mne_raw.copy().set_eeg_reference().get_data(picks=picks, units='uV', reject_by_annotation='NaN')
        else:
            data = self.mne_raw.get_data(picks=picks, units='uV', reject_by_annotation='NaN')
        data = np.ma.array(data, mask=np.isnan(data))
        signal_per_stage = {}
        psd_per_stage = {}
        for stage, index in sleep_stages.items():
            signal_per_stage[stage] = np.take(
                data, 
                np.where(np.in1d(self.hypno_up, index))[0], 
                axis=1)
            freqs, psd = signal.welch(
                np.ma.compress_cols(signal_per_stage[stage]), 
                self.sf, 
                nperseg=self.sf*sec_per_seg,
                axis=1)
            if dB:
                psd = 10 * np.log10(psd)
            psd_per_stage[stage] = [freqs, psd, round(np.ma.compress_cols(signal_per_stage[stage]).shape[1]/np.ma.compress_cols(data).shape[1]*100, 2)]
        return psd_per_stage



    def __plot_hypnospectrogram(
        self,
        data,
        sf,
        hypno,
        win_sec,
        fmin,
        fmax,
        trimperc,
        cmap,
        overlap
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
        stats = sleep_statistics(self.hypno, self.hypno_freq)
        if save_to_csv:
            with open(self.output_dir/'sleep_stats.csv', 'w', newline='') as csv_file:
                w = DictWriter(csv_file, stats.keys())
                w.writeheader()
                w.writerow(stats)
            return
        return stats

