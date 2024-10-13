import os
import pytest
from sleepeegpy.pipeline import CleaningPipe  # Adjust this import to your actual module name

import numpy as np
import mne

def _basic_eeg_file_creation():
    sfreq = 250
    n_channels = 22
    duration = 10
    times = np.arange(0, duration, 1 / sfreq)

    data = np.zeros((n_channels, len(times)))
    for i in range(n_channels):
        alpha = 0.5 * np.sin(2 * np.pi * 10 * times + np.random.rand())
        beta = 0.3 * np.sin(2 * np.pi * 20 * times + np.random.rand())
        theta = 0.1 * np.sin(2 * np.pi * 4 * times + np.random.rand())
        noise = np.random.normal(0, 0.05, size=times.shape)
        data[i] = alpha + beta + theta + noise

    ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1',
                'FC2', 'Cz', 'C3', 'C4', 'T7', 'T8', 'Pz', 'P3',
                'P4', 'P7', 'P8', 'O1', 'O2', 'Iz']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    return mne.io.RawArray(data, info)

@pytest.fixture
def setup_eeg_file(tmp_path):
    raw = _basic_eeg_file_creation()
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    eeg_file_path = tmp_path / "test_eeg_file.fif"
    raw.save(eeg_file_path, overwrite=True)
    return eeg_file_path


@pytest.fixture
def setup_cleaning_pipe(setup_eeg_file, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaning_pipe = CleaningPipe(path_to_eeg=setup_eeg_file, output_dir=output_dir)
    return cleaning_pipe


def test_resample(setup_cleaning_pipe):
    cleaning_pipe = setup_cleaning_pipe
    assert cleaning_pipe.mne_raw.info["sfreq"] == 250
    cleaning_pipe.resample(sfreq=125)
    assert cleaning_pipe.mne_raw.info["sfreq"] == 125


def test_filter(setup_cleaning_pipe):
    cleaning_pipe = setup_cleaning_pipe
    original_data = cleaning_pipe.mne_raw.get_data()
    cleaning_pipe.filter(l_freq=1.0, h_freq=40.0)
    filtered_data = cleaning_pipe.mne_raw.get_data()
    assert original_data.shape == filtered_data.shape


def test_notch(setup_cleaning_pipe):
    cleaning_pipe = setup_cleaning_pipe
    cleaning_pipe.mne_raw = mne.io.read_raw_fif(cleaning_pipe.path_to_eeg, preload=True)
    original_data = cleaning_pipe.mne_raw.get_data()
    cleaning_pipe.notch(freqs='50s')
    filtered_data = cleaning_pipe.mne_raw.get_data()
    assert original_data.shape == filtered_data.shape


def test_auto_detect_bad_channels(setup_cleaning_pipe):
    cleaning_pipe = setup_cleaning_pipe
    bad_channels_file = cleaning_pipe.auto_detect_bad_channels()
    assert os.path.isfile(bad_channels_file)
    with open(bad_channels_file, 'r') as f:
        bad_channels = f.read().splitlines()
    assert isinstance(bad_channels, list)


def test_save_annotations(setup_cleaning_pipe):
    cleaning_pipe = setup_cleaning_pipe
    cleaning_pipe.mne_raw = mne.io.read_raw_fif(cleaning_pipe.path_to_eeg, preload=True)
    cleaning_pipe.mne_raw.annotations.append(onset=1, duration=1, description='bad')
    annotations_file_path = cleaning_pipe.output_dir /"CleaningPipe"/ "annotations.txt"
    cleaning_pipe.save_annotations(overwrite=True)
    assert os.path.isfile(annotations_file_path), f"File {annotations_file_path} not found."
    loaded_annotations = mne.read_annotations(annotations_file_path)
    assert len(loaded_annotations) > 0, "loaded_annotations is 0"

if __name__ == "__main__":
    pytest.main()
