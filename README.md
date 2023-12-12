# sleepeegpy
**sleepeegpy** is a high-level package built on top of [mne-python](https://mne.tools/stable/index.html), [yasa](https://raphaelvallat.com/yasa/build/html/index.html) for preprocessing, analysis and visualisation of sleep EEG data.
## Installation
0. Make sure you have [Python](https://www.python.org/downloads/) version installed. Requires Python 3.10 or higher.
1. Create a Python virtual environment, for more info you can refer to python [venv](https://docs.python.org/3/tutorial/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
2. Activate the environment
3. 
    ```
    pip install sleepeegpy
    ```
4. [Download](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/NirLab-TAU/sleepeegpy/tree/main/notebooks) notebooks.

## Quickstart
1. Open the complete pipeline notebook in the created environment.
2. Follow the notebook's instructions. 

## RAM requirements
For overnight, high density (256 channels) EEG recordings downsampled to 250 Hz expect at least 64 GB RAM expenditure for cleaning, spectral analyses and event detection.

## Retrieve example dataset
```
odie = pooch.create(
    path=pooch.os_cache("sleepeegpy_dataset"),
    base_url="doi:10.5281/zenodo.10362189",
)
odie.load_registry_from_doi()
bad_channels = odie.fetch("bad_channels.txt")
annotations = odie.fetch("annotations.txt")
path_to_eeg = odie.fetch("resampled_raw.fif")
for i in range(1,4):
    odie.fetch(f"resample_raw-{i}.fif")
```