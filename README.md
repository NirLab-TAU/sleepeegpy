# sleepeegpy
**sleepeegpy** is a high-level package built on top of [mne-python](https://mne.tools/stable/index.html), [yasa](https://raphaelvallat.com/yasa/build/html/index.html) for preprocessing and analysis of sleep EEG data.
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
1. Open any of the downloaded notebooks using the created environment.
2. Follow the notebook's instructions.

## RAM requirements
For overnight, high density (256 channels) EEG recordings downsampled to 250 Hz expect at least 64 GB RAM expenditure for cleaning and spectral analyses, and at least 128 GB for event detection (or 64 GB if you downsample the data to 100 Hz before running the detection algorithm).