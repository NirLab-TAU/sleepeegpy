# sleepeegpy

**sleepeegpy** is a high-level package built on top of [MNE-python](https://mne.tools/stable/index.html), [yasa](https://raphaelvallat.com/yasa/build/html/index.html) and [specparam (fooof)](https://fooof-tools.github.io/fooof/) for preprocessing, analysis, and visualization of sleep EEG data.

The repository also includes a Jupyter notebook that demonstrates how to use this package and provides a ready-made workflow for common use cases.
![image](https://github.com/user-attachments/assets/f26c2023-44fc-48d7-ba72-d0de89a5dcee)

## Installation
### Prerequisites
- **Python Version**: Ensure you have [Python](https://www.python.org/downloads/) version >3.9 and <3.12 installed.
### Steps
1. **Create a Python Virtual Environment**:
   Create a Python virtual environment. For more info you can refer to python [venv](https://docs.python.org/3/tutorial/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
2. **Activate the Environment**:
3. **Install `sleepeegpy`**:
   ```bash
   pip install sleepeegpy
4. **Download notebooks**: [Download](https://github.com/NirLab-TAU/sleepeegpy/archive/refs/heads/main.zip) this repository zip folder, you will need only the notebooks folder.

## Quickstart
1. Install [Jupyter](https://jupyter.org/install) notebook:

   Run the following command to install JupyterLab:
   ```bash
   pip install jupyterlab
3.  Navigate to the  Pipeline Notebooks folder and run jupyter.
    ```bash
    jupyter notebook
    ```
4. Open the complete_pipeline notebook using Jupyter Notebook within the activated environment and follow the instructions.

## RAM requirements
For overnight, high-density (256 channels) EEG recordings downsampled to 250 Hz expect at least 64 GB RAM expenditure for cleaning, spectral analyses, and event detection.

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
    odie.fetch(f"resampled_raw-{i}.fif")
```
