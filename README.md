<p align="center">
  <img src="logo.png" alt="Logo" style="width:50%;">
</p>

**sleepeegpy** is a high-level package built on top of [MNE-python](https://mne.tools/stable/index.html), [yasa](https://raphaelvallat.com/yasa/build/html/index.html), [PyPREP](https://pyprep.readthedocs.io/en/latest/) and [specparam (fooof)](https://fooof-tools.github.io/fooof/) for preprocessing, analysis, and visualization of sleep EEG data.

The repository also includes a Jupyter notebook demonstrating how to use this package and provides a ready-made workflow for common use cases.
![image](https://github.com/user-attachments/assets/f26c2023-44fc-48d7-ba72-d0de89a5dcee)

## Installation
### Prerequisites
- **Python Version**: Ensure you have [Python](https://www.python.org/downloads/) version >3.9 and <3.12 installed.
### Steps
1. **Create a Python Virtual Environment**:
   Create a Python virtual environment. For more information you can refer to python [venv](https://docs.python.org/3/tutorial/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
2. **Activate the Environment**
3. **Install sleepeegpy**:
   ```bash
   pip install sleepeegpy
4. **Download notebooks**: [Download](https://github.com/NirLab-TAU/sleepeegpy/archive/refs/heads/main.zip) this repository zip folder, you will need only the notebooks folder.

## Quickstart
The notebooks are useful for familiarizing yourself with the library's functionalities. To use them:
1.  Navigate to the  Pipeline Notebooks folder and run Jupyter.
    ```bash
    jupyter notebook
    ```
2. Open the complete_pipeline notebook using Jupyter Notebook within the activated environment and follow the instructions.

Additionally, detailed [documentation](https://nirlab-tau.github.io/sleepeegpy/) is available for further reference.
## RAM requirements
For overnight, high-density (256 channels) EEG recordings downsampled to 250 Hz expect at least 64 GB RAM expenditure for cleaning, spectral analyses, and event detection.

## Citation
- **Paper**\
  Falach, R., G. Belonosov, J. F. Schmidig, M. Aderka, V. Zhelezniakov, R. Shani-Hershkovich, E. Bar, and Y. Nir. "SleepEEGpy: a Python-based software integration package to organize preprocessing, analysis, and visualization of sleep EEG data." Computers in Biology and Medicine 192 (2025): 110232.\
  https://doi.org/10.1016/j.compbiomed.2025.110232
- **Dataset**\
  https://doi.org/10.5281/ZENODO.10362189.
