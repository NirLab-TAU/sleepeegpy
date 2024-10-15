# sleepeegpy

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
1.  Navigate to the  Pipeline Notebooks folder and run Jupyter.
    ```bash
    jupyter notebook
    ```
2. Open the complete_pipeline notebook using Jupyter Notebook within the activated environment and follow the instructions.

## RAM requirements
For overnight, high-density (256 channels) EEG recordings downsampled to 250 Hz expect at least 64 GB RAM expenditure for cleaning, spectral analyses, and event detection.

## Citation
* Belonosov, G., Falach, R., Schmidig, J.F., Aderka, M., Zhelezniakov, V., Shani-Hershkovich, R., Bar, E., Nir, Y. "SleepEEGpy: a Python-based software “wrapper” package to organize preprocessing, analysis, and visualization of sleep EEG data." bioRxiv (2023). doi: https://doi.org/10.1101/2023.12.17.572046
* Belonosov, G., Falach, R., Schmidig, F., Aderka, M., Zhelezniakov, V., Shani-Hershkovich, R., Bar, E., & Nir, Y. (2024). SleepEEGpy: A Python-based package for preprocessing, analysis, and visualization of sleep EEG data [Dataset]. Zenodo. https://doi.org/10.5281/ZENODO.13903088
  
## Troubleshooting

<details>
<summary>Installation error on macOS - `libomp` not found</summary>

If you encounter the following error when installing `sleepeegpy` on macOS:

```bash
ERROR: Could not find a version that satisfies the requirement libomp (from versions: none)
ERROR: No matching distribution found for libomp
```
You can resolve this by running:
```bash
brew install cmake libomp
pip install lightgbm
pip install sleepeegpy
```
</details> 

