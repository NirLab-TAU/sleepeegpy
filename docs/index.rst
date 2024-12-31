.. sleepeegpy documentation master file, created by
   sphinx-quickstart on Mon Mar 13 05:36:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sleepeegpy
==========

**sleepeegpy** is a high-level package built on top of several powerful libraries, including:

- `MNE-python <https://mne.tools/stable/index.html>`_ for electrophysiological data analysis
- `yasa <https://raphaelvallat.com/yasa/build/html/index.html>`_ for sleep staging and analysis
- `PyPREP <https://pyprep.readthedocs.io/en/latest/>`_ for preprocessing EEG data
- `specparam (fooof) <https://fooof-tools.github.io/fooof/>`_ for spectral analysis and parameter estimation

This package is designed to streamline the preprocessing, analysis, and visualization of sleep EEG data.

Additionally, the repository includes a Jupyter notebook demonstrating how to use this package, along with a ready-made workflow for common use cases.

.. image:: https://github.com/user-attachments/assets/f26c2023-44fc-48d7-ba72-d0de89a5dcee
   :alt: sleepeegpy workflow

API Reference
===============
.. toctree::
   :maxdepth: 1

   notebooks
   api


Installation
============

Prerequisites
-----
Before installing, ensure that you have a compatible version of Python installed:

- **Python Version**: `Python <https://www.python.org/downloads/>`_ version >3.9 and <3.12.

Steps
-----
Follow these steps to install and set up **sleepeegpy**:

1. **Create a Python Virtual Environment**:
   First, create a Python virtual environment. For more information, refer to:

   - `venv <https://docs.python.org/3/tutorial/venv.html>`
   - `virtualenv <https://virtualenv.pypa.io/en/latest/user_guide.html>`
   - `conda <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`

2. **Activate the Environment**

3. **Install sleepeegpy**:
   Run the following command to install **sleepeegpy**:

   .. code-block:: bash
      pip install sleepeegpy

Usage Example
=============

The included notebooks are a great way to get started with **sleepeegpy**. They provide examples and instructions for using the package in common workflows.

Steps to Use the Notebooks
------------------
1. **Download the Notebooks**:
   Download the zip folder of this repository by clicking `Download <https://github.com/NirLab-TAU/sleepeegpy/archive/refs/heads/main.zip>`_. You will only need the notebooks folder.

2. **Run Jupyter**:
   Navigate to the "Pipeline Notebooks" folder and start Jupyter:

   .. code-block:: bash
      jupyter notebook

3. **Open the Complete Pipeline Notebook**:
   Open the `complete_pipeline` notebook using Jupyter Notebook within the activated environment and follow the instructions provided in the notebook.

RAM Requirements
================

For high-density EEG recordings (256 channels) downsampled to 250 Hz, you will need at least 64 GB of RAM for tasks such as data cleaning, spectral analysis, and event detection.

Citation
========

To cite **sleepeegpy** in your research, please use the following references:

1. Belonosov, G., Falach, R., Schmidig, J.F., Aderka, M., Zhelezniakov, V., Shani-Hershkovich, R., Bar, E., Nir, Y. "SleepEEGpy: A Python-based software package to organize preprocessing, analysis, and visualization of sleep EEG data." bioRxiv (2023). doi: `https://doi.org/10.1101/2023.12.17.572046 <https://doi.org/10.1101/2023.12.17.572046>`_

2. Belonosov, G., Falach, R., Schmidig, F., Aderka, M., Zhelezniakov, V., Shani-Hershkovich, R., Bar, E., & Nir, Y. (2024). SleepEEGpy: A Python-based package for preprocessing, analysis, and visualization of sleep EEG data [Dataset]. Zenodo. `https://doi.org/10.5281/ZENODO.13903088 <https://doi.org/10.5281/ZENODO.13903088>`_

