# eeg-processing-pipeline
**eeg-processing-pipeline** is a high-level API built on top of [mne-python](https://mne.tools/stable/index.html) for processing raw eeg data recorded by high density (256 electrodes) EGI net.
## Installation
0. Make sure you have [Python](https://www.python.org/downloads/) installed.
    - The script was tested with the version 3.10.9
    - Python 3.11.x currently isn't supported because of the yasa's dependency on the numba library.
1. Clone the GitHub repository
2. Navigate to the cloned repo
3. Create a python environment
4. Activate the environment
5. Install requirements:
```
pip install -r requirements.txt
```
## How to run resample CLI
0. Make sure the environment is activated
1. Run the resample CLI:
```
py resample_cli.py path/to/mff
```
2. The resampled files will appear at the mff's parent directory
3. For more options: `py resample_cli.py --help`