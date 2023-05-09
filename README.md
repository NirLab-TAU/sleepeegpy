# sleep-eeg-processing
**sleepeeg** is a high-level API built on top of [mne-python](https://mne.tools/stable/index.html) for preprocessing and analysis of sleep EEG data.
## Installation
0. Make sure you have [Python](https://www.python.org/downloads/) installed.
    - The script was tested with the version 3.10.9
    - Python 3.11.x currently isn't supported because of the yasa's dependency on the numba library.
1. Create a Python virtual environment, for more info you can refer to python [venv](https://docs.python.org/3/tutorial/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
2. Activate the environment
3. 
    ```
    pip install sleepeeg
    ```
4. (Optional, but recommended) [Download](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/NirLab-TAU/sleep-eeg-processing/tree/main/notebooks) notebooks.
5. (Optional, but recommended) Prepare [GPU acceleration](https://mne.tools/stable/install/advanced.html#gpu-acceleration-with-cuda) by installing [CUDA](https://developer.nvidia.com/cuda-downloads) and [CuPy](https://cupy.dev/). After installation, to permanently enable CUDA use, do: 
    ```
    mne.utils.set_config('MNE_USE_CUDA', 'true')
    ```
6. (Optional) For OpenGL acceleration of [MNE plot](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot) install [pyopengl](https://pyopengl.sourceforge.net/documentation/installation.html) and use `plot(use_opengl=True)`.

## Quickstart
1. Open any of the downloaded notebooks using the created environment.
2. Follow the notebook's instructions.