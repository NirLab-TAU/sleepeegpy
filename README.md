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
6. (Optional, but recommended) Prepare [GPU acceleration](https://mne.tools/stable/install/advanced.html#gpu-acceleration-with-cuda) by installing [CUDA](https://developer.nvidia.com/cuda-downloads) and [CuPy](https://cupy.dev/).
7. (Optional) For OpenGL acceleration of [MNE plot](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot) install [pyopengl](https://pyopengl.sourceforge.net/documentation/installation.html) and use `plot(use_opengl=True)`.