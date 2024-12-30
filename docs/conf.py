# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import inspect
import shutil
from pathlib import Path
sys.path.insert(0, os.path.abspath('./notebooks'))
sys.path.append(os.path.abspath(".."))

project = "sleepeegpy"
copyright = "2023, Yuval Nir lab"
author = "Gennadiy Belonosov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
    "myst_nb",
    "nbsphinx",
]
html_logo = "sleepEEGpy.png"

rst_epilog = """
.. |AbstractClass| replace:: 🚨 **Abstract Class**  - 
   This class is **abstract** and cannot be used directly.  
   Subclass it and implement the required methods.

.. |ExamplePipe| replace:: 💡 **Usage Examples**  - 
   See the notebook for usage examples and code snippets:

.. |CleaningPipe| replace:: |ExamplePipe| :doc:`/notebooks/1_cleaning`
.. |ICAPipe| replace:: |ExamplePipe| :doc:`/notebooks/2_ica`
.. |SpectralPipe| replace:: |ExamplePipe| :doc:`/notebooks/4_spectral`
.. |GrandSpectralPipe| replace:: |ExamplePipe| :doc:`/notebooks/5_grand spectral`
.. |SpindlesPipe| replace:: |ExamplePipe| :doc:`/notebooks/6_spindles`
.. |SlowWavesPipe| replace:: |ExamplePipe| :doc:`/notebooks/7_slow_waves`
.. |RapidEyeMovementsPipe| replace:: |ExamplePipe| :doc:`/notebooks/8_rems`

.. |BasePipe| replace:: |AbstractClass|
.. |BaseEventPipe| replace:: |AbstractClass|
.. |BaseHypnoPipe| replace:: |AbstractClass|
.. |SpectrumPlots| replace:: |AbstractClass|
"""

nb_execution_mode = "off"
# autosummary_generate = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
# autoclass_content = "class"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

intersphinx_mapping = {
    "mne": ("https://mne.tools/stable/", None),
    "yasa": ("https://raphaelvallat.com/yasa/build/html/", None),
    "mpl": ("https://matplotlib.org/stable/", None),
    "fooof": ("https://fooof-tools.github.io/fooof/", None),
}

intersphinx_disabled_reftypes = ["*"]


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object.

    Adapted from MNE (doc/source/conf.py).
    """

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""
    return f"https://github.com/NirLab-TAU/sleepeegpy/blob/main/{info['module'].replace('.', '/')}.py{linespec}"


def all_but_ipynb(dir, contents):
    return [c for c in contents if (Path(dir) / c).suffix != ".ipynb"]


print("Copy example notebooks into docs/notebooks")
project_root = Path("../")
shutil.rmtree(project_root / "docs/notebooks", ignore_errors=True)
shutil.copytree(
    project_root / "notebooks",
    project_root / "docs/notebooks",
    ignore=all_but_ipynb,
)
