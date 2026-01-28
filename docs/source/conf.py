import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../los_estimator"))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LoS Estimator"
copyright = "2026, Younes Müller"
author = "Younes Müller"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
]

napoleon_google_docstring = True


templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = [
    "joblib",
    "numba",
    "numba.np",
    "numba.np.ufunc",
    "numba.np.ufunc.decorators",
    "numba.np.ufunc._internal",
    "numpy",
    "numpy._core",
    "numpy._core.multiarray",
    "pandas",
    "flask",
    "bokeh",
    "tqdm",
    "rich",
    "scipy",
    "scipy.signal",
    "scipy.optimize",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.animation",
    "seaborn",
    "statsmodels",
    "statsmodels.compat",
    "statsmodels.compat.pandas",
    "statsmodels.compat.patsy",
    "packaging",
    "packaging.version",
]
coverage_show_missing_items = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
