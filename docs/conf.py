# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pyabsa import __version__ as pyabsa_version

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
project = "PyABSA"
copyright = "2020, Heng Yang"
author = "Heng Yang"

# The full version, including alpha/beta/rc tags
release = pyabsa_version

# Set master doc to `index.rst`.
master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    # Enable .ipynb doc files
    "nbsphinx",
    # Enable .md doc files
    "recommonmark",
    "sphinx_markdown_tables",
    "IPython.sphinxext.ipython_console_highlighting",
]
autosummary_generate = True

autoapi_type = 'python'
autoapi_dirs = ['../pyabsa']

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Mock expensive textattack imports. Docs imports are in `docs/requirements.txt`.
autodoc_mock_imports = []

# Output file base name for HTML help builder.
htmlhelp_basename = "PyABASA_documentation"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

latex_engine = 'xelatex'
latex_use_xindy = False
latex_elements = {
    'preamble': '\\usepackage[UTF8]{ctex}\n',
}

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 5,
    "logo_only": False,
    "style_nav_header_background": "transparent",
    "analytics_id": "G-TC6R5H0R74",
}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

# Path to favicon.
html_favicon = "favicon.png"

# Don't show module names in front of class names.
add_module_names = True

# Sort members by group
autodoc_member_order = "groupwise"

# -- Options for Sphinx Copy Button-------------------------------------------------

# Exclude the prompt symbol ">>>" when copying text
copybutton_prompt_text = ">>> "

# Specify Line Continuation Character so that all the entire Line is copied
copybutton_line_continuation_character = "\\"
