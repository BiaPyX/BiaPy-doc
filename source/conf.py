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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sys
import os
import datetime

# Clone
from git import Repo
dir_path = os.path.abspath(os.path.dirname(__file__))
biapy_dir = os.path.join(dir_path, "..","..","..", "BiaPy")
if not os.path.exists(biapy_dir):
    os.makedirs(biapy_dir, exist_ok=True)
    print("Cloning BiaPy repo . . .")
    Repo.clone_from("https://github.com/BiaPyX/BiaPy", biapy_dir)

base_path = os.path.abspath(biapy_dir)
sys.path.insert(0, base_path)
print("PATH: {}".format(sys.path))
numpydoc_show_class_members = False


# -- Project information -----------------------------------------------------

project = u'BiaPy'
copyright = u'%d, BiaPy Team' % (datetime.datetime.now().year,)
author = u'Daniel Franco-Barranco'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx_tabs.tabs',
    'sphinx_toolbox.collapse',
    "sphinx_carousel.carousel",
    'numpydoc',
    "sphinxcontrib.googleanalytics",
]

googleanalytics_id = "G-N5K8QZNCY2"
sphinx_tabs_disable_tab_closing = True

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

imgmath_image_format = 'svg'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = 'friendly'

bibtex_bibfiles = ['refs.bib']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_favicon = 'img/biapy_logo_icon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [] #['_static']

html_show_sourcelink = False

autodoc_member_order = 'bysource'

# The master toctree document.
master_doc = 'index'

