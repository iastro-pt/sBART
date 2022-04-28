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

# -- Project information -----------------------------------------------------

project = 'sBART'
copyright = '2022, André'
author = 'André'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "nbsphinx"
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Autodoc only inserts the class name
add_module_names = False

# Concatenates classes docstrings with the ones from the __init__
autoclass_content = 'class'
autodoc_class_signature = "separated"

html_theme_options = {
    'navigation_depth': 5,
}
# autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    "private-members":False,
    # 'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__',
    "show-inheritance": True
}

numpydoc_show_class_members=False

def setup(app):
    # https://stackoverflow.com/a/43186995
    app.add_css_file('my_theme.css')
