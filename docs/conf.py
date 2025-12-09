# docs/conf.py
# Sphinx configuration for forge3d documentation build
# Exists so docs match current release metadata and extensions
# RELEVANT FILES:docs/index.rst,pyproject.toml,CMakeLists.txt,CHANGELOG.md
# Configuration file for the Sphinx documentation builder.

import sys
import os

# Add Python source to path for autodoc
sys.path.insert(0, os.path.abspath('../python'))

project = 'forge3d'
copyright = '2025, forge3d contributors'
author = 'forge3d contributors'

# The short X.Y version
version = '0.94.0'
# The full version, including alpha/beta/rc tags
release = '0.94.0'

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'myst_parser',  # For Markdown support
]

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': None,
}

# Napoleon settings for Google/NumPy docstring styles
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Intersphinx mapping for cross-referencing external libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Todo extension settings
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    '**/*.pyc',
    '__pycache__',
    'task.xml',
    'task-gpt.txt',
    'PLAN.json',
    'REPORT.md',
    'QUESTIONS.md',
    'CLAUDE_UPDATE_REPORT.md',
    # Prefer Markdown versions of these docs
    'interop_zero_copy.rst',
    'memory_budget.rst',
]

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'  # Better theme for API docs

# Theme options
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'style_nav_header_background': '#2980B9',
}

# Add any paths that contain custom static files (such as style sheets) here.
# Only include when the directory exists to avoid warnings.
_HERE = os.path.dirname(__file__)
_STATIC_DIR = os.path.join(_HERE, '_static')
if os.path.isdir(_STATIC_DIR):
    html_static_path = ['_static']
else:
    html_static_path = []

# HTML output options
html_title = 'forge3d Documentation'
html_short_title = 'forge3d'
# Guard logo and favicon to avoid warnings when assets are missing.
_LOGO = os.path.join(_HERE, 'assets', 'logo-forge3d.png')
_FAVICON = os.path.join(_HERE, 'assets', 'favicon.ico')
html_logo = 'assets/logo-forge3d.png' if os.path.exists(_LOGO) else None
html_favicon = 'assets/favicon.ico' if os.path.exists(_FAVICON) else None

# HTML context
html_context = {
    'display_github': True,
    'github_user': 'forge3d',
    'github_repo': 'forge3d',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# LaTeX output options
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '',
    'printindex': '',
}

latex_documents = [
    ('index', 'forge3d.tex', 'forge3d Documentation',
     'forge3d contributors', 'manual'),
]

# Man page output
man_pages = [
    ('index', 'forge3d', 'forge3d Documentation',
     ['forge3d contributors'], 1)
]

# Text info output  
texinfo_documents = [
    ('index', 'forge3d', 'forge3d Documentation',
     'forge3d contributors', 'forge3d', 'Cross-platform GPU rendering library.',
     'Miscellaneous'),
]

# Epub output
epub_title = 'forge3d'
epub_author = 'forge3d contributors'
epub_publisher = 'forge3d contributors'
epub_copyright = '2025, forge3d contributors'

# Custom CSS
def setup(app):
    app.add_css_file('custom.css')

