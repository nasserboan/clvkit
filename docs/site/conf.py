"""Sphinx configuration for the clvkit documentation site."""

import pathlib
import shutil

# -- Project ----------------------------------------------------------------
project = "clvkit"
author = "Nasser Boan"
copyright = "2026, Nasser Boan"

# -- Extensions -------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# The notebooks carry committed outputs and the datasets are gitignored, so the
# build renders them as-is rather than executing.
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- HTML / theme -----------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "clvkit"
html_favicon = "_static/clvkit-favicon.svg"

html_theme_options = {
    "logo": {
        "image_light": "_static/clvkit-light.svg",
        "image_dark": "_static/clvkit-dark.svg",
        "alt_text": "clvkit",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nasserboan/clvkit",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/clvkit/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_align": "left",
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_end": [],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}

html_context = {"default_mode": "light"}

# -- Analytics --------------------------------------------------------------
# GitHub Pages records nothing by default, so referrers and utm_* keys land
# nowhere. This Counter.dev beacon captures both, which is what makes launch-day
# channels distinguishable. The data-id is a public client-side identifier.
# Only the clvkit repo deploys the site (see .github/workflows/pages.yml), so
# this ships from there. See docs/launch-measurement.md for the keyed links.
html_js_files = [
    (
        "https://cdn.counter.dev/script.js",
        {"data-id": "9618de6c-12c8-41d5-9269-675c3c9fb7b0", "data-utcoffset": "-3"},
    )
]


# -- Notebook sourcing ------------------------------------------------------
def _copy_notebooks(app):
    """Bring the repo's example notebooks into the Sphinx source tree.

    They live at ``examples/`` in the repo root; nbsphinx only renders notebooks
    under the source dir. Copying at build time keeps one source of truth and
    works the same locally and in CI. The copies are gitignored.
    """
    src = pathlib.Path(app.srcdir).parent.parent / "examples"
    dst = pathlib.Path(app.srcdir) / "examples"
    dst.mkdir(exist_ok=True)
    for stem in ("start_here", "cdnow_clv", "online_retail_ii_cohort"):
        shutil.copy(src / f"{stem}.ipynb", dst / f"{stem}.ipynb")


def setup(app):
    app.connect("builder-inited", _copy_notebooks)
