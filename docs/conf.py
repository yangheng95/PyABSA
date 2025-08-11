# -- Path setup --------------------------------------------------------------
import os
import sys
import re
from pathlib import Path

# Prevent heavy imports and side effects during docs build
os.environ["PYABSA_DOCS"] = "1"

# Resolve version without importing the package
try:
    from importlib.metadata import version as _pkg_version  # Python 3.8+
except Exception:  # pragma: no cover
    try:
        from importlib_metadata import version as _pkg_version  # backport
    except Exception:
        _pkg_version = None


def _get_pyabsa_version():
    # 1) Environment override
    env_v = os.environ.get("PYABSA_DOCS_VERSION")
    if env_v:
        return env_v
    # 2) Try installed distribution
    if _pkg_version is not None:
        try:
            return _pkg_version("pyabsa")
        except Exception:
            pass
    # 3) Parse from source file without importing
    init_path = Path(__file__).parent.parent / "pyabsa" / "__init__.py"
    try:
        text = init_path.read_text(encoding="utf-8")
        m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "0.0.0"


sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
project = "PyABSA"
author = "Yang, Heng"
copyright = "2022, Heng Yang"
release = _get_pyabsa_version()
master_doc = "index"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    # "autodoc2",
    "myst_parser",
    "piccolo_theme",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx_markdown_tables",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
]

autosummary_generate = True
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
nbsphinx_execute = "never"

# Avoid importing heavy optional dependencies during autodoc
autodoc_mock_imports = [
    "torch",
    "transformers",
    "metric_visualizer",
    "termcolor",
    "requests",
    "spacy",
    "thinc",
    "cupy",
    "sklearn",
    "numpy",
    "scipy",
    "pandas",
    "tqdm",
    "matplotlib",
]

# ===== AutoAPI configuration =====
autoapi_type = "python"
autoapi_dirs = ["../pyabsa"]
autoapi_ignore = [
    "**/__pycache__/**",
    "**/tests/**",
    "**/_Archive/**",
    "**/dev/**",
]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "special-members",
    "show-inheritance",
    "show-module-summary",
]

# ===== 传统autodoc配置 =====
# 启用autodoc扩展
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# 如果你的 docstring 里是传统的 :param: 风格，启用 fieldlist 兼容
myst_enable_extensions = ["fieldlist"]

# 仅文档化公开 API（遵循 __all__），可选；按需放开
autodoc2_module_all_regexes = [r"pyabsa\..*"]

# 如果有第三方类型注解老是 “reference target not found”，建议配 intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "torch": ("https://pytorch.org/docs/stable", {}),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", {}),
}

# -- HTML --------------------------------------------------------------------
language = "en"
html_theme = "piccolo_theme"
html_theme_options = {
    # Keep minimal options; theme provides clean defaults
    # You can set repository links, analytics, etc., here if needed
}

html_favicon = "favicon.png"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
add_module_names = True
autodoc_member_order = "groupwise"

# 复制按钮
copybutton_prompt_text = ">>> "
copybutton_line_continuation_character = "\\"
