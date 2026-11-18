# AGENTS.md

Sphinx documentation for BiaPy (`https://biapy.readthedocs.io/`), hosted on Read the Docs. Content is primarily reStructuredText (`.rst`); match the markup format of the file you edit.

## Layout
- `source/`: documentation pages, `conf.py`, `index.rst` (root `toctree`), and `requirements.txt`.
- `source/workflows/`, `source/get_started/`, `source/for_developers/`, `source/tutorials/`: hand-written guides.
- `source/API/`: hand-written `automodule` stubs, one per `biapy` module. Docstring content is not stored here: `conf.py` clones `BiaPyX/BiaPy` (to `../../../BiaPy` relative to `source/`) and autodoc reads it from there. Fix docstring problems upstream; edit `source/API/` only to add, remove or reorganize module pages.
- `source/img/`, `source/_static/`: compressed images and custom CSS/JS.
- `build/`: git-ignored build output. Never edit or commit files here.

## Environment & Build
Python/Sphinx only—there is no Node.js toolchain. Local Python must match `.readthedocs.yaml` (currently 3.11); update both together if upgrading.

1. **One-time setup** (skip if `conda env list` already shows `BiaPy_doc`):
   ```bash
   conda create -n BiaPy_doc python=3.11 -y
   conda run -n BiaPy_doc pip install -r source/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

2. **Build & validate.** Use `conda run`, not `conda activate`, which fails in non-interactive shells. The first build clones `BiaPyX/BiaPy` (needs network access) and is slow.
   ```bash
   conda run -n BiaPy_doc make html SPHINXOPTS="-W --keep-going"
   ```
   The build is currently warning-free, so a non-zero exit code means your change introduced a warning (broken ref, malformed directive, orphan page). Fix it before finishing. Output goes to `build/html/`.

## Conventions
- **Code blocks:** In `.rst`, use `.. code-block:: yaml` or `.. code-block:: bash` (never Markdown triple-backticks inside `.rst`). Keep YAML indentation valid.
- **Navigation:** Register every new page in a `toctree` (`index.rst` or its parent section index). Add `:orphan:` at the top of a page only if it must intentionally stay out of every toctree (e.g. `source/API/models/init.rst`).
- **Internal links:** Use `:ref:` / `:doc:` in `.rst`, and `{ref}` or relative links in `.md`.
- **Assets:** Place images in `source/img/` (compressed) and always include `:alt:` text.
- **Notebooks:** Tutorials link to Colab notebooks in `BiaPyX/BiaPy` or `BiaPyX/BiaPy-notebooks`. Verify URLs resolve; never commit `.ipynb` files or datasets to this repo.

## Guardrails
- **Preserve public URLs:** Do not rename or move existing pages unless asked. It changes the published URL and breaks external links (notebooks, papers, issues). If a move is required, update all in-repo references and tell the user a Read the Docs redirect is needed.
- **Read the Docs limits:** Check `.readthedocs.yaml` before modifying `source/requirements.txt` or `source/conf.py`; RTD enforces strict memory and build-time limits.
- **Mocked imports:** Do not remove entries from `autodoc_mock_imports` in `source/conf.py` (currently `resunet++`) unless you verify RTD can install the package.