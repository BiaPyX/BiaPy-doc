# AGENTS.md

Sphinx documentation for BiaPy, published on Read the Docs. Content is mostly `.rst`; match the format of the file you edit.

## Layout
- `source/`: all docs, plus `conf.py`, `index.rst` (root toctree) and `requirements.txt`.
- `source/workflows/`, `source/get_started/`, `source/for_developers/`, `source/tutorials/`: hand-written pages.
- `source/API/`: API reference pages for the BiaPy package.
- `source/img/`, `source/_static/`: images and custom CSS.
- `build/` is git-ignored output. Never edit or commit it.

## Build & validate
Python/Sphinx only; there is no Node toolchain.
```bash
conda create -n BiaPy_doc python=3.11 -y && conda activate BiaPy_doc
pip install -r source/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
make html   # output in build/html/
```
- Run `make html` after every change and fix any new Sphinx warnings (broken refs, bad directives, orphan pages).
- The local Python version must match `.readthedocs.yaml` (currently 3.11); update both together.

## Conventions
- Register every new page in a `toctree` (`index.rst` or the parent index), or it will warn as orphaned.
- Link internally with `:ref:`/`:doc:` (`.rst`) or `{ref}`/relative links (`.md`).
- Give code blocks a language (`yaml`, `bash`) and keep YAML indentation valid.
- Images go in `source/img/` (compressed, with descriptive `alt` text).
- Tutorials link to Colab notebooks hosted in `BiaPyX/BiaPy` or `BiaPyX/BiaPy-notebooks`; check links still resolve. Do not commit notebooks or datasets here.

## Guardrails
- Check `.readthedocs.yaml` before changing `source/requirements.txt` or `source/conf.py`; RTD has build time and memory limits.
- Do not remove entries from `autodoc_mock_imports` in `conf.py` unless RTD can install the package.
