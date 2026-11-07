.. _command-line-reference:

BiaPy command-line interface
=============================

After installing BiaPy, use the ``biapy`` command to run a workflow or perform
standalone checks. This page documents the four utility subcommands. For the
Python API, see :doc:`python_api`; for the command-line entry point in the API
reference, see :doc:`api`.

You can ask a subcommand for its argument help with ``biapy <command> --help``.

Validate a configuration
------------------------

Check a YAML configuration without running a workflow, loading data, building
a model or writing output.

.. code-block:: console

   biapy validate-config <config.yaml>
   biapy validate-config <config.yaml> --no-data-check

Arguments and options:

* ``config.yaml``: path to the YAML configuration file.
* ``--no-data-check``: validate the configuration without checking whether
  the data paths exist. Use this when the data is not available yet.

The command prints ``OK`` when the configuration is valid, or an error naming
the first problem. It exits with status ``0`` on success, ``1`` if validation
fails, and ``2`` for invalid command-line arguments.

Check data
----------

Check whether data can be used by BiaPy. Provide either a configuration file,
or the workflow and raw-data folder directly:

.. code-block:: console

   biapy check-data <config.yaml> [--percentage PERCENT] [--json]
   biapy check-data --workflow WORKFLOW --ndim 2D_OR_3D \
       --raw <folder> [--gt <folder>] [--percentage PERCENT] [--json]

Arguments and options:

* ``config.yaml``: optional path to a YAML configuration. The data folders
  configured in the file are checked against the configuration.
* ``--workflow``: workflow type (the ``PROBLEM.TYPE`` value), such as
  ``SEMANTIC_SEG``. Required when checking folders without a configuration.
* ``--ndim``: ``2D`` or ``3D``. Required when checking folders without a
  configuration.
* ``--raw``: folder containing raw images. Required when checking folders
  without a configuration.
* ``--gt``: optional folder containing target/ground-truth images. Use when
  the workflow has target data.
* ``--percentage``: percentage of each Zarr/H5 image to check; defaults to
  ``100``.
* ``--json``: print the full report as JSON, for use by scripts and agents.

The configuration-file form and the direct-folder form are alternatives; do
not combine them. The command exits with status ``0`` if the data is usable,
``1`` if the data or configuration is invalid, and ``2`` for invalid
command-line arguments.

List workflows
--------------

List the workflow types available in BiaPy (the values accepted for
``PROBLEM.TYPE``):

.. code-block:: console

   biapy list-workflows
   biapy list-workflows --json

The ``--json`` option prints the list as JSON. Without it, each workflow is
printed on its own line.

Inspect an image
----------------

Inspect an image file to see how BiaPy reads its dimensions, axes and data
type, and which problem dimensionality (2D or 3D) it suggests:

.. code-block:: console

   biapy inspect-image <path_to_image>
   biapy inspect-image <path_to_image> --json

Arguments and options:

* ``path_to_image``: path to the image to inspect.
* ``--json``: print the report as JSON.

The text report shows the stored shape, axes and data type, possible 2D and
3D interpretations, and a suggested ``PROBLEM.NDIM`` when one can be
determined. The command exits with status ``0`` on success and ``1`` if the
image cannot be inspected.
