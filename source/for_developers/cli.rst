.. _command-line-reference:

BiaPy command-line interface
=============================

After installing BiaPy, use the ``biapy`` command to run a workflow or perform
standalone checks. This page documents the utility subcommands. For the
Python API, see :doc:`python_api`; for the command-line entry point in the API
reference, see :doc:`api`.

You can ask a subcommand for its argument help with ``biapy <command> --help``.

All the subcommands accept ``--json``, which prints the result as JSON, for use
by scripts and agents. It always has ``schema_version`` (the version of this
format), ``ok`` (whether the command succeeded), ``errors`` and ``warnings``
(lists of messages), and ``data``, with the data of each command described in
its section. For example:

.. code-block:: json

   {
     "schema_version": "1.0",
     "ok": true,
     "errors": [],
     "warnings": [],
     "data": {"workflows": ["SEMANTIC_SEG", "INSTANCE_SEG", "..."]}
   }

Progress messages go to the standard error, so the standard output only holds
the result. The exit code is ``0`` on success, ``1`` if the command fails, and
``2`` for invalid command-line arguments.

Validate a configuration
------------------------

Check a YAML configuration without running a workflow, loading data, building
a model or writing output.

.. code-block:: console

   biapy validate-config <config.yaml>
   biapy validate-config <config.yaml> --no-data-check
   biapy validate-config <config.yaml> --json

Arguments and options:

* ``config.yaml``: path to the YAML configuration file.
* ``--no-data-check``: validate the configuration without checking whether
  the data paths exist. Use this when the data is not available yet.
* ``--json``: print the result as JSON, with the workflow (``workflow``) and
  dimensionality (``ndim``) of the configuration.

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
* ``--json``: print the result as JSON, with what was found in each folder
  (``folders``) and, in super-resolution, the upscaling of the targets
  (``upscaling``).

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

The ``--json`` option prints the result as JSON, with the workflows in
``workflows``. Without it, each workflow is printed on its own line.

Inspect an image
----------------

Inspect an image file to see how BiaPy reads its dimensions, axes and data
type, and which problem dimensionality (2D or 3D) it suggests:

.. code-block:: console

   biapy inspect-image <path_to_image>
   biapy inspect-image <path_to_image> --json

Arguments and options:

* ``path_to_image``: path to the image to inspect.
* ``--json``: print the result as JSON: the image as stored in the file
  (``stored``), how BiaPy reads it in 2D and 3D (``2D`` and ``3D``), whether
  the stored axes are known (``axes_known``) and the suggested dimensionality
  (``suggested_ndim``).

The text report shows the stored shape, axes and data type, possible 2D and
3D interpretations, and a suggested ``PROBLEM.NDIM`` when one can be
determined. The command exits with status ``0`` on success and ``1`` if the
image cannot be inspected.

Get a configuration template
----------------------------

Print or save the basic configuration template of a workflow, the one in
``biapy/config/templates``. Its data paths are placeholders (``/path/to/data``)
to replace before running it:

.. code-block:: console

   biapy get-config-template --workflow WORKFLOW --ndim 2D_OR_3D [--store <file.yaml>] [--json]

Arguments and options:

* ``--workflow``: workflow type (the ``PROBLEM.TYPE`` value), such as
  ``SEMANTIC_SEG``.
* ``--ndim``: ``2D`` or ``3D``.
* ``--store``: save the template to this YAML file, as it is (with its
  comments).
* ``--json``: print the result as JSON, with the template in ``config`` and the
  file it was saved to, if any, in ``stored_in``.

Without ``--store`` nor ``--json``, the template is printed as YAML. The command
exits with status ``0`` on success, ``1`` if the workflow or dimensionality is
not valid, and ``2`` for invalid command-line arguments.

Create a configuration from the data
------------------------------------

Create a configuration adapted to the data, as the wizard does, to train a
model from scratch. The data is checked first, as ``check-data`` does; if it
can be used, the configuration is planned from it (channels, classes, patch
size, network, batch size for the available GPU, training length...):

.. code-block:: console

   biapy create-config --workflow WORKFLOW --ndim 2D_OR_3D --train-raw <folder> \
       [--train-gt <folder>] [--test-raw <folder>] [--test-gt <folder>] \
       [--quick] [--store <file.yaml>] [--json]

Arguments and options:

* ``--workflow``: workflow type (the ``PROBLEM.TYPE`` value), such as
  ``SEMANTIC_SEG``.
* ``--ndim``: ``2D`` or ``3D``.
* ``--train-raw``: folder containing the training raw images.
* ``--train-gt``: folder containing the training target/ground-truth data,
  when the workflow uses it.
* ``--test-raw``, ``--test-gt``: folders with the test raw images and their
  targets, to also test the trained model.
* ``--quick``: a short training, to check that everything works before a full
  one.
* ``--store``: save the configuration to this YAML file, with a summary of what
  was decided on top.
* ``--json``: print the result as JSON, with what was found in each folder
  (``folders``), the configuration (``config``) and the summary of what was
  decided (``summary``).

Semantic segmentation, instance segmentation and detection are planned from the
data; for the other workflows the configuration is valid but generic, and a
warning says so. The command exits with status ``0`` when the configuration is
created, ``1`` if the data cannot be used, and ``2`` for invalid command-line
arguments.

Search pretrained models in the BioImage Model Zoo
--------------------------------------------------

Search the `BioImage Model Zoo <https://bioimage.io>`__ for pretrained models
that BiaPy can use. It needs internet and may take a minute:

.. code-block:: console

   biapy search-bmz-models [--workflow WORKFLOW] [--ndim 2D_OR_3D] \
       [--nchannels N] [--nclasses N] [--query WORDS] [--json]

Arguments and options (all optional; by default, all the models):

* ``--workflow``: workflow type (the ``PROBLEM.TYPE`` value), such as
  ``INSTANCE_SEG``.
* ``--ndim``: ``2D`` or ``3D``.
* ``--nchannels``: number of channels of the images (e.g. ``1`` for grayscale).
* ``--nclasses``: number of classes of the data.
* ``--query``: words that must appear in the model's name, description or tags,
  such as ``nuclei``.
* ``--json``: print the result as JSON, with the models in ``models``.

Each model is listed with its identifier (the value for
``MODEL.BMZ.SOURCE_MODEL_ID``), workflow, dimensionality, channels, name and
link to the BioImage Model Zoo. The command exits with status ``0`` when the
search is done (even if no model is found) and ``1`` if it fails.

