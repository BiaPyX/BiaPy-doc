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

Inspect the results of a job
----------------------------

Describe what a job left in its folder: whether it finished, how the training
went, the test metrics, the checkpoints and the outputs. The job's folder is
``<result_dir>/<name>``, the ``--result_dir`` and ``--name`` it was run with:

.. code-block:: console

   biapy inspect-results <job_folder> [--run-id N] [--json]

Arguments and options:

* ``job_folder``: folder of the job, ``<result_dir>/<name>``.
* ``--run-id``: run to describe, the ``--run_id`` it was run with. By default,
  the one run last.
* ``--json``: print the result as JSON, with:

  * ``job`` and ``run_id``: the run described.
  * ``finished``: whether it finished.
  * ``log`` and ``config``: the log and the configuration of its last execution,
    with its workflow and dimensionality.
  * ``training``: the epochs reached of the planned ones, whether it stopped by
    early stopping, and the values of the last epoch and of the best one (the
    one with the lowest validation loss, saved in the best checkpoint). As in
    BiaPy's training log, ``test_*`` values are the validation ones and epochs
    count from 0.
  * ``test``: the test metrics BiaPy printed at the end of the test of its last
    execution, named as it prints them (e.g. ``PSNR (merge patches)``), and the
    CSV file with the metrics of each image. The matching metrics of instance
    segmentation (precision, recall, F1... by IoU threshold) are only in the
    CSV file.
  * ``checkpoints``: the best one and all of them.
  * ``predictions``: the output folders and their number of files (in
    classification, the ``predictions.csv`` file).
  * ``charts``: the plots of the training.

A run that is still running, or that stopped with an error, is not finished;
its error is in the output of the command that ran it. Jobs run with the Python
API (``train()``, ``test()``) are never finished: only ``biapy --config ...``
marks the end of the job in its log. Parts the job didn't do, such as the
training of a job that only tested, are ``null``.

The command exits with status ``0`` when the job is described (even if it
didn't finish), ``1`` if the folder doesn't exist or holds no run, and ``2``
for invalid command-line arguments.

Plan a job
----------

Describe what running a configuration would do, without running it: nothing
is trained, no model is built and nothing is written. Use it to check a job
before launching it, which can take hours of GPU or write over an existing
job:

.. code-block:: console

   biapy plan <config.yaml> [--result_dir DIR [--name NAME] [--run_id N]] \
       [--percentage PERCENT] [--json]

Arguments and options:

* ``config.yaml``: path to the YAML configuration file.
* ``--result_dir``, ``--name`` and ``--run_id``: optional, the ones the job
  would be run with (``biapy --config ...``), to also check where it would
  write and whether that run already exists. Give them before launching a job.
  ``--name`` defaults to ``unknown_job`` and ``--run_id`` to ``1``.
* ``--percentage``: percentage of each Zarr/H5 image to check; defaults to
  ``100``.
* ``--json``: print the result as JSON, with:

  * ``workflow``, ``ndim`` and ``phases``: whether the job would train and
    test.
  * ``job``: the job's folder and run, whether that run already ``exists``
    (running it would write over its outputs) and the folders it would write;
    ``null`` without ``--result_dir``.
  * ``data``: for each data folder, what ``check-data`` finds (number of files,
    channels, classes) and the smallest and largest shape of its images, when
    known.
  * ``model``: the architecture, its source and the checkpoint it would load.
  * ``training``: epochs, batch size, optimizer, learning rate, patience and
    patch size.
  * ``hardware``: the NVIDIA GPUs and the RAM of this computer.
  * ``estimates``: the RAM to train and to test and the disk space the job
    would need, as ``create-config`` estimates them. They are estimates, not
    guarantees, and ``null`` when they can't be made. The GPU memory is not
    estimated.

The data is checked as ``check-data`` does, and its problems are the plan's
errors. That the run already exists is a warning. The command exits with
status ``0`` if the job can be run, ``1`` if the configuration is not valid or
its data can't be used, and ``2`` for invalid command-line arguments.

Run a job in the background
---------------------------

Run a job and return at once, while it keeps running in the background. Follow
it with ``job-status`` and stop it with ``stop-job``, from this or any other
terminal. It is the job ``biapy --config ...`` would run, but it is first
checked as ``plan`` does:

.. code-block:: console

   biapy run <config.yaml> --result_dir DIR [--name NAME] [--run_id N] \
       [--gpu GPU] [--overwrite] [--json]

Arguments and options:

* ``config.yaml``: path to the YAML configuration file.
* ``--result_dir``, ``--name`` and ``--run_id``: where the job writes, as in
  ``biapy --config ...``. ``--name`` defaults to ``unknown_job`` and
  ``--run_id`` to ``1``.
* ``--gpu``: GPU to use, such as ``0``, or ``mps`` on Apple Silicon. Several
  GPUs, such as ``0,1``, run one process per GPU with
  ``torch.distributed.run``, which BiaPy supports on Linux. By default, the job
  runs on the CPU.
* ``--overwrite``: run it even if the run already exists, writing over its
  outputs.
* ``--json``: print the result as JSON, with:

  * ``job_dir``, ``job`` and ``run_id``: the job's folder,
    ``<result_dir>/<name>``, and its run.
  * ``pid``: the process of the job.
  * ``command``: the command it was run with.
  * ``output``: the file with everything the job prints, errors included.

The job is not run if its configuration is not valid or its data can't be used
(the errors of ``plan``), if that run is already running, or if it already
exists and ``--overwrite`` is not given. Closing the terminal doesn't stop it.

The job's state and output are saved in the ``jobs`` folder of the job's
folder: ``jobs/<name>_<run_id>.json`` and ``jobs/<name>_<run_id>.out``.

The command exits with status ``0`` when the job is started, ``1`` when it is
not, and ``2`` for invalid command-line arguments.

Follow a job
------------

Say whether a job is running, where it is (the training epoch, the test images
processed) and, if it failed, why:

.. code-block:: console

   biapy job-status <job_folder> [--run-id N] [--json]

Arguments and options:

* ``job_folder``: folder of the job, ``<result_dir>/<name>``.
* ``--run-id``: run to describe. By default, the one started last with
  ``biapy run``, or else the one run last.
* ``--json``: print the result as JSON, with:

  * ``job`` and ``run_id``: the run described.
  * ``state``: one of:

    * ``starting``: it is loading the data or building the model;
    * ``training`` or ``testing``: it is training or testing;
    * ``completed``: it finished;
    * ``stopped``: it was stopped with ``stop-job``;
    * ``failed``: it ended with an error;
    * ``unknown``: it was not started with ``biapy run`` and didn't finish, so
      it may be running or have stopped.

  * ``pid``: its process, or ``null`` if it is not running.
  * ``progress``: the ``phase`` (``train`` or ``test``), the ``epoch`` it is in
    of the ``epochs`` planned (counted from 1), the training time ``elapsed``
    and the ``estimated_total``, as BiaPy prints them after each epoch, the
    ``test_images`` processed (not known in classification), and the values of
    the ``last`` epoch, as in ``inspect-results``.
  * ``resources``: the CPU, RAM and GPU memory it uses now, or ``null`` if it
    is not running. The GPU memory is only known for NVIDIA GPUs.
  * ``error``: if it failed, the end of its output, with the error.
  * ``output``: the file with everything it printed or, if it was not started
    with ``biapy run``, its log.

For jobs started otherwise, such as with ``biapy --config ...`` or BiaPy-GUI,
only their log is known: whether they finished and where they are.

The command exits with status ``0`` when the state is given (even if the job
failed), ``1`` if the folder doesn't exist or holds no run, and ``2`` for
invalid command-line arguments.

Stop a job
----------

Stop a job started with ``biapy run``, and the processes it started:

.. code-block:: console

   biapy stop-job <job_folder> [--run-id N] [--timeout SECONDS] [--json]

Arguments and options:

* ``job_folder``: folder of the job, ``<result_dir>/<name>``.
* ``--run-id``: run to stop. By default, the one started last.
* ``--timeout``: seconds to wait for it to end before killing it; defaults to
  ``30``.
* ``--json``: print the result as JSON, with:

  * ``job`` and ``run_id``: the run stopped.
  * ``was_running``: whether it was running. If not, nothing is done, with a
    warning.
  * ``killed``: whether it had to be killed because it didn't end in time.

BiaPy saves nothing when it is stopped: the checkpoints are the ones it saved
before, and one being written at that moment may be left incomplete. After it,
``job-status`` says ``stopped``.

The command exits with status ``0`` when the job is stopped or was not
running, ``1`` if no job of that folder was started with ``biapy run``, and
``2`` for invalid command-line arguments.