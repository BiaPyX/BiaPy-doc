Using BiaPy from Python
***********************

BiaPy can be used not only via its command-line interface or GUI, but also directly from Python. This is especially useful when integrating BiaPy into other pipelines or using it in custom scripts.

This page is a task-oriented guide to the ``BiaPy`` class and the ``build_config`` helper. For the full, auto-generated listing of every module, class and function, see the `API Reference <api.html>`_.

.. note::

   When using BiaPy programmatically, make sure that any custom code dependencies or paths are correctly configured in your environment.


Minimal example
~~~~~~~~~~~~~~~

The shortest way to run a workflow: point BiaPy at a `YAML configuration file <configuration.html>`_ and run it.

.. code-block:: python

    from biapy import BiaPy

    config_path = "/path/to/config.yaml"   # Path to your YAML configuration file
    result_dir = "/path/to/results"        # Directory to store the results
    job_name = "my_biapy_job"              # Name of the job
    run_id = 1                             # Run ID for logging/versioning
    gpu = "0"                              # GPU to use (as string, e.g., "0")

    biapy = BiaPy(config_path, result_dir=result_dir, name=job_name, run_id=run_id, gpu=gpu)
    biapy.run_job()

This executes the workflow defined in the YAML file and stores the output in ``result_dir``. The rest of this guide covers the remaining options in detail.


Import
~~~~~~

The API exposes one class, ``BiaPy``, plus a helper, ``build_config``:

.. code-block:: python

    from biapy import BiaPy, build_config


Constructor options
~~~~~~~~~~~~~~~~~~~~~

Every way of creating a ``BiaPy`` object (Section 1) accepts the same options. These are referenced throughout the rest of the guide:

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Option
     - Default
     - Meaning
   * - ``result_dir``
     - —
     - Folder where all outputs go (results, checkpoints, logs, config backup). **Required** unless ``save_files=False``.
   * - ``name`` / ``run_id``
     - ``"unknown_job"`` / ``1``
     - Job identity. Outputs are written under ``result_dir/name/``, and log/config files are tagged with ``run_id``.
   * - ``gpu``
     - ``""``
     - GPU id(s), e.g. ``"0"`` or ``"0,1"``. ``""`` runs on CPU.
   * - ``verbose``
     - ``False``
     - If ``True``, mirror BiaPy's build-time messages to the console. Actual runs (``train``/``test``) always print; ``predict`` is quiet unless asked (Section 5).
   * - ``save_files``
     - ``True``
     - If ``True``, write the config backup and run log to ``result_dir``. If ``False``, nothing is written to disk (Section 9).


1. Instantiate a BiaPy object (three ways)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first argument is the configuration source; the rest are the constructor options above.

.. code-block:: python

    BiaPy("config.yaml", result_dir="/out", name="job", gpu="0")   # a) path to a YAML config
    BiaPy(cfg_dict_or_cfgnode, result_dir="/out", name="job")      # b) in-memory config (dict / CfgNode)
    BiaPy("model.pth", result_dir="/out", name="job")              # c) a .pth checkpoint

- **(a)** and **(b)** are equivalent ways to provide a full configuration — from a file, or built in memory (Section 2).
- **(c)** A BiaPy ``.pth`` embeds its configuration: it is recovered and switched to inference automatically, so the object is ready to test/predict.
- ``.safetensors`` checkpoints carry no configuration and are rejected — build one with ``build_config(...)`` and use form (b) instead.


2. Build a config without a YAML file — ``build_config``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``build_config`` turns high-level arguments into a config dict you pass to ``BiaPy`` (form (b)). The ``*_data`` and ``model`` dicts map their keys onto the corresponding config sections; ``extra_config`` is an escape hatch for any raw key.

.. code-block:: python

    cfg = build_config(
        workflow="INSTANCE_SEG", dims="2D", phase="both",   # phase: "train" | "test" | "both"
        patch_size=(256, 256, 1),
        model={"architecture": "unet"},                                  # -> MODEL.*
        train_data={"path": "...", "gt_path": "...", "in_memory": True}, # -> DATA.TRAIN.*
        val_data={"split_train": 0.1},                                   # -> DATA.VAL.*
        test_data={"path": "...", "load_gt": True},                      # -> DATA.TEST.*
        extra_config={"TRAIN": {"EPOCHS": 50}},                          # raw keys, deep-merged last
    )
    biapy = BiaPy(cfg, result_dir="/out", name="job", gpu="0")

``phase`` sets which phases are enabled (``TRAIN.ENABLE`` / ``TEST.ENABLE``); ``"both"`` enables training followed by testing.


3. Train and/or test
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    biapy.run_job()   # runs whatever the config enables: train, then test, then optional BMZ export
    biapy.train()     # training phase only  (requires the config to have TRAIN.ENABLE=True)
    biapy.test()      # test phase only      (requires the config to have TEST.ENABLE=True)

These phases write checkpoints, results and logs, so the object must have been created with a ``result_dir`` and ``save_files=True`` (the default) — see the constructor options above. ``run_job`` is the one-call equivalent of running the enabled phases in order; ``train``/``test`` let you run them individually.


4. Load a trained model to test/predict directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A shortcut that builds a **test-ready** BiaPy straight from a trained model, inferring the workflow for you:

.. code-block:: python

    biapy = BiaPy.load_workflow_from_model("model.pth", result_dir="/out", name="infer", gpu="0")
    biapy = BiaPy.load_workflow_from_model("affable-shark", ...)   # a BMZ id/nickname

- From a ``.pth``: the workflow is rebuilt from the embedded config (same as constructor form (c)).
- From a BMZ id/nickname: the workflow and dimensionality are inferred from the model's RDF.

Extra keyword arguments (``result_dir``, ``name``, ``gpu``, ``save_files``, ...) are forwarded to the constructor.


5. Predict on an in-memory array — ``predict``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run inference on a single NumPy image without going through files. The model is prepared automatically on the first call.

.. code-block:: python

    mask = biapy.predict(image)                       # returns an ndarray, writes NO files
    mask = biapy.predict(image, gt=gt)                # gt is only used if metrics are enabled
    biapy.predict(image, verbose=True)                # show the inference output on the console
    biapy.predict(image, return_prediction=False)     # write the normal on-disk outputs, return None

- ``return_prediction=True`` (default) captures the result in memory and skips all disk output.
- ``return_prediction=False`` produces the normal output files instead, so it needs a writable ``result_dir`` (Section 1). The same applies to synapse detection and ``TEST.BY_CHUNKS`` workflows, which must write intermediate files to complete.


6. Fine-tune a trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load a trained model, switch it to training, point it at your data, and continue from the loaded weights:

.. code-block:: python

    biapy = BiaPy.load_workflow_from_model("model.pth", result_dir="/out", name="finetune", gpu="0")
    biapy.update_config({
        "TRAIN.ENABLE": True, "TRAIN.EPOCHS": 50,
        "DATA.TRAIN.PATH": ".../x", "DATA.TRAIN.GT_PATH": ".../y",
    })
    biapy.train()

The loaded checkpoint is already set as the starting weights, so ``train()`` fine-tunes from them rather than from scratch. See Section 8 for ``update_config``.


7. Inspect the workflow
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    biapy                      # short summary (workflow, dims, patch size, model, device...)
    biapy.print_config()       # the full resolved configuration
    biapy.print_train_info()   # training overview: model, patch, epochs, LR, optimizer, augmentations, files
    biapy.print_test_info()    # test overview: data path, GT, overlap/padding, post-processing, files

The ``print_*_info`` methods also list the log and config file paths being written (or note that file writing is disabled).


8. Change the configuration — ``update_config``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit configuration values after construction. By default the workflow is rebuilt so its cached state stays consistent with the new values.

.. code-block:: python

    biapy.update_config({"DATA.PATCH_SIZE": (512, 512, 1)})       # structural change -> workflow rebuilt
    biapy.update_config({"TRAIN.EPOCHS": 100}, rebuild=False)     # value read at run time -> skip the rebuild
    biapy.update_config(TRAIN__EPOCHS=100)                        # kwargs form: "__" becomes "."
    biapy.update_config({"SYSTEM.NUM_GPUS": 2})                   # raises -> see below

- ``rebuild=True`` (default) reconstructs the workflow so construction-time state (dimensionality, model, post-processing, ...) reflects the change. Use ``rebuild=False`` only for values read fresh at run time (epochs, learning rate, data paths) to skip that cost.
- ``SYSTEM.*`` (GPUs, CPUs, seed) and the compute device are decided when the object is created and cannot be changed here — build a new ``BiaPy(...)`` for those.


9. Ephemeral mode (write nothing) — ``save_files=False``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For pure in-memory inference, disable all disk writes. ``result_dir`` then becomes optional.

.. code-block:: python

    biapy = BiaPy.load_workflow_from_model("model.pth", save_files=False, gpu="0")  # no result_dir needed
    mask = biapy.predict(image)   # nothing is written to disk

- With ``save_files=False`` there is no config backup, no log file, and no output folder is created.
- Operations that must write raise a clear error: ``train()``, ``test()``, ``predict(return_prediction=False)``, ``export_model_to_bmz()``, and synapse / ``TEST.BY_CHUNKS`` workflows.


Beyond workflows: using individual utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides running a full workflow, you can use many of the helper functions available in the `API Reference <api.html>`_. For example, this loads 3D raw and label images into memory:

.. code-block:: python

    from biapy.data.data_manipulation import load_data_from_dir

    raw_dir = '/content/data/train/raw'        # Directory containing raw images
    label_dir = '/content/data/train/label'    # Directory containing label images

    raw_images = load_data_from_dir(raw_dir, is_3d=True)
    label_images = load_data_from_dir(label_dir, is_3d=True)
