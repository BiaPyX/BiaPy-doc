Library use
***********

BiaPy can be used not only via its command-line interface or GUI, but also directly from Python. This is especially useful when integrating BiaPy into other pipelines or using it in custom scripts.

Minimal example
~~~~~~~~~~~~~~~

Here is a minimal example of **how to run BiaPy programmatically** from Python:

.. code-block:: python

    from biapy import BiaPy

    # Set up your parameters
    config_path = "/path/to/config.yaml"            # Path to your YAML configuration file
    result_dir = "/path/to/results"                 # Directory to store the results
    job_name = "my_biapy_job"                       # Name of the job
    run_id = 1                                      # Run ID for logging/versioning
    gpu = "0"                                       # GPU to use (as string, e.g., "0")

    # Create and run the BiaPy job
    biapy = BiaPy(config_path, result_dir=result_dir, name=job_name, run_id=run_id, gpu=gpu)
    biapy.run_job()

This will execute the workflow specified in the `YAML configuration file <configuration.html>`_ (defined by ``config_path``) and store the output in the given result directory (defined by ``result_dir``).

.. note::

   When using BiaPy programmatically, make sure that any custom code dependencies or paths are correctly configured in your environment.


Data loading example
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Besides running a full workflow, you can use many useful methods available in `BiaPy's API Overview <api.html>`. For example, here is a short Python script that loads 3D raw and label images into memory:

.. code-block:: python

    from biapy.data.data_manipulation import load_data_from_dir

    # Set the paths to the image directories
    raw_dir = '/content/data/train/raw'        # Directory containing raw images
    label_dir = '/content/data/train/label'    # Directory containing label images

    # Load 3D images into memory
    raw_images = load_data_from_dir(raw_dir, is_3d=True)
    label_images = load_data_from_dir(label_dir, is_3d=True)
