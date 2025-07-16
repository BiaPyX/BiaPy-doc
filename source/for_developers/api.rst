BiaPy as a library
------------------

Library use
***********

BiaPy can be used not only via its command-line interface or GUI, but also directly from Python. This is especially useful when integrating BiaPy into other pipelines or using it in custom scripts.

Here is a minimal example of how to run BiaPy programmatically from Python:

.. code-block:: python

    from biapy import BiaPy

    # Set up your parameters
    config_path = "/path/to/config.yaml"            # Path to your YAML configuration file
    result_dir = "/path/to/results"                 # Directory to store the results
    job_name = "my_biapy_ob"                        # Name of the job
    run_id = 1                                      # Run ID for logging/versioning
    gpu = "0"                                       # GPU to use (as string, e.g., "0")

    # Create and run the BiaPy job
    biapy = BiaPy(config_path, result_dir=result_dir, name=job_name, run_id=run_id, gpu=gpu)
    biapy.run_job()

This will execute the workflow specified in the configuration file and store the output in the given result directory.

.. note::

   When using BiaPy programmatically, make sure that any custom code dependencies or paths are correctly configured in your environment.

----------------------------

API
***

.. toctree::                                                                    
   :maxdepth: 1
   :glob:

   ../API/config/config
   ../API/data/data
   ../API/engine/engine
   ../API/models/models
   ../API/utils/utils