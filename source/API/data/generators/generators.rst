biapy.data.generators
---------------------

.. automodule:: biapy.data.generators
    :members:
    :undoc-members:
    :show-inheritance:

Submodules
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :mod:`biapy.data.generators.augmentors`
     - Data augmentation utilities for generators.
   * - :mod:`biapy.data.generators.chunked_test_pair_data_generator`
     - Test data generator for paired data that is read chunk by chunk.
   * - :mod:`biapy.data.generators.chunked_workflow_process_generator`
     - Data generator for processing large data chunk by chunk, without augmentation.
   * - :mod:`biapy.data.generators.membrane_augmentors`
     - Corruption augmentors for the membrane-repair problem.
   * - :mod:`biapy.data.generators.membrane_repair_data_generator`
     - Dedicated data generator for the membrane-repair problem.
   * - :mod:`biapy.data.generators.pair_base_data_generator`
     - Base class for generators that handle paired data (2D and 3D).
   * - :mod:`biapy.data.generators.pair_data_2D_generator`
     - Data generator for paired 2D images (e.g., input–target pairs).
   * - :mod:`biapy.data.generators.pair_data_3D_generator`
     - Data generator for paired 3D volumes (e.g., input–target pairs).
   * - :mod:`biapy.data.generators.single_base_data_generator`
     - Base class for generators that handle single-input data (2D and 3D).
   * - :mod:`biapy.data.generators.single_data_2D_generator`
     - Data generator for single-input 2D images.
   * - :mod:`biapy.data.generators.single_data_3D_generator`
     - Data generator for single-input 3D volumes.
   * - :mod:`biapy.data.generators.test_pair_data_generators`
     - Test data generator for paired images and masks, without augmentation.
   * - :mod:`biapy.data.generators.test_single_data_generator`
     - Test data generator for single images, without augmentation.

.. toctree::
   :maxdepth: 1
   :hidden:

   augmentors
   chunked_test_pair_data_generator
   chunked_workflow_process_generator
   membrane_augmentors
   membrane_repair_data_generator
   pair_base_data_generator
   pair_data_2D_generator
   pair_data_3D_generator
   single_base_data_generator
   single_data_2D_generator
   single_data_3D_generator
   test_pair_data_generators
   test_single_data_generator
