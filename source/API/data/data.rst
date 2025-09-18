biapy.data
----------

.. automodule:: biapy.data
    :members:
    :undoc-members:
    :show-inheritance:

Submodules
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :mod:`biapy.data.data_2d_manipulation`
     - Functions for manipulating 2D image data, including resizing, cropping, and augmentations.
   * - :mod:`biapy.data.data_3d_manipulation`
     - Functions for manipulating 3D image data, including resizing, cropping, and augmentations.
   * - :mod:`biapy.data.data_manipulation`
     - General data manipulation functions that work for both 2D and 3D images.
   * - :mod:`biapy.data.dataset`
     - Dataset classes for loading and managing image data for training and inference.
   * - :mod:`biapy.data.norm`
     - Normalization functions for image data, including standardization and min-max scaling.
   * - :mod:`biapy.data.pre_processing`
     - Pre-processing functions for preparing image data before feeding it into models.
   * - :mod:`biapy.data.generators`
     - Data generators for creating batches of image data with optional augmentations.
   * - :mod:`biapy.data.post_processing`
     - Post-processing functions for processing model outputs, including thresholding and morphological operations.

.. toctree::
   :maxdepth: 1
   :hidden:

   data_2d_manipulation
   data_3d_manipulation
   data_manipulation
   dataset
   norm
   pre_processing
   generators/generators
   post_processing/post_processing
