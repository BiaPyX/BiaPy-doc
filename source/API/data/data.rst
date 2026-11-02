biapy.data
----------

.. automodule:: biapy.data
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
   * - :mod:`biapy.data.data_2D_manipulation`
     - Functions for manipulating 2D image data, including resizing, cropping, and augmentations.
   * - :mod:`biapy.data.data_3D_manipulation`
     - Functions for manipulating 3D image data, including resizing, cropping, and augmentations.
   * - :mod:`biapy.data.data_manipulation`
     - General data manipulation functions that work for both 2D and 3D images.
   * - :mod:`biapy.data.chunked_writer`
     - Zarr writer shared by the processes that create a large output file patch by patch.
   * - :mod:`biapy.data.dataset`
     - Dataset classes for loading and managing image data for training and inference.
   * - :mod:`biapy.data.membrane_channels`
     - Derived input channels for the membrane-repair problem.
   * - :mod:`biapy.data.norm`
     - Normalization functions for image data, including standardization and min-max scaling.
   * - :mod:`biapy.data.ome_zarr`
     - OME-Zarr (OME-NGFF) support.
   * - :mod:`biapy.data.omnipose_core`
     - Omnipose math: target generation and test-time mask reconstruction.
   * - :mod:`biapy.data.patch_sampling`
     - Foreground oversampling of the training patches.
   * - :mod:`biapy.data.pre_processing`
     - Pre-processing functions for preparing image data before feeding it into models.
   * - :mod:`biapy.data.roi_mask`
     - Region of interest (ROI) mask support for inference.
   * - :mod:`biapy.data.generators`
     - Data generators for creating batches of image data with optional augmentations.
   * - :mod:`biapy.data.post_processing`
     - Post-processing functions for processing model outputs, including thresholding, morphological operations, instance reconstruction and test-time augmentation.

.. toctree::
  :hidden:
  :maxdepth: 1

  chunked_writer
  data_2D_manipulation
  data_3D_manipulation
  data_manipulation
  dataset
  membrane_channels
  norm
  ome_zarr
  omnipose_core
  patch_sampling
  pre_processing
  roi_mask
  generators/generators
  post_processing/main_post_proc
