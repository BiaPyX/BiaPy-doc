biapy.engine
------------

.. automodule:: biapy.engine
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
   * - :mod:`biapy.engine.base_workflow`
     - Base workflow class that provides the main structure and utility methods for building training and inference workflows in BiaPy.
   * - :mod:`biapy.engine.check_configuration`
     - Configuration checking utilities for BiaPy.
   * - :mod:`biapy.engine.classification`
     - Engine for image classification workflows.
   * - :mod:`biapy.engine.chunked_tiles`
     - Post-processing of the "by chunks" inference while the patches are being predicted.
   * - :mod:`biapy.engine.classification`
     - Engine for image classification workflows.
   * - :mod:`biapy.engine.denoising`
     - Engine for image denoising workflows.
   * - :mod:`biapy.engine.detection`
     - Engine for object detection workflows.
   * - :mod:`biapy.engine.image_to_image`
     - Engine for image-to-image translation workflows.
   * - :mod:`biapy.engine.instance_seg`
     - Engine for instance segmentation workflows.
   * - :mod:`biapy.engine.malis_loss`
     - MALIS-weighted affinity loss for the membrane-repair problem.
   * - :mod:`biapy.engine.membrane_repair`
     - Membrane-repair sub-problem of the image-to-image workflow.
   * - :mod:`biapy.engine.metrics`
     - Metrics and loss functions for model training and validation.
   * - :mod:`biapy.engine.schedulers`
     - Learning rate schedulers and related utilities.
   * - :mod:`biapy.engine.schedulers.warmup_cosine_decay`
     - Warmup cosine decay learning rate scheduler.
   * - :mod:`biapy.engine.schedulers.warmup_reduce_on_plateau`
     - Warmup and reduce-on-plateau learning rate scheduler, adapted from Cellpose.
   * - :mod:`biapy.engine.self_supervised`
     - Engines for self-supervised learning workflows (pre-training).
   * - :mod:`biapy.engine.semantic_seg`
     - Engine for semantic segmentation workflows.
   * - :mod:`biapy.engine.super_resolution`
     - Engine for single image super-resolution workflows.
   * - :mod:`biapy.engine.supervoxel_loss`
     - Supervoxel-based structure-aware affinity loss for the membrane-repair problem.
   * - :mod:`biapy.engine.swt_loss`
     - Stationary Wavelet Transform (SWT) loss, used to train Wavelettention.
   * - :mod:`biapy.engine.train_engine`
     - Training and evaluation engine for BiaPy.
   * - :mod:`biapy.engine.workflow_utils`
     - Reusable, workflow-specific helpers (e.g. Cellpose/Omnipose test-phase helpers).

.. toctree::
   :maxdepth: 1
   :hidden:

   base_workflow
   check_configuration
   chunked_tiles
   classification
   denoising
   detection
   image_to_image
   instance_seg
   malis_loss
   membrane_repair
   metrics
   self_supervised
   semantic_seg
   super_resolution
   supervoxel_loss
   swt_loss
   train_engine
   schedulers/schedulers
   schedulers/warmup_cosine_decay
   schedulers/warmup_reduce_on_plateau
   workflow_utils/workflow_utils
