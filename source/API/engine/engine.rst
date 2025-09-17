biapy.engine
------------

.. automodule:: biapy.engine
   :members:
   :undoc-members:
   :show-inheritance:

Overview
~~~~~~~~

The ``biapy.engine`` package provides the core training and inference
workflows, as well as utilities such as metrics, optimizers, and
schedulers.

The following table summarizes the available submodules:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Description
   * - :mod:`biapy.engine`
     - Core engine package containing workflows and utilities for different tasks.
   * - :mod:`biapy.engine.base_workflow`
     - Base workflow class that provides the main structure and utility methods for building training and inference workflows in BiaPy.
   * - :mod:`biapy.engine.check_configuration`
     - Configuration checking utilities for BiaPy.
   * - :mod:`biapy.engine.classification`
     - Engine for image classification workflows.
   * - :mod:`biapy.engine.denoising`
     - Engine for image denoising workflows.
   * - :mod:`biapy.engine.detection`
     - Engine for object detection workflows.
   * - :mod:`biapy.engine.instance_seg`
     - Engine for instance segmentation workflows.
   * - :mod:`biapy.engine.metrics`
     - Metrics and evaluation utilities for model training and validation.
   * - :mod:`biapy.engine.schedulers`
     - Learning rate schedulers and related utilities.
   * -     :mod:`biapy.engine.schedulers.warmup_cosine_decay`
     - Warmup cosine decay learning rate scheduler.
   * - :mod:`biapy.engine.self_supervised`
     - Engines for self-supervised learning workflows (pre-training).
   * - :mod:`biapy.engine.semantic_seg`
     - Engine for semantic segmentation workflows.
   * - :mod:`biapy.engine.super_resolution`
     - Engine for single image super-resolution workflows.
   * - :mod:`biapy.engine.train_engine`
     - Training and evaluation engine for BiaPy.

.. toctree::
   :maxdepth: 1
   :hidden:

   base_workflow
   classification
   denoising
   detection
   instance_seg
   metrics
   self_supervised
   semantic_seg
   super_resolution
   train_engine
   schedulers/schedulers