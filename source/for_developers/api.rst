.. _api-reference:

API Reference
=============

This section provides a comprehensive overview of the BiaPy library's public API, automatically generated from the source code's docstrings. The documentation is organized by the main modules of the project, covering everything from configuration and data handling to deep learning models and utility functions.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :doc:`biapy <../API/biapy/biapy>`
     - Top-level package. Exposes the ``BiaPy`` class, ``build_config`` and ``validate_config`` 
       (entry points to define, validate and run workflows from Python) and the ``biapy`` command-line ``main``.
   * - :doc:`BiaPy command-line interface <cli>`
     - Reference for the installed ``biapy`` command and its ``validate-config``, ``check-data``,
       ``list-workflows`` and ``inspect-image`` subcommands.
   * - :doc:`biapy.config <../API/config/index>`
     - Main configuration module. Contains the logic for parsing and managing 
       configuration files and the default values of every workflow.
   * - :doc:`biapy.data <../API/data/data>`
     - Data handling tools. Provides functions and classes for loading, manipulating, 
       and preparing bioimages for training and inference, including augmentation, 
       normalization, dataset generators, OME-Zarr support and post-processing.
   * - :doc:`biapy.engine <../API/engine/engine>`
     - Core execution engine. Defines and runs training and prediction workflows for tasks 
       such as semantic segmentation, instance segmentation, detection, denoising, 
       super-resolution, self-supervised learning, classification and image-to-image translation.
   * - :doc:`biapy.models <../API/models/models>`
     - Collection of deep learning models. Includes implementations of U-Net variants, 
       ResNet-based architectures, transformer-based networks (ViT, UNETR, DPT...), 
       super-resolution and image-to-image models, and BioImage Model Zoo utilities.
   * - :doc:`biapy.utils <../API/utils/utils>`
     - General-purpose utilities. Provides helper functions for file handling, 
       metric calculation, object matching, environment setup, callbacks, and more.
   * - :doc:`biapy.citations <../API/citations/citations>`
     - Works to cite when using BiaPy with a given configuration, exported as 
       BibTeX or Markdown.
   * - :doc:`biapy.wizard <../API/wizard/wizard>`
     - Logic behind the configuration wizards of BiaPy's front-ends (BiaPy-GUI, Fiji plugin...): 
       data checks, dataset fingerprints, configuration planning and resource estimation.


**Links to package modules:**

.. toctree::
   :maxdepth: 1
   :titlesonly:

   ../API/biapy/biapy
   ../API/config/index
   ../API/data/data
   ../API/engine/engine
   ../API/models/models
   ../API/utils/utils
   ../API/citations/citations
   ../API/wizard/wizard