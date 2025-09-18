.. _api-reference:

API Reference
=============

This section provides a comprehensive overview of the BiaPy library's public API, automatically generated from the source code's docstrings. The documentation is organized by the main modules of the project, covering everything from configuration and data handling to deep learning models and utility functions.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :mod:`biapy.config`
     - Main configuration module. Contains the core ``BiaPy`` class and logic for parsing 
       and managing configuration files. Entry point for defining and running workflows.
   * - :mod:`biapy.data`
     - Data handling tools. Provides functions and classes for loading, manipulating, 
       and preparing bioimages for training and inference, including augmentation, 
       normalization, and dataset generators.
   * - :mod:`biapy.engine`
     - Core execution engine. Defines and runs training and prediction workflows for tasks 
       such as semantic segmentation, instance segmentation, and denoising.
   * - :mod:`biapy.models`
     - Collection of deep learning models. Includes implementations of U-Net variants, 
       ResNet-based architectures, and attention-based networks used in bioimage analysis.
   * - :mod:`biapy.utils`
     - General-purpose utilities. Provides helper functions for file handling, 
       metric calculation, environment setup, callbacks, and more.

.. toctree::
   :maxdepth: 1
   :hidden:

   ../API/config/index
   ../API/data/data
   ../API/engine/engine
   ../API/models/models
   ../API/utils/index