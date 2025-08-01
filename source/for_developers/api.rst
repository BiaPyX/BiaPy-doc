.. _api-reference:

API Reference
=============

This section provides a comprehensive overview of the BiaPy library's public API, automatically generated from the source code's docstrings. The documentation is organized by the main modules of the project, covering everything from configuration and data handling to deep learning models and utility functions.

Configuration and Core
----------------------
This module contains the main `BiaPy` class and the configuration parsing logic. It is the entry point for defining and running entire analysis workflows.

.. toctree::
   :maxdepth: 1

   ../API/config/config

Data Handling and Preprocessing
-------------------------------
Explore the functions and classes for loading, manipulating, and preparing bioimages for training and inference. This includes data augmentation, normalization, and different data generators.

.. toctree::
   :maxdepth: 1

   ../API/data/data

Training and Workflow Engines
-----------------------------
The core of BiaPy's functionality for defining and running training and prediction workflows. This section details the engine classes for various tasks like semantic segmentation, instance segmentation, and denoising.

.. toctree::
   :maxdepth: 1

   ../API/engine/engine

Deep Learning Models
--------------------
A collection of custom and pre-built deep learning models available in BiaPy, including implementations of U-Net, ResNet, and various attention-based architectures.

.. toctree::
   :maxdepth: 1

   ../API/models/models

Utility Functions and Helpers
-----------------------------
Miscellaneous functions that assist in various tasks across the library, such as file handling, metric calculation, and environment setup.

.. toctree::
   :maxdepth: 1

   ../API/utils/utils