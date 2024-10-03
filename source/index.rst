BiaPy: Bioimage analysis pipelines in Python
============================================

.. image:: img/biapy_logo.svg
   :width: 50%
   :align: center 

`BiaPy <https://biapyx.github.io/>`__ is an open source ready-to-use all-in-one library that provides **deep-learning workflows for a large variety of bioimage analysis tasks**, including 2D and 3D `semantic segmentation <workflows/semantic_segmentation.html>`__, `instance segmentation <workflows/instance_segmentation.html>`__, `object detection <workflows/detection.html>`__, `image denoising <workflows/denoising.html>`__, `single image super-resolution <workflows/super_resolution.html>`__, `self-supervised learning <workflows/self_supervision.html>`__ (for model pretraining), `image classification <workflows/classification.html>`__ and `image-to-image translation <workflows/image_to_image.html>`__.

BiaPy is a versatile platform designed to accommodate both proficient computer scientists and users less experienced in programming. It offers diverse and user-friendly access points to our workflows.
                                                                         
Find a comprehensive overview of BiaPy and its functionality in the following videos:

.. list-table:: 

  * - .. image:: https://raw.githubusercontent.com/BiaPyX/BiaPy/master/img/BiaPy_presentation_and_demo_at_RTmfm.jpg
          :alt: BiaPy history and GUI demo
          :target: https://www.youtube.com/watch?v=Gnm-VsZQ6Cc

    - .. image:: https://raw.githubusercontent.com/BiaPyX/BiaPy/master/img/BiaPy-Euro-BioImaging-youtube.png
          :alt: BiaPy presentation
          :target: https://www.youtube.com/watch?v=6eYtX-ySpc0

.. toctree::
   :maxdepth: 1
   :caption: Get started
   :glob:
   
   get_started/quick_start
   get_started/installation
   get_started/how_it_works
   get_started/configuration
   get_started/select_workflow
   get_started/bmz
   get_started/faq
   get_started/contribute
   get_started/cpu_vs_gpu

.. toctree::
   :maxdepth: 1
   :caption: Workflow configuration
   :glob:

   workflows/classification
   workflows/denoising
   workflows/image_to_image
   workflows/instance_segmentation
   workflows/detection
   workflows/self_supervision
   workflows/semantic_segmentation
   workflows/super_resolution
   
.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :glob:

   
   tutorials/image-to-image/tutorials_i2i
   tutorials/instance_seg/tutorials_inst_seg
   tutorials/semantic_seg/tutorials_sem_seg
   tutorials/detection/tutorials_det

.. toctree::                                                                    
   :maxdepth: 1
   :caption: API
   :glob:

   API/config/config
   API/data/data
   API/engine/engine
   API/models/models
   API/utils/utils


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Bibliography

   bibliography
   

Citation
========

.. code-block:: text
    
   Franco-Barranco, Daniel, et al. "BiaPy: a ready-to-use library for Bioimage Analysis 
   Pipelines." 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI). 
   IEEE, 2023.
  
