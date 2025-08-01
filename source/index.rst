BiaPy: Accessible deep learning on bioimages
============================================

.. image:: img/biapy_logo.svg
   :width: 50%
   :align: center 

`BiaPy <https://biapyx.github.io/>`__ is an open source library and application that streamlines the use of common **deep-learning workflows for a large variety of bioimage analysis tasks**, including 2D and 3D `semantic segmentation <workflows/semantic_segmentation.html>`__, `instance segmentation <workflows/instance_segmentation.html>`__, `object detection <workflows/detection.html>`__, `image denoising <workflows/denoising.html>`__, `single image super-resolution <workflows/super_resolution.html>`__, `self-supervised learning <workflows/self_supervision.html>`__ (for model pretraining), `image classification <workflows/classification.html>`__ and `image-to-image translation <workflows/image_to_image.html>`__.

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
   :caption: 🚀 Get started
   :glob:
   
   get_started/quick_start
   get_started/installation
   get_started/how_it_works
   get_started/select_workflow
   get_started/bmz
   get_started/faq

.. toctree::
   :maxdepth: 1
   :caption: ⚙️ Workflow configuration
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
   :caption: 📘 Tutorials
   :glob:

   tutorials/image-to-image/tutorials_i2i
   tutorials/instance_seg/tutorials_inst_seg
   tutorials/semantic_seg/tutorials_sem_seg
   tutorials/detection/tutorials_det

.. toctree::
   :maxdepth: 1
   :caption: 🛠️ For developers
   :glob:

   for_developers/configuration
   for_developers/library_examples
   for_developers/api
   for_developers/cpu_vs_gpu
   for_developers/contribute

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 📖 Bibliography

   bibliography
   

📖 Citation
===========
Please note that BiaPy is based on a publication. If you use it successfully for your research please be so kind to cite our work:

.. code-block:: text
    
   Franco-Barranco, D., Andrés-San Román, J.A., Hidalgo-Cenalmor, I.,
   Backová, L., González-Marfil, A., Caporal, C., Chessel, A., Gómez-Gálvez, P.,
   Escudero, L.M., Wei, D., Muñoz-Barrutia, A. and Arganda-Carreras, I.
   BiaPy: accessible deep learning on bioimages. Nat Methods 22, 1124-1126 (2025).
   https://doi.org/10.1038/s41592-025-02699-y
  
