.. _i2i_lightmycell:

(Paper) LightMyCells challenge: self-supervised Vision Transformers for image-to-image labeling
-----------------------------------------------------------------------------------------------

This tutorial aims to reproduce the results reported in the following paper:

.. code-block:: text

  Franco-Barranco, Daniel, et al. "Self-supervised Vision Transformers for image-to-image 
  labeling: a BiaPy solution to the LightMyCells Challenge." 2024 IEEE 21th International 
  Symposium on Biomedical Imaging (ISBI). IEEE, 2024.

In this work, we address the Cell Painting problem within the `LightMyCells challenge at the International Symposium on Biomedical Imaging (ISBI) 2024 <https://lightmycells.grand-challenge.org/>`__, aiming to predict optimally focused fluorescence images from label-free transmitted light inputs. We used the image to image workflow to solve this problem, where the goal is to learn a mapping between an input image and an output image. 


.. figure:: ../../img/i2i/lightmycells_fig1.png
    :align: center

    Proposed approach.

.. _lightmycells_data_prep:

Data preparation
~~~~~~~~~~~~~~~~

In our proposed approach we implemented a custom data loader to handle more than one out-of-focus image. To ensure the proper operation of the library the data directory tree should be something like this (here actin training data as example): 

.. collapse:: Expand directory tree 

    .. code-block:: bash
  
      lightmycells_dataset/
      ├── train
      │   ├── x
      │   │   ├── Study_3_BF_image_53_Actin.ome.tiff
      |   │   |   ├── Study_3_BF_image_53_BF_z0.ome.tiff   
      |   │   |   ├── Study_3_BF_image_53_BF_z1.ome.tiff
      |   │   |   ├── . . .  
      |   │   |   ├── Study_3_BF_image_53_BF_z19.ome.tiff       
      │   │   ├── Study_3_BF_image_54_Actin.ome.tiff
      |   │   |   ├── Study_3_BF_image_54_BF_z0.ome.tiff    
      │   │   ├── . . .
      │   │   ├── Study_6_PC_image_111_Actin.ome.tiff/
      |   │   |   ├── Study_6_PC_image_111_PC_z0.ome.tiff   
      |   │   |   ├── Study_6_PC_image_111_PC_z1.ome.tiff
      |   │   |   ├── . . .  
      |   │   |   ├── Study_6_PC_image_111_PC_z10.ome.tiff 
      │   └── y
      │       ├── Study_3_BF_image_53_Actin.ome.tiff
      |       |   └── Study_3_BF_image_53_Actin.ome.tiff          
      │       ├── Study_3_BF_image_54_Actin.ome.tiff
      |       |   └── Study_3_BF_image_54_Actin.ome.tiff   
      │       ├── . . .
      │       ├── Study_6_PC_image_111_Actin.ome.tiff
      |       |   └── Study_6_PC_image_111_Actin.ome.tiff  
      └── val
          ├── . . .

\

For the new images you want to predict (test data), you can follow the same directory structure (so you can use the validation folder if you created it) or just put all the images in a directory. 

Configuration                                                                                                                 
~~~~~~~~~~~~~

Templates will be ready soon (after the challenge). 

.. _lightmycells_run:

Run
~~~

For that you need to download the templates:


.. tabs::

   .. tab:: Reuse our model

        Soon.

   .. tab:: Train by your own

        Soon.

.. _lightmycells_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. figure:: ../../img/i2i/lightmycells_fig2.png
   :align: center                  

   Results on the LightMyCells challenge of our approach. 

