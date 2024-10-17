Select workflow
---------------

In bioimage analysis, the **input** and **output** data vary depending on the specific workflow being used. The following are the workflows implemented in BiaPy and the corresponding input and output data they require. Once you've identified the one you wish to use, follow the running instructions found on each workflow's page (under "How to run").

* `Semantic segmentation <../workflows/semantic_segmentation.html>`_, the input is an image of the area or object of interest, while the output is another image of the same shape as the input, with a **semantic label** (a numerical value defining its category) **assigned to each pixel**. During the training phase, the expected label image of the input (i.e. the ground truth) needs to be also provided for the model to learn:

  .. image:: ../img/semantic_seg/workflow-scheme.svg
   :width: 70%
   :align: center 

  Below is a list of publicly available datasets that are ready to be used in BiaPy for this workflow:

  .. list-table::
    :widths: auto
    :header-rows: 1
    :align: center

    * - Example dataset
      - Image dimensions
      - Link to data
    * - `Electron Microscopy Dataset (EPFL - CVLAB) <https://www.epfl.ch/labs/cvlab/data/data-em/>`__
      - 2D
      - `fibsem_epfl.zip <https://drive.google.com/file/d/1DfUoVHf__xk-s4BWSKbkfKYMnES-9RJt/view?usp=drive_link>`__
    * - `Electron Microscopy Dataset (EPFL - CVLAB) <https://www.epfl.ch/labs/cvlab/data/data-em/>`__
      - 3D
      - `lucchi3D.zip <https://drive.google.com/file/d/10Cf11PtERq4pDHCJroekxu_hf10EZzwG/view?usp=sharing>`__

  
\

* `Instance segmentation <../workflows/instance_segmentation.html>`_, the input and output are similar to semantic segmentation, but the output also includes **a unique identifier for each individual object of interest**. During the training phase, the expected instance label image of the input (i.e. the ground truth) needs to be also provided for the model to learn:

  .. image:: ../img/instance-seg/workflow-scheme.svg
   :width: 70%
   :align: center 

  Below is a list of publicly available datasets that are ready to be used in BiaPy for this workflow:

  .. list-table::
    :widths: auto
    :header-rows: 1
    :align: center

    * - Example dataset
      - Image dimensions
      - Link to data
    * - `Stardist V2 <https://www.nature.com/articles/s41597-022-01721-8>`__
      - 2D
      - `Stardist_v2.zip <https://drive.google.com/file/d/1b7_WDDGEEaEoIpO_1EefVr0w0VQaetmg/view?usp=drive_link>`__
    * - `3D demo (from StarDist 0.3.0 release) <https://github.com/mpicbg-csbd/stardist/releases/download/0.3.0/demo3D.zip>`__
      - 3D
      - `demo3D.zip <https://drive.google.com/file/d/1pypWJ4Z9sRLPlVHbG6zpwmS6COkm3wUg/view?usp=drive_link>`__

\


* `Object detection <../workflows/detection.html>`_, the goal is to recognize objects in images without needing a pixel-level accuracy output. The input is an image, while the output is a CSV file containing the **coordinates of the center point of each object**. During the training phase, the list of coordinates from the input objects (i.e. the ground truth) needs to be also provided for the model to learn:

  .. image:: ../img/detection/workflow-scheme.svg
   :width: 70%
   :align: center 

  \
  Additionally, Biapy may output an image with the probability map of each object's center.

  Below is a list of publicly available datasets that are ready to be used in BiaPy for this workflow:

  .. list-table::
    :widths: auto
    :header-rows: 1
    :align: center

    * - Example dataset
      - Image dimensions
      - Link to data
    * - `Stardist V2 (detection) <https://zenodo.org/record/3715492#.Y4m7FjPMJH6>`__
      - 2D
      - `Stardist_v2_detection.zip <https://drive.google.com/file/d/1pWqQhcWY15b5fVLZDkPS-vnE-RU6NlYf/view?usp=drive_link>`__
    * - `NucMM-Z <https://arxiv.org/abs/2107.05840>`__
      - 3D
      - `NucMM-Z_training.zip <https://drive.google.com/file/d/19P4AcvBPJXeW7QRj92Jh1keunGa5fi8d/view?usp=drive_link>`__



* `Image denoising <../workflows/denoising.html>`_, the goal is to remove noise from a given input image. The input is a noisy image, and the **output is the denoised image**. No ground truth is required as the model uses an unsupervised learning technique to remove noise (`Noise2Void <https://arxiv.org/abs/1811.10980>`__).

  .. image:: ../img/denoising/workflow-scheme.svg
   :width: 70%
   :align: center 

  Below is a list of publicly available datasets that are ready to be used in BiaPy for this workflow:

  .. list-table::
    :widths: auto
    :header-rows: 1
    :align: center

    * - Example dataset
      - Image dimensions
      - Link to data
    * - `Noise2void Convallaria 2D (by B. Schroth-Diez) <https://github.com/juglab/n2v>`__
      - 2D
      - `convallaria2D.zip <https://drive.google.com/file/d/1TFvOySOiIgVIv9p4pbHdEbai-d2YGDvV/view?usp=drive_link>`__
    * - `Noise2void Flywing 3D (by R. Piscitello) <https://github.com/juglab/n2v>`__
      - 3D
      - `flywing3D.zip <https://drive.google.com/file/d/1OIjnUoJKdnbClBlpzk7V5R8wtoLont-r/view?usp=drive_link>`__


\  

* `Single image super-resolution <../workflows/super_resolution.html>`_, the goal is to reconstruct high-resolution (HR) images from low-resolution (LR) ones. The input is a LR image, and the **output is a HR** (usually ``×2`` or ``×4`` larger) **version of the same image**. During the training phase, the expected HR image of the input LR image (i.e. the ground truth) needs to be also provided for the model to learn:

  .. image:: ../img/super-resolution/workflow-scheme.svg
   :width: 70%
   :align: center 


  Below is a list of publicly available datasets that are ready to be used in BiaPy for this workflow:

  .. list-table::
    :widths: auto
    :header-rows: 1
    :align: center

    * - Example dataset
      - Image dimensions
      - Link to data
    * - `F-actin dataset (ZeroCostDL4Mic) <https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki>`__
      - 2D
      - `f_actin_sr_2d.zip <https://drive.google.com/file/d/1rtrR_jt8hcBEqvwx_amFBNR7CMP5NXLo/view?usp=drive_link>`__
    * - `Confocal 2 STED - Nuclear Pore complex <https://zenodo.org/records/4624364#.YF3jsa9Kibg>`__
      - 3D
      - `Nuclear_Pore_complez_3D.zip <https://drive.google.com/file/d/1TfQVK7arJiRAVmKHRebsfi8NEas8ni4s/view?usp=drive_link>`__


\

* `Self-supervised pre-training <../workflows/self_supervision.html>`_, the model is trained without the use of labeled data. Instead, the model is presented with a so-called pretext task, such as predicting the rotation of an image, which allows it to learn useful features from the data. Once this initial training is complete, the model can be fine-tuned using labeled data for a specific task, such as image classification. The input in this workflow is simply a set of images, and the **output is the pre-trained model**.

  .. image:: ../img/self-supervised/workflow-scheme.svg
   :width: 70%
   :align: center 

  Below is a list of publicly available datasets that are ready to be used in BiaPy for this workflow:

  .. list-table::
    :widths: auto
    :header-rows: 1
    :align: center

    * - Example dataset
      - Image dimensions
      - Link to data
    * - `Electron Microscopy Dataset (EPFL - CVLAB) <https://www.epfl.ch/labs/cvlab/data/data-em/>`__
      - 2D
      - `fibsem_epfl.zip <https://drive.google.com/file/d/1DfUoVHf__xk-s4BWSKbkfKYMnES-9RJt/view?usp=drive_link>`__
    * - `Electron Microscopy Dataset (EPFL - CVLAB) <https://www.epfl.ch/labs/cvlab/data/data-em/>`__
      - 3D
      - `lucchi3D.zip <https://drive.google.com/file/d/10Cf11PtERq4pDHCJroekxu_hf10EZzwG/view?usp=sharing>`__


\

* `Image classification <../workflows/classification.html>`_, the goal is to match a given input image to its corresponding class. The **input is a set of images, and the output is a file (usually CSV) containing the predicted class of each input image**.

  .. image:: ../img/classification/workflow-scheme.svg
   :width: 70%
   :align: center 

  Below is a list of publicly available datasets that are ready to be used in BiaPy for this workflow:

  .. list-table::
    :widths: auto
    :header-rows: 1
    :align: center

    * - Example dataset
      - Image dimensions
      - Link to data
    * - `DermaMNIST <https://www.nature.com/articles/s41597-022-01721-8>`__
      - 2D
      - `DermaMNIST.zip <https://drive.google.com/file/d/15_pnH4_tJcwhOhNqFsm26NQuJbNbFSIN/view?usp=drive_link>`__
    * - `OrganMNIST3D <https://medmnist.com/>`__
      - 3D
      - `organMNIST3D.zip <https://drive.google.com/file/d/1pypWJ4Z9sRLPlVHbG6zpwmS6COkm3wUg/view?usp=drive_link>`__
    * - `Butterfly Image Classification <https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification>`__
      - 2D
      - `butterfly_data.zip <https://drive.google.com/file/d/1m4_3UAgUsZ8FDjB4HyfA50Sht7_XkfdB/view?usp=drive_link>`__


\

* `Image-to-image translation <../workflows/image_to_image.html>`_, the purpose of this workflow is to **translate or map input images to corresponding target images**. Often referred to as "image-to-image," this process is versatile and can be applied to various goals, including **image inpainting, colorization, and even super-resolution** (with a scale factor of ``x1``). During the training phase, the expected "translated" image of the input image (i.e. the ground truth) needs to be also provided for the model to learn:

  .. image:: ../img/i2i/workflow-scheme.svg
   :width: 70%
   :align: center 

  Below is a list of publicly available datasets that are ready to be used in BiaPy for this workflow:

  .. list-table::
    :widths: auto
    :header-rows: 1
    :align: center

    * - Example dataset
      - Image dimensions
      - Link to data
    * - `lifeact-RFP and sir-DNA dataset <https://zenodo.org/records/3941889#.XxrkzWMzaV4>`__
      - 2D
      - `Dapi_dataset.zip <https://drive.google.com/file/d/1L8AXNjh0_updVI3-v1duf6CbcZb8uZK7/view?usp=drive_link>`__
    * - `Nucleoli Dataset (Allen Institute) <https://downloads.allencell.org/publication-data/label-free-prediction/index.html>`__
      - 3D
      - `label-free-allen-nucleoli-3D.zip <https://drive.google.com/file/d/18vD7vDAx_lQfSD6uCMEwHIPn1BBhUGQq/view?usp=drive_link>`__


