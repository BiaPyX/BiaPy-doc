.. _ovarian_reserve_tutorial:

Ovarian Reserve: 3D Instance Segmentation of Oocytes
-----------------------------------------------------

About this tutorial
~~~~~~~~~~~~~~~~~~~

This tutorial explains how to use **BiaPy** for **3D instance segmentation of oocytes** in whole-mount mouse ovaries, based on :cite:`ovarianreserve2025`.

The goal is to make this workflow accessible to all BiaPy users (GUI, notebook, Galaxy, Docker, CLI, or API), even if this is your first time working with 3D instance segmentation.

.. list-table::
  :align: center

  * - .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/oocyte_train_sample.gif
         :align: center
         :scale: 100%

         Left: raw DDX4 image stack. Right: corresponding instance-label stack.


.. note::

   The pretrained model checkpoint will be published as soon as possible (BioImage Model Zoo and direct PyTorch file). In the meantime, you can train your own model using the provided training dataset and YAML file (see below).

Paper overview
~~~~~~~~~~~~~~

Our publication :cite:`ovarianreserve2025` presents a pipeline to map the entire ovarian reserve in 3D by imaging intact mouse ovaries with **light-sheet fluorescence microscopy (SPIM)** and segmenting every individual oocyte with a deep learning model. The key steps of the pipeline are:

#. **Whole-ovary SPIM imaging**: intact ovaries are cleared and imaged at single-cell resolution across the full organ, yielding large 3D fluorescence volumes (DDX4 channel marking oocyte cytoplasm).
#. **3D instance segmentation with BiaPy**: a 3D residual U-Net (*ResU-Net*) is trained on manually curated oocyte labels using the **BCD** channel representation (Binary mask + Contour + Distance), followed by marker-controlled watershed to recover individual instances.
#. **Age-resolved quantification**: segmented oocyte counts and spatial distributions are compared across seven age groups (5–60 weeks), revealing the dynamics of ovarian reserve decline.

.. list-table::
  :align: center
  :widths: 50 50

  * - .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/F1.large.jpg
         :align: center
         :figwidth: 300px

         Paper Figure 1 from :cite:`ovarianreserve2025`: SPIM whole-ovary imaging and model-based oocyte segmentation workflow.

    - .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/F2.large.jpg
         :align: center
         :figwidth: 300px

         Paper Figure 2 from :cite:`ovarianreserve2025`: age-resolved ovarian reserve quantification enabled by 3D oocyte segmentation.


Data preparation
~~~~~~~~~~~~~~~~

This tutorial uses two datasets. Note that they represent a **small but representative subset** of the full data described in :cite:`ovarianreserve2025` — the complete study involved many more labeled slices and full-ovary volumes. These subsets have been selected so that both training and inference (applying the model to new images to produce segmentations) can be completed in a **reasonable amount of time** on a standard GPU workstation.

The datasets are presented in the natural order of the workflow: first the training data (used to learn the model), then the test data (used to evaluate it).

* **Training dataset (sample)**: ``oocyte_training.zip`` (240.9 MB) containing a curated set of paired 2D raw and label slices extracted from 3D oocyte image stacks, sufficient to train or fine-tune the segmentation model: `Google Drive link <https://drive.google.com/file/d/1xA2b9nY1KuIGC-ZjYg--MXQ8r8GSyOwP/view?usp=sharing>`__.

  Once unzipped, you should find the following directory tree:

  .. code-block::

    oocyte_training/
    ├── raw/
    │   ├── 10W_100330_frame70.tif
    │   ├── 10W_105114_1.tif
    │   ├── ...
    │   └── 5W_150806_frame54.tif
    └── label/
        ├── 10W_100330_frame70.tif
        ├── 10W_105114_1.tif
        ├── ...
        └── 5W_150806_frame54.tif

* **Test dataset (Zenodo)**: ``raw_ovary.rar`` (13.0 GB) containing seven full 3D ovary volumes covering a range of ages (5, 10, 22, 31, 40, 50, and 60 weeks) in TIFF format. These are a subset of the ovaries analyzed in the paper and serve as the held-out test set: `Zenodo link <https://zenodo.org/records/19085211>`__.

  Once unrared, you should find the following directory tree:

  .. code-block::

    raw_ovary/
    ├── w5_134934.tif
    ├── w10_112648.tif
    ├── w22_090202.tif
    ├── w31_084030.tif
    ├── w40_094116.tif
    ├── w50_142422.tif
    └── w60_155112.tif

Quick start: run predictions on your own data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If this is your first time here and you simply want to apply the pretrained model to the provided data (a process called **inference**) with minimal setup, follow these steps:

#. **Install BiaPy** if you have not done so yet. Follow the :ref:`installation guide <installation>` to set up your preferred interface (GUI, Docker, CLI, etc.). First-time users will find the **GUI** the easiest option. The pipeline described in :cite:`ovarianreserve2025` was developed using **BiaPy 3.6.3**; the steps in this tutorial should work with any version from 3.6.3 onwards.
#. **Prepare your data** — use your own 3D TIFF ovary images, or, if you do not have your own data yet, download the provided test dataset: ``raw_ovary.rar`` (13.0 GB) from `Zenodo <https://zenodo.org/records/19085211>`__ and unrar it to obtain the ``raw_ovary/`` folder (see `Data preparation`_ above for the expected directory layout).
#. **Download the prediction configuration file**: :download:`ovarian-reserve-inference.yaml <ovarian-reserve-inference.yaml>`.
#. **Edit two paths** in the YAML: set ``DATA.TEST.PATH`` to your ``raw_ovary/`` folder and ``PATHS.CHECKPOINT_FILE`` to the pretrained model checkpoint (see the `Inference configuration`_ section for details).
#. **Run BiaPy** using the interface of your choice — step-by-step instructions for every option (GUI, Colab, Galaxy, Docker, Command line, API) are in the `Run predictions on new images`_ section below.

Image and data requirements
***************************

* Input images must be **3D TIFF** volumes.
* Typical axis order is ``ZYX`` (single channel).
* Expected physical resolution is approximately ``(5.0, 0.867, 0.867)`` µm in ``(Z, Y, X)``.
* If your data use a different resolution, resampling to this scale is recommended for best results.


Inference configuration
~~~~~~~~~~~~~~~~~~~~~~~

You can download the ready-to-edit YAML file here:

* :download:`ovarian-reserve-inference.yaml <ovarian-reserve-inference.yaml>`

.. collapse:: Expand to preview ovarian-reserve-inference.yaml

   .. literalinclude:: ovarian-reserve-inference.yaml
      :language: yaml

The two mandatory edits are:

* ``DATA.TEST.PATH`` → folder containing your TIFF test volumes.
* ``PATHS.CHECKPOINT_FILE`` → path to the pretrained model checkpoint.


Run predictions on new images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again, **BiaPy** offers different options to run this prediction workflow (also called *inference*) depending on your level of computer expertise. Select the one that is most appropriate for you:

.. tabs::

   .. tab:: GUI

     First, download the prediction configuration file :download:`ovarian-reserve-inference.yaml <ovarian-reserve-inference.yaml>` and prepare the pretrained model checkpoint (or your own trained checkpoint).

     Next, in BiaPy's GUI, follow the following instructions:

     .. carousel::
      :show_controls:
      :show_captions_below:
      :data-bs-interval: false
      :show_indicators:
      :show_dark:

      .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/GUI-load-and-modify-workflow.png

         Step 1: Click on "Load and modify workflow" and select the ``ovarian-reserve-inference.yaml`` file you just downloaded.

      .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/GUI-load-information.png

         Step 2: Click on "OK".

      .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/GUI-modify-configuration-file-inference.png

         Step 3: Introduce the corresponding paths to your test data (raw images), a pretrained ``.pth`` model file, and a name for your modified configuration file (red boxes indicate missing information).

      .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/GUI-modify-configuration-file-inference-filled.png

         Step 4: Once that information is correctly introduced, click on "Save File".

      .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/GUI-modify-configuration-file-inference-success.png

         Step 5: A success window should appear. Click on "OK".

      .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/GUI-results-folder-and-job-name-inference.png

         Step 6: Input the folder you wish to use to store the results of the workflow by clicking on the "Browse" button of "Output folder to save the results".

      .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/GUI-results-folder-and-job-name-inference-filled.png

         Step 7: Once filled, click on "Check File".

      .. figure:: ../../img/tutorials/instance-segmentation/ovarian-reserve/GUI-results-folder-and-job-name-inference-success.png

         Step 8: A success window should appear. Click on "OK" and then on "Run Workflow".


     \

     .. note:: BiaPy's GUI requires that all data and configuration files reside on the same machine where the GUI is being executed.


     .. tip:: If you need additional help with the parameters of the GUI, watch BiaPy's `GUI walkthrough video <https://www.youtube.com/embed/vY7aBh5FUNk?si=yvVolBnu5APNeHwB>`__.

   .. tab:: Google Colab

     Open the BiaPy inference notebook `here <https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/BiaPy_Inference.ipynb>`__ and follow its instructions to run the prediction workflow.

     .. tip:: If you need additional help, watch BiaPy's `Notebook walkthrough video <https://youtu.be/KEqfio-EnYw>`__.

   .. tab:: Galaxy

     Open BiaPy in Galaxy using this `launch link <https://imaging.usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Fbiapy%2Fbiapy%2F3.6.5%2Bgalaxy0&version=latest>`__.

     Then upload your TIFF images, the YAML configuration file, and the model checkpoint. Select ``Yes, I already have one and I want to run BiaPy directly``, choose your configuration and checkpoint files, and run the job to download the predictions.

   .. tab:: Docker

     First, download the prediction configuration file :download:`ovarian-reserve-inference.yaml <ovarian-reserve-inference.yaml>` and prepare the pretrained model checkpoint (or your own trained checkpoint).

     Next edit the configuration file to set the correct path to your test data folder (``DATA.TEST.PATH``) and the pretrained model (``PATHS.CHECKPOINT_FILE``).

     Then, open a terminal as described in :ref:`installation` and execute the following commands:

      .. code-block:: bash

         job_cfg_file=/home/user/ovarian-reserve-inference.yaml
         data_dir=/home/user/raw_ovary
         result_dir=/home/user/exp_results
         job_name=my_ovarian_reserve_test
         job_counter=1
         gpu_number=0

         docker run --rm \
            --gpus "device=$gpu_number" \
            --mount type=bind,source=$job_cfg_file,target=$job_cfg_file \
            --mount type=bind,source=$result_dir,target=$result_dir \
            --mount type=bind,source=$data_dir,target=$data_dir \
            biapyx/biapy:latest-11.8 \
               biapy \
               --config $job_cfg_file \
               --result_dir $result_dir \
               --name $job_name \
               --run_id $job_counter \
               --gpu "$gpu_number"

      .. note::
          Note that ``data_dir`` must contain the folder pointed to by ``DATA.TEST.PATH`` so the container can find it. For instance, in this example ``DATA.TEST.PATH`` could be ``/home/user/raw_ovary``.

   .. tab:: Command line

      First, download the prediction configuration file :download:`ovarian-reserve-inference.yaml <ovarian-reserve-inference.yaml>` and prepare the pretrained model checkpoint (or your own trained checkpoint).

      Next edit the configuration file to set the correct path to your test data folder (``DATA.TEST.PATH``) and the pretrained model (``PATHS.CHECKPOINT_FILE``).

      Next, run the following commands from a terminal:

      .. code-block:: bash

         job_cfg_file=/home/user/ovarian-reserve-inference.yaml
         result_dir=/home/user/exp_results
         job_name=my_ovarian_reserve_test
         job_counter=1
         gpu_number=0

         conda activate BiaPy_env

         biapy \
            --config $job_cfg_file \
            --result_dir $result_dir \
            --name $job_name \
            --run_id $job_counter \
            --gpu "$gpu_number"

      Before running the command, make sure to update the following parameters:

        * ``job_cfg_file``: full path to the ovarian reserve prediction configuration file.
        * ``result_dir``: full path to the folder where results will be stored. A new subfolder will be created within this folder for each run.
        * ``job_name``: a name for your experiment. Tip: avoid using hyphens (``-``) or spaces in the name.
        * ``job_counter``: a number to identify each execution of your experiment. Start with ``1`` and increase it if you run the experiment multiple times.

   .. tab:: API

      If you prefer to integrate the workflow into your own Python code, you can run the same prediction setup through the **BiaPy** API once ``DATA.TEST.PATH`` and ``PATHS.CHECKPOINT_FILE`` are correctly defined in ``ovarian-reserve-inference.yaml``.

      .. code-block:: python

         from biapy import BiaPy

         config_path = "/home/user/ovarian-reserve-inference.yaml"
         result_dir = "/home/user/exp_results"
         job_name = "my_ovarian_reserve_test"

         biapy = BiaPy(config_path, result_dir=result_dir, name=job_name, run_id=1, gpu="0")
         biapy.run_job()


Training or fine-tuning from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can train a model using ``oocyte_training.zip`` and test it on the Zenodo 3D volumes.

Download the training YAML file here:

* :download:`ovarian-reserve-training.yaml <ovarian-reserve-training.yaml>`

.. collapse:: Expand to preview ovarian-reserve-training.yaml

   .. literalinclude:: ovarian-reserve-training.yaml
      :language: yaml

Before running training, update at least these paths:

* ``DATA.TRAIN.PATH`` → ``.../oocyte_training/raw/``
* ``DATA.TRAIN.GT_PATH`` → ``.../oocyte_training/label/``
* ``DATA.TEST.PATH`` → ``.../raw_ovary/`` (Zenodo test set)

.. note::

   The test data are TIFF volumes (not Zarr input files). The provided YAML is already configured for TIFF input axis order.

Run training (CLI):

.. code-block:: bash

   job_cfg_file=/home/user/ovarian-reserve-training.yaml
   result_dir=/home/user/exp_results
   job_name=my_ovarian_reserve_training
   job_counter=1
   gpu_number=0

   conda activate BiaPy_env

   biapy \
      --config $job_cfg_file \
      --result_dir $result_dir \
      --name $job_name \
      --run_id $job_counter \
      --gpu "$gpu_number"


Post-analysis scripts
~~~~~~~~~~~~~~~~~~~~~

After segmentation, you can run the analysis scripts from the `Boke-Lab ovarian_reserve repository <https://github.com/Boke-Lab/ovarian_reserve>`__:

* **oocyte density**: quantifies oocytes per volume.
* **radial quantification**: measures the radial spatial distribution of oocytes.

These scripts reproduce the quantitative analyses described in :cite:`ovarianreserve2025`.
