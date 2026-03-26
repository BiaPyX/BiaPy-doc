.. _ovarian_reserve_tutorial:

Ovarian Reserve: 3D Instance Segmentation of Oocytes
-----------------------------------------------------

This tutorial explains how to use **BiaPy** for **3D instance segmentation of oocytes** in whole-mount mouse ovary images, as described in :cite:`ovarianreserve2025`. The workflow automatically detects and delineates individual oocytes in 3D fluorescence volumes stained for **DDX4** (a germ-cell-specific marker) and acquired by light-sheet (SPIM) microscopy.

.. role:: raw-html(raw)
    :format: html

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

.. note::

   The pretrained model will be made available in the `BioImage Model Zoo <https://bioimage.io>`__ and as a **PyTorch** checkpoint file as soon as possible. Stay tuned for updates.


Dataset
~~~~~~~

Two datasets are associated with this tutorial:

**Inference dataset (Zenodo)**
   A representative subset of the full dataset is publicly available on `Zenodo <https://zenodo.org/records/19085211>`__ :cite:`ovarianreserve2025`. It contains **seven whole-mount ovarian samples** from mice of different ages (**5, 10, 22, 30, 40, 50,** and **60 weeks**) imaged by SPIM microscopy and stained for **DDX4**. Download and extract the archive; after extraction the folder will look like this:

   .. collapse:: Expand to see the directory structure

      .. code-block::

         raw_ovary/
         ├── w5_134934.tif
         ├── w10_112648.tif
         ├── w22_090202.tif
         ├── w31_084030.tif
         ├── w40_094116.tif
         ├── w50_142422.tif
         └── w60_155112.tif

**Training dataset (Google Drive)**
   A curated set of annotated 2D image slices extracted from the 3D volumes, suitable for training or fine-tuning the model, is available for download `here <https://drive.google.com/file/d/1xA2b9nY1KuIGC-ZjYg--MXQ8r8GSyOwP/view?usp=sharing>`__ (``oocyte_training.zip``). After extracting it you will find two folders, ``raw`` (fluorescence images) and ``label`` (corresponding instance masks):

   .. collapse:: Expand to see the directory structure

      .. code-block::

         oocyte_training/
         ├── raw/
         │   ├── 5W_130858_frame206.tif
         │   ├── 5W_130858_frame277.tif
         │   ├── 10W_100330_frame70.tif
         │   └── ...   (75 TIFF slices in total, covering ages 5–40 weeks)
         └── label/
             ├── 5W_130858_frame206.tif
             ├── 5W_130858_frame277.tif
             ├── 10W_100330_frame70.tif
             └── ...   (matching instance-label TIFFs)

   The full annotated 3D training dataset will be made publicly available upon final publication of the paper.

\ 

How to run inference with the pretrained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow these steps to segment oocytes in your own images using the pretrained model.

#. **Prepare your images.** The model expects:

   * **3D grayscale TIFF files** with a single DDX4 fluorescence channel, in ``ZYX`` order.
   * Image resolution of approximately ``5.0 × 0.867 × 0.867`` µm/pixel (Z × Y × X). If your images have a different pixel size, rescale them to this resolution before running inference for optimal results.
   * Any standard bit-depth (``float32``, ``uint16``, etc.) is accepted. BiaPy applies zero-mean unit-variance normalisation internally.
   * Large whole-ovary volumes are fully supported: BiaPy processes them in chunks and saves results as TIFF.

   You can use the images from the `Zenodo dataset <https://zenodo.org/records/19085211>`__ to test the model directly. For questions about image preparation, contact us through the `Image.sc Forum <https://forum.image.sc/>`__ (tag ``biapy``) or `GitHub issues <https://github.com/BiaPyX/BiaPy/issues>`__.

#. **Choose how to run BiaPy.** BiaPy supports six interfaces: GUI, Jupyter/Colab, Galaxy, Docker, CLI, and API. Install BiaPy following the :ref:`installation` instructions for your chosen method.

#. **Get the inference configuration file.** Download it from the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve/tree/main/ByaPy_YAML>`__, or copy the YAML below into a file called ``ovarian-reserve-inference.yaml``:

   .. code-block:: yaml

      SYSTEM:
        NUM_WORKERS: 5

      PROBLEM:
        TYPE: INSTANCE_SEG
        NDIM: 3D
        INSTANCE_SEG:
          DATA_CHANNELS: BCD
          DATA_MW_TH_TYPE: manual
          DATA_MW_TH_BINARY_MASK: 0.5
          DATA_MW_TH_CONTOUR: 0.2
          DATA_MW_TH_DISTANCE: 1.0
          DATA_REMOVE_SMALL_OBJ_BEFORE: 10
          DATA_REMOVE_BEFORE_MW: true
          DISTANCE_CHANNEL_MASK: true

      DATA:
        NORMALIZATION:
          TYPE: zero_mean_unit_variance
        PATCH_SIZE: (40,128,128,1)
        REFLECT_TO_COMPLETE_SHAPE: true
        EXTRACT_RANDOM_PATCH: false
        TEST:
          PATH: "/home/user/raw_ovary/"      # <-- set this to your images folder
          LOAD_GT: false
          IN_MEMORY: false
          RESOLUTION: (5,0.867,0.867)
          OVERLAP: (0,0,0)
          PADDING: (10,50,50)

      MODEL:
        ARCHITECTURE: resunet
        FEATURE_MAPS: [48, 64, 80, 96]
        Z_DOWN: [1, 1, 1]
        DROPOUT_VALUES: [0, 0, 0, 0]
        LOAD_CHECKPOINT: true

      PATHS:
        CHECKPOINT_FILE: "/home/user/ovarian_reserve_model.pt"  # <-- set this to your checkpoint

      LOSS:
        TYPE: CE

      TRAIN:
        ENABLE: false

      TEST:
        ENABLE: true
        REDUCE_MEMORY: true
        BY_CHUNKS:
          ENABLE: true
          FORMAT: zarr
          SAVE_OUT_TIF: true
          INPUT_IMG_AXES_ORDER: ZYXC
          WORKFLOW_PROCESS:
            ENABLE: true
            TYPE: entire_pred
        POST_PROCESSING:
          MEASURE_PROPERTIES:
            ENABLE: true
            REMOVE_BY_PROPERTIES:
              ENABLE: true
              PROPS: [["npixels"], ["sphericity", "npixels"]]
              VALUES: [[150], [0.01, 2000]]
              SIGNS: [["le"], ["lt", "lt"]]

   The two lines you **must** edit are:

   * ``DATA.TEST.PATH`` — folder containing your input TIFF images.
   * ``PATHS.CHECKPOINT_FILE`` — path to the downloaded pretrained model weights.

#. **Download the pretrained model checkpoint** (available soon — follow updates on the `BioImage Model Zoo <https://bioimage.io>`__ or the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve>`__).

#. **Run BiaPy** using your preferred interface:

.. tabs::

   .. tab:: GUI

      Follow these steps in the BiaPy Graphical User Interface (GUI, version ``1.2.2`` or later):

      #. Click ``"Load and modify workflow"`` and load the YAML configuration file.
      #. If any paths are invalid you will be redirected to a window to fix them — update ``DATA.TEST.PATH`` and ``PATHS.CHECKPOINT_FILE`` as needed.
      #. Confirm or rename the configuration file and click ``"Save file"``.
      #. In the ``"Run Workflow"`` window, select an output folder for the results.
      #. Optionally click ``"Check file"`` to validate the configuration before running.
      #. Click ``"Run"`` to start the workflow.

      .. figure:: ../../img/gui/GUI_load_yaml_generic.png
         :align: center
         :figwidth: 500px

         Click on ``"Load and modify workflow"`` and load the YAML configuration file.

   .. tab:: Jupyter/Colab

      #. Open the BiaPy inference notebook `here <https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/BiaPy_Inference.ipynb>`__.
      #. Upload the YAML configuration file and the pretrained model checkpoint to the Colab environment as explained in the notebook.
      #. Follow the notebook instructions to run inference.

   .. tab:: Galaxy

      #. Open the BiaPy tool in Galaxy ToolShed at this `link <https://imaging.usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Fbiapy%2Fbiapy%2F3.6.5%2Bgalaxy0&version=latest>`__. New to Galaxy? Check the `Galaxy tutorials <https://training.galaxyproject.org/training-material/topics/introduction/>`__.
      #. Upload your images, the YAML file, and the checkpoint to Galaxy.
      #. For ``"Do you have a configuration file?"`` select ``"Yes, I already have one and I want to run BiaPy directly"``.
      #. Select your configuration file under ``"Select a configuration file"``.
      #. Select your checkpoint under ``"Select the model checkpoint (if needed)"``.
      #. Leave ``"If train is enabled, select the training images"`` empty.
      #. Under ``"If test is enabled, select the test images"``, choose the images you want to predict.
      #. In the output section, select ``"Test predictions (if exist)"`` and ``"Post-processed test predictions (if exist)"``.
      #. Run the job.

   .. tab:: Docker

      #. Edit the configuration file to set the two required paths:

         .. code-block:: yaml

            DATA:
              TEST:
                PATH: "/home/user/raw_ovary/"
            PATHS:
              CHECKPOINT_FILE: "/home/user/ovarian_reserve_model.pt"

      #. Requires BiaPy version ``3.6.8`` or newer. Run:

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

         Results will be saved to ``/home/user/exp_results/my_ovarian_reserve_test/my_ovarian_reserve_test_1/results``: instance masks under ``per_image_instances``, post-processed outputs under ``per_image_post_processing``.

   .. tab:: CLI

      #. Edit the configuration file to set the two required paths:

         .. code-block:: yaml

            DATA:
              TEST:
                PATH: "/home/user/raw_ovary/"
            PATHS:
              CHECKPOINT_FILE: "/home/user/ovarian_reserve_model.pt"

      #. Requires BiaPy version ``3.6.8`` or newer. Run:

         .. code-block:: bash

            job_cfg_file=/home/user/ovarian-reserve-inference.yaml
            result_dir=/home/user/exp_results
            job_name=my_ovarian_reserve_test
            job_counter=1
            gpu_number=0

            conda activate BiaPy_env

            biapy \
               --config $job_cfg_file \
               --result_dir $result_dir  \
               --name $job_name    \
               --run_id $job_counter  \
               --gpu "$gpu_number"

         Results will be saved to ``/home/user/exp_results/my_ovarian_reserve_test/my_ovarian_reserve_test_1/results``: instance masks under ``per_image_instances``, post-processed outputs under ``per_image_post_processing``.

   .. tab:: API

      #. Edit the configuration file to set the two required paths:

         .. code-block:: yaml

            DATA:
              TEST:
                PATH: "/home/user/raw_ovary/"
            PATHS:
              CHECKPOINT_FILE: "/home/user/ovarian_reserve_model.pt"

      #. Requires BiaPy version ``3.6.8`` or newer. Add the following to your Python script:

         .. code-block:: python

            from biapy import BiaPy

            config_path = "/home/user/ovarian-reserve-inference.yaml"
            result_dir  = "/home/user/exp_results"
            job_name    = "my_ovarian_reserve_test"
            run_id      = 1
            gpu         = "0"

            biapy = BiaPy(config_path, result_dir=result_dir, name=job_name, run_id=run_id, gpu=gpu)
            biapy.run_job()

         Results will be saved to ``/home/user/exp_results/my_ovarian_reserve_test/my_ovarian_reserve_test_1/results``: instance masks under ``per_image_instances``, post-processed outputs under ``per_image_post_processing``.

After obtaining the segmented oocyte instances, you can analyse the results further using the scripts in the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve>`__:

* **Oocyte density** (``oocyte density`` folder): counts oocytes per unit volume across the ovary.
* **Radial quantification** (``radial quantification`` folder): maps the spatial distribution of oocytes along the radial axis of the ovary.

These scripts require: ``pandas``, ``numpy``, ``matplotlib``, ``seaborn``, ``scipy``, and ``vedo``.


Training the model from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to train the model from scratch — for example to reproduce the results of :cite:`ovarianreserve2025`, to fine-tune on your own data, or to experiment with different hyperparameters — follow the steps below. We assume you have already :ref:`installed BiaPy <installation>` and are comfortable running commands from a terminal.

#. **Download the training dataset.** Download ``oocyte_training.zip`` from `Google Drive <https://drive.google.com/file/d/1xA2b9nY1KuIGC-ZjYg--MXQ8r8GSyOwP/view?usp=sharing>`__ and extract it. You will obtain a folder with two sub-directories:

   * ``raw/`` — 75 2D TIFF image slices extracted from 3D ovary volumes (ages 5–40 weeks).
   * ``label/`` — corresponding instance-label TIFFs (each oocyte has a unique integer ID).

   A 10% validation split is created automatically from the training data at runtime (``VAL.FROM_TRAIN: true``), so no manual split is needed.

   For the test set, use the whole-ovary volumes from Zenodo (``raw_ovary/`` folder) as described in the `Dataset`_ section above.

#. **Get the training configuration file.** Download it from the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve/tree/main/ByaPy_YAML>`__, or copy the YAML below into a file called ``ovarian-reserve-training.yaml``. Then set the four paths marked with ``<--``:

   .. code-block:: yaml

      SYSTEM:
        NUM_WORKERS: 5

      PROBLEM:
        TYPE: INSTANCE_SEG
        NDIM: 3D
        INSTANCE_SEG:
          DATA_CHANNELS: BCD
          DATA_MW_TH_TYPE: manual
          DATA_MW_TH_BINARY_MASK: 0.5
          DATA_MW_TH_CONTOUR: 0.2
          DATA_MW_TH_DISTANCE: 1.0
          DATA_REMOVE_SMALL_OBJ_BEFORE: 10
          DATA_REMOVE_BEFORE_MW: true

      DATA:
        NORMALIZATION:
          TYPE: zero_mean_unit_variance
        PATCH_SIZE: (40,128,128,1)
        REFLECT_TO_COMPLETE_SHAPE: true
        CHECK_GENERATORS: false
        EXTRACT_RANDOM_PATCH: false
        TRAIN:
          PATH: "/home/user/oocyte_training/raw/"     # <-- path to training raw images
          GT_PATH: "/home/user/oocyte_training/label/" # <-- path to training label images
          IN_MEMORY: true
        VAL:
          FROM_TRAIN: true
          SPLIT_TRAIN: 0.1
        TEST:
          PATH: "/home/user/raw_ovary/"               # <-- path to Zenodo test images
          LOAD_GT: false
          RESOLUTION: (5,0.867,0.867)
          CHECK_DATA: false
          PADDING: (10,50,50)
          OVERLAP: (0,0,0)
          FILTER_SAMPLES:
            ENABLE: true
            PROPS: [["mean"]]
            VALUES: [[50.0]]
            SIGNS: [["lt"]]

      AUGMENTOR:
        ENABLE: true
        DA_PROB: 0.5
        RANDOM_ROT: true
        VFLIP: true
        HFLIP: true
        ZFLIP: true
        AFFINE_MODE: reflect
        GRIDMASK: true
        GRID_RATIO: 0.7
        ELASTIC: true

      MODEL:
        ARCHITECTURE: resunet
        FEATURE_MAPS: [48, 64, 80, 96]
        Z_DOWN: [1, 1, 1]
        DROPOUT_VALUES: [0, 0, 0, 0]
        LOAD_CHECKPOINT: false              # <-- set to true to fine-tune from a checkpoint

      LOSS:
        TYPE: CE

      TRAIN:
        ENABLE: true
        OPTIMIZER: ADAMW
        LR: 1.0e-4
        BATCH_SIZE: 4
        EPOCHS: 800
        PATIENCE: 100
        LR_SCHEDULER:
          NAME: reduceonplateau
          MIN_LR: 1.0e-6
          REDUCEONPLATEAU_PATIENCE: 80

      TEST:
        ENABLE: true
        REDUCE_MEMORY: true
        REUSE_PREDICTIONS: false
        BY_CHUNKS:
          ENABLE: true
          FORMAT: zarr
          SAVE_OUT_TIF: true
          INPUT_IMG_AXES_ORDER: ZYXC
          WORKFLOW_PROCESS:
            ENABLE: true
            TYPE: entire_pred
        POST_PROCESSING:
          MEASURE_PROPERTIES:
            ENABLE: true
            REMOVE_BY_PROPERTIES:
              ENABLE: true
              PROPS: [["npixels"], ["sphericity", "npixels"]]
              VALUES: [[150], [0.01, 2000]]
              SIGNS: [["le"], ["lt", "lt"]]

#. **Run BiaPy** using the CLI:

   .. code-block:: bash

      job_cfg_file=/home/user/ovarian-reserve-training.yaml
      result_dir=/home/user/exp_results
      job_name=my_ovarian_reserve_training
      job_counter=1
      gpu_number=0

      conda activate BiaPy_env

      biapy \
            --config $job_cfg_file \
            --result_dir $result_dir  \
            --name $job_name    \
            --run_id $job_counter  \
            --gpu "$gpu_number"

   Results will be saved to ``/home/user/exp_results/my_ovarian_reserve_training/my_ovarian_reserve_training_1/results``. Test-set instance masks are under ``per_image_instances`` and post-processed outputs under ``per_image_post_processing``.

#. **Analyse the results** using the scripts in the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve>`__:

   * **Oocyte density** (``oocyte density`` folder): computes the number of oocytes per unit volume.
   * **Radial quantification** (``radial quantification`` folder): maps the spatial distribution of oocytes from the ovarian surface to the core.

   These scripts reproduce the quantitative analyses presented in :cite:`ovarianreserve2025`.

