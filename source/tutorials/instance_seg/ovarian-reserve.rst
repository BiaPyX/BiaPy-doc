.. _ovarian_reserve_tutorial:

Ovarian Reserve: 3D Instance Segmentation of Oocytes
-----------------------------------------------------

This tutorial shows how to run **BiaPy** to perform **3D instance segmentation of oocytes** in whole-mount mouse ovary samples, as described in :cite:`ovarianreserve2025`. The goal is the automatic detection and segmentation of individual oocytes from 3D fluorescence images stained for **DDX4**, a germ-cell-specific marker, acquired by light-sheet fluorescence (SPIM) microscopy.

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

   The pretrained model will be available soon in the `BioImage Model Zoo <https://bioimage.io>`__ and as a regular **PyTorch** checkpoint file. Stay tuned for updates.


Dataset
~~~~~~~

A representative subset of the full dataset is publicly available in `Zenodo <https://zenodo.org/records/19085211>`__ :cite:`ovarianreserve2025`. It contains **seven whole-mount ovarian samples** from mice of different ages imaged by SPIM microscopy and stained for **DDX4**. Each sample corresponds to one of the following ages: **5, 10, 22, 30, 40, 50,** and **60 weeks**. The full dataset, including training annotations, will be made publicly available upon final publication of the paper. 

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

\ 

How to use the pretrained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the pretrained model to predict new images using several methods supported by BiaPy. Follow the steps below:

#. Prepare your images according to the model requirements:

   * **3D grayscale images** with a single channel (DDX4 staining). Images should be stored as **TIFF** files.

   * The model expects images to be 3D volumes in ``ZYX`` order with a single fluorescence channel.

   * The expected image resolution is approximately ``5.0 × 0.867 × 0.867`` µm/pixel (Z × Y × X). If your images have a different pixel size, rescale them accordingly before running inference to ensure optimal performance.

   * The images are expected to be in a standard fluorescence format (e.g., ``float32`` or ``uint16``). Large whole-mount ovary volumes are supported via patch-based processing and chunked I/O (Zarr format).

   If you have doubts, check the dataset provided in `Zenodo <https://zenodo.org/records/19085211>`__ for examples. For questions regarding image preparation, contact us via the `Image.sc Forum <https://forum.image.sc/>`__ (tag ``biapy``) or through our `GitHub issues page <https://github.com/BiaPyX/BiaPy/issues>`__.

   .. note::

      You can use the images from the Zenodo dataset to test the model directly.

#. Decide which method you want to use to run BiaPy: GUI, Jupyter/Colab, Galaxy, Docker, CLI, or API. Based on that, follow the installation instructions described in the :ref:`installation` section.

#. Download the YAML configuration file for inference from the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve/tree/main/ByaPy_YAML>`__ or paste the configuration below into a file called ``ovarian-reserve-inference.yaml``:

   .. code-block:: yaml

      DATA:
        EXTRACT_RANDOM_PATCH: false
        NORMALIZATION:
          CUSTOM_MEAN: -1.0
          CUSTOM_STD: -1.0
          TYPE: custom
        PATCH_SIZE: (40,128,128,1)
        REFLECT_TO_COMPLETE_SHAPE: true
        TEST:
          ARGMAX_TO_OUTPUT: true
          IN_MEMORY: false
          LOAD_GT: false
          OVERLAP: (0,0,0)
          PADDING: (10,50,50)
          PATH: "insert the folder path of your data in tiff format"
          RESOLUTION: (5,0.867,0.867)
      LOSS:
        CLASS_REBALANCE: true
        TYPE: CE
      MODEL:
        ARCHITECTURE: resunet
        DROPOUT_VALUES:
        - 0
        - 0
        - 0
        - 0
        FEATURE_MAPS:
        - 48
        - 64
        - 80
        - 96
        LOAD_CHECKPOINT: true
        N_CLASSES: 2
        SOURCE: biapy
        Z_DOWN:
        - 1
        - 1
        - 1
      PATHS:
        CHECKPOINT_FILE: "insert the path of the checkpoint file"
      PROBLEM:
        INSTANCE_SEG:
          DATA_CHANNELS: BCD
          DATA_CHANNEL_WEIGHTS: (1,1)
          DATA_MW_TH_BINARY_MASK: 0.5
          DATA_MW_TH_CONTOUR: 0.2
          DATA_MW_TH_DISTANCE: 1.0
          DATA_MW_TH_FOREGROUND: 0.3
          DATA_MW_TH_TYPE: manual
          DATA_REMOVE_BEFORE_MW: true
          DATA_REMOVE_SMALL_OBJ_BEFORE: 10
          DISTANCE_CHANNEL_MASK: true
        NDIM: 3D
        TYPE: INSTANCE_SEG
      SYSTEM:
        NUM_CPUS: -1
        NUM_WORKERS: 5
        SEED: 0
      TEST:
        BY_CHUNKS:
          ENABLE: true
          FLUSH_EACH: 100
          FORMAT: zarr
          INPUT_IMG_AXES_ORDER: ZYXC
          INPUT_MASK_AXES_ORDER: TZCYX
          SAVE_OUT_TIF: true
          WORKFLOW_PROCESS:
            ENABLE: true
            TYPE: entire_pred
        ENABLE: true
        MATCHING_STATS: true
        MATCHING_STATS_THS:
        - 0.3
        - 0.5
        - 0.75
        POST_PROCESSING:
          MEASURE_PROPERTIES:
            ENABLE: true
            REMOVE_BY_PROPERTIES:
              ENABLE: true
              PROPS:
              - - npixels
              - - sphericity
                - npixels
              SIGNS:
              - - le
              - - lt
                - lt
              VALUES:
              - - 150
              - - 0.01
                - 2000
        REDUCE_MEMORY: true
        VERBOSE: true
      TRAIN:
        ENABLE: false

   Make sure to set:

   * ``DATA.TEST.PATH`` to the folder containing your input TIFF images.
   * ``PATHS.CHECKPOINT_FILE`` to the path of the downloaded pretrained model weights.

#. Download the pretrained model checkpoint (available soon — follow updates on the `BioImage Model Zoo <https://bioimage.io>`__ or the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve>`__).

#. Then follow the instructions corresponding to your selected method to run BiaPy with the downloaded configuration file.

.. tabs::

   .. tab:: GUI

      To begin, please follow these steps within the Graphical User Interface (GUI) (version ``1.2.2`` or later):

      #. Click on ``"Load and modify workflow"`` button in BiaPy GUI and load the YAML configuration file.
      #. If some of the paths in the YAML are not valid, you will be redirected to a window to modify them. Update ``DATA.TEST.PATH`` and ``PATHS.CHECKPOINT_FILE`` with the correct paths.
      #. Once all paths are valid, you can change the configuration filename if you want and click on ``"Save file"``.
      #. You will be redirected to the ``"Run Workflow"`` window. The recently loaded YAML configuration will be selected automatically, and a job name will be pre-filled based on the configuration filename. Select an output folder where all results will be saved.
      #. Optionally check the configuration file consistency before running BiaPy by clicking on the ``"Check file"`` button.
      #. Start the workflow by clicking the ``"Run"`` button.

      .. figure:: ../../img/gui/GUI_load_yaml_generic.png
         :align: center
         :figwidth: 500px

         Click on ``"Load and modify workflow"`` and load the YAML configuration file.

   .. tab:: Jupyter/Colab

      You can predict new images using the pretrained model in a Jupyter or Colab notebook. Follow these steps:

      #. Open our notebook prepared to do inference with the pretrained models `here <https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/BiaPy_Inference.ipynb>`__.
      #. Upload the YAML configuration file to the Colab environment as explained in the notebook.
      #. Upload the pretrained model checkpoint to the Colab environment.
      #. Follow the instructions in the notebook to run inference with the pretrained model.

   .. tab:: Galaxy

      #. Open BiaPy tool in Galaxy ToolShed at this `link <https://imaging.usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Fbiapy%2Fbiapy%2F3.6.5%2Bgalaxy0&version=latest>`__. If you are not familiar with Galaxy, you can check the `Galaxy tutorials <https://training.galaxyproject.org/training-material/topics/introduction/>`__ for further information.
      #. Upload your images, the YAML configuration file, and the pretrained model checkpoint to Galaxy.
      #. For the question ``"Do you have a configuration file?"`` select ``"Yes, I already have one and I want to run BiaPy directly"``.
      #. Select the uploaded configuration file in the ``"Select a configuration file"`` option.
      #. Select the uploaded checkpoint in ``"Select the model checkpoint (if needed)"``.
      #. Leave ``"If train is enabled, select the training images"`` empty, as we are not going to train the model.
      #. In the section ``"If test is enabled, select the test images"`` select the images you want to predict under ``"Specify the test raw images"``.
      #. In the output selection section, select ``"Test predictions (if exist)"`` and ``"Post-processed test predictions (if exist)"`` to download the predicted masks.
      #. Run the job.

   .. tab:: Docker

      #. Modify the configuration file to set the correct paths to the test data and to the pretrained model checkpoint:

         .. code-block:: yaml

            DATA:
              TEST:
                PATH: "/home/user/ovarian_reserve/"
            PATHS:
              CHECKPOINT_FILE: "/home/user/ovarian_reserve_model.pt"

      #. Execution requires BiaPy version ``3.6.8`` or newer. Please run the command below, taking care to configure the relevant file paths:

         .. code-block:: bash

            # Configuration file
            job_cfg_file=/home/user/ovarian-reserve-inference.yaml
            # Path to the data directory
            data_dir=/home/user/ovarian_reserve
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results
            # Just a name for the job
            job_name=my_ovarian_reserve_test
            # Number that should be increased when one needs to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
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

         This will generate the results in ``/home/user/exp_results/my_ovarian_reserve_test/my_ovarian_reserve_test_1/results``. Specifically, in that folder you will find the predicted masks for the test images under ``per_image_instances`` and the post-processed images under ``per_image_post_processing``.

   .. tab:: CLI

      #. Modify the configuration file to set the correct paths to the test data and to the pretrained model checkpoint:

         .. code-block:: yaml

            DATA:
              TEST:
                PATH: "/home/user/ovarian_reserve/"
            PATHS:
              CHECKPOINT_FILE: "/home/user/ovarian_reserve_model.pt"

      #. Execution requires BiaPy version ``3.6.8`` or newer. Please run the command below, taking care to configure the relevant file paths:

         .. code-block:: bash

            # Configuration file
            job_cfg_file=/home/user/ovarian-reserve-inference.yaml
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results
            # Just a name for the job
            job_name=my_ovarian_reserve_test
            # Number that should be increased when one needs to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0

            # Load the environment
            conda activate BiaPy_env

            biapy \
               --config $job_cfg_file \
               --result_dir $result_dir  \
               --name $job_name    \
               --run_id $job_counter  \
               --gpu "$gpu_number"

         This will generate the results in ``/home/user/exp_results/my_ovarian_reserve_test/my_ovarian_reserve_test_1/results``. Specifically, in that folder you will find the predicted masks for the test images under ``per_image_instances`` and the post-processed images under ``per_image_post_processing``.

   .. tab:: API

      #. Modify the configuration file to set the correct paths to the test data and to the pretrained model checkpoint:

         .. code-block:: yaml

            DATA:
              TEST:
                PATH: "/home/user/ovarian_reserve/"
            PATHS:
              CHECKPOINT_FILE: "/home/user/ovarian_reserve_model.pt"

      #. Execution requires BiaPy version ``3.6.8`` or newer. You can plug-in the following code into your Python scripts in order to run the workflow:

         .. code-block:: python

            from biapy import BiaPy

            # Set up your parameters
            config_path = "/path/to/ovarian-reserve-inference.yaml"  # Path to your YAML configuration file
            result_dir = "/home/user/exp_results"                     # Directory to store the results
            job_name = "my_ovarian_reserve_test"                      # Name of the job
            run_id = 1                                                 # Run ID for logging/versioning
            gpu = "0"                                                  # GPU to use (as string, e.g., "0")

            # Create and run the BiaPy job
            biapy = BiaPy(config_path, result_dir=result_dir, name=job_name, run_id=run_id, gpu=gpu)
            biapy.run_job()

         This will generate the results in ``/home/user/exp_results/my_ovarian_reserve_test/my_ovarian_reserve_test_1/results``. Specifically, in that folder you will find the predicted masks for the test images under ``per_image_instances`` and the post-processed images under ``per_image_post_processing``.


After obtaining the segmented oocyte instances, you can further analyze the results using the scripts provided in the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve>`__:

* **Oocyte density**: quantifies the number of oocytes per unit volume across the ovary.
* **Radial quantification**: analyzes the spatial distribution of oocytes along the radial axis of the ovary.

These scripts require the following Python packages: ``pandas``, ``numpy``, ``matplotlib``, ``seaborn``, ``scipy``, and ``vedo``.


Reproducing the results from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to train the model from scratch using your own annotated data or reproduce the results of :cite:`ovarianreserve2025`, you can follow the steps described below. We assume some level of expertise and describe how to do it using the CLI. We also assume you have already :ref:`installed BiaPy <installation>`.

.. note::

   The full annotated training dataset will be made publicly available upon final publication of the paper. In the meantime, you can download the representative inference dataset from `Zenodo <https://zenodo.org/records/19085211>`__ and create your own annotations using standard tools (e.g., `napari <https://napari.org>`__).

#. Download the annotated dataset and organise it into ``train``, ``validation``, and ``test`` splits. Each split should contain a ``raw`` folder with TIFF images and a ``label`` folder with corresponding instance mask TIFF files. For example:

   .. code-block::

      dataset/
      ├── train/
      │   ├── raw/
      │   │   ├── sample_01.tif
      │   │   └── ...
      │   └── label/
      │       ├── sample_01.tif
      │       └── ...
      ├── validation/
      │   ├── raw/
      │   │   └── ...
      │   └── label/
      │       └── ...
      └── test/
          └── raw/
              └── ...

#. Download the YAML configuration file for training from the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve/tree/main/ByaPy_YAML>`__ and modify it to set the correct paths to your training, validation, and test data. For example, if you extracted the dataset to ``/home/user/ovarian_reserve_full/``, set the following paths in the configuration file:

   .. code-block:: yaml

      DATA:
        TRAIN:
          PATH: "/home/user/ovarian_reserve_full/train/raw/"
          GT_PATH: "/home/user/ovarian_reserve_full/train/label/"
        VAL:
          PATH: "/home/user/ovarian_reserve_full/validation/raw/"
          GT_PATH: "/home/user/ovarian_reserve_full/validation/label/"
        TEST:
          PATH: "/home/user/ovarian_reserve_full/test/raw/"

   Also make sure to set ``TRAIN.ENABLE: true`` and remove or leave blank the ``PATHS.CHECKPOINT_FILE`` entry if you want to train from scratch (or set it to a pre-existing checkpoint to fine-tune).

#. Run BiaPy using the CLI as described below:

   .. code-block:: bash

      # Configuration file
      job_cfg_file=/home/user/ovarian-reserve-training.yaml
      # Where the experiment output directory should be created
      result_dir=/home/user/exp_results
      # Just a name for the job
      job_name=my_ovarian_reserve_training
      # Number that should be increased when one needs to run the same job multiple times (reproducibility)
      job_counter=1
      # Number of the GPU to run the job in (according to 'nvidia-smi' command)
      gpu_number=0

      # Load the environment
      conda activate BiaPy_env

      biapy \
            --config $job_cfg_file \
            --result_dir $result_dir  \
            --name $job_name    \
            --run_id $job_counter  \
            --gpu "$gpu_number"

   This will generate the results in ``/home/user/exp_results/my_ovarian_reserve_training/my_ovarian_reserve_training_1/results``. Specifically, in that folder you will find the predicted masks for the test images under ``per_image_instances`` and the post-processed images under ``per_image_post_processing``.

#. After training, evaluate and visualize results using the analysis scripts in the `Boke-Lab GitHub repository <https://github.com/Boke-Lab/ovarian_reserve>`__:

   * **Oocyte density** (``oocyte density`` folder): computes the number of oocytes per unit volume.
   * **Radial quantification** (``radial quantification`` folder): analyzes the spatial distribution of oocytes from the ovary surface to the core.

   These scripts reproduce the quantitative analyses presented in :cite:`ovarianreserve2025`.
