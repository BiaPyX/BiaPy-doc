.. _image-to-image:

Image to image
--------------

Description of the task
~~~~~~~~~~~~~~~~~~~~~~~

The goal of this workflow is to **translate/map input images into target images**. Because of that, this task is commonly known as "image to image", and can be used for different purposes such as **image inpainting, colorization or even super-resolution** (with a scale factor of x1). In bioimage analysis, this workflow can be used for `virtual staining <https://www.cell.com/trends/biotechnology/fulltext/S0167-7799%2824%2900038-6>`__, i.e., training a model to produce stained images from an unstained tissue image, or through transferring information from one stain to another.

An example of this task is displayed in the figure below, with a pair of (input-output) fluorescence microscopy images:

.. role:: raw-html(raw)
    :format: html


.. list-table:: 
  :align: center
  :width: 680px

  * - .. figure:: ../img/i2i/i2i_raw.png
         :align: center
         :width: 300px

         Input image (lifeact-RFP) from the :raw-html:`<br />` `ZeroCostDL4Mic pix2pix example dataset <https://zenodo.org/records/3941889#.XxrkzWMzaV4>`__.

    - .. figure:: ../img/i2i/i2i_target.png
         :align: center
         :width: 300px

         Target image (sir-DNA) from the :raw-html:`<br />` `ZeroCostDL4Mic pix2pix example dataset <https://zenodo.org/records/3941889#.XxrkzWMzaV4>`__.


Inputs and outputs
~~~~~~~~~~~~~~~~~~
The image-to-image workflows in BiaPy expect a series of **folders** as input:

* **Training Raw Images**: A folder that contains the unprocessed (single-channel or multi-channel) images that will be used to train the model.
  
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image to image*, twice *Continue*, under *General options* > *Train data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/i2i/GUI-general-options.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image-to-image notebook, go to *Paths for Input Images and Output Files*, edit the field **train_raw_data_path**:
        
        .. image:: ../img/i2i/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TRAIN.PATH`` with the absolute path to the folder with your training raw images.



* **Training Target Images**: A folder that contains the target (single-channel) images for training. Ensure the number and dimensions match the training raw images.
  
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image to image*, twice *Continue*, under *General options* > *Train data*, click on the *Browse* button of **Input target folder**:

        .. image:: ../img/i2i/GUI-general-options.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image-to-image notebook, go to *Paths for Input Images and Output Files*, edit the field **train_target_data_path**:
        
        .. image:: ../img/i2i/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TRAIN.GT_PATH`` with the absolute path to the folder with your training target images.

* .. raw:: html

      <b><span style="color: darkgreen;">[Optional]</span> Test Raw Images</b>: A folder that contains the images to evaluate the model's performance.
 
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image to image*, three times *Continue*, under *General options* > *Test data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/i2i/GUI-test-data.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image-to-image notebook, go to *Paths for Input Images and Output Files*, edit the field **test_raw_data_path**:
        
        .. image:: ../img/i2i/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TEST.PATH`` with the absolute path to the folder with your test raw images.

* .. raw:: html

      <b><span style="color: darkgreen;">[Optional]</span> Test Target Images</b>: A folder that contains the target images for testing. Again, ensure their count and sizes align with the test raw images.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image to image*, three times *Continue*, under *General options* > *Test data*, select "Yes" in the *Do you have target test data?* field, and then click on the *Browse* button of **Input target folder**:

        .. image:: ../img/i2i/GUI-test-data-gt.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image-to-image notebook, go to *Paths for Input Images and Output Files*, edit the field **test_target_data_path**:
        
        .. image:: ../img/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TEST._GT_PATH`` with the absolute path to the folder with your test target images.


Upon successful execution, a directory will be generated with the image-to-image translation results. Therefore, you will need to define:

* **Output Folder**: A designated path to save the image-to-image outcomes.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, click on the *Browse* button of **Output folder to save the results**:

        .. image:: ../img/i2i/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image-to-image notebook, go to *Paths for Input Images and Output Files*, edit the field **output_path**:
        
        .. image:: ../img/i2i/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--result_dir`` flag. See the *Command line* configuration of :ref:`image_to_image_data_run` for a full example.


.. list-table::
  :align: center

  * - .. figure:: ../img/i2i/Inputs-outputs.svg
         :align: center
         :width: 500
         :alt: Graphical description of minimal inputs and outputs in BiaPy for image-to-image translation.
        
         **BiaPy input and output folders for image-to-image translation**.



.. _i2i2_data_prep:

Data preparation
~~~~~~~~~~~~~~~~

To ensure the proper operation of the library the data directory tree should be something like this: 

.. collapse:: Expand directory tree 

    .. code-block:: bash
  
      dataset/
      ├── train
      │   ├── x
      │   │   ├── training-0001.tif
      │   │   ├── training-0002.tif
      │   │   ├── . . .
      │   │   ├── training-9999.tif
      │   └── y
      │       ├── training_groundtruth-0001.tif
      │       ├── training_groundtruth-0002.tif
      │       ├── . . .
      │       ├── training_groundtruth-9999.tif
      └── test
          ├── x
          │   ├── testing-0001.tif
          │   ├── testing-0002.tif
          │   ├── . . .
          │   ├── testing-9999.tif
          └── y
              ├── testing_groundtruth-0001.tif
              ├── testing_groundtruth-0002.tif
              ├── . . .
              ├── testing_groundtruth-9999.tif

\

.. warning:: Ensure that images and their corresponding masks are sorted in the same way. A common approach is to fill with zeros the image number added to the filenames (as in the example). 

Configuration                                                                                                                 
~~~~~~~~~~~~~

Find in `templates/image-to-image <https://github.com/BiaPyX/BiaPy/tree/master/templates/image-to-image>`__ folder of BiaPy a few YAML configuration templates for this workflow. 


Special workflow configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here some special configuration options that can be selected in this workflow are described:

* If each training sample is composed by several images, e.g. transformed versions of the sample, you need to set ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER``. Find an example of this configuration in the `LightMyCells tutorial <https://biapy.readthedocs.io/en/latest/tutorials/image-to-image/lightmycells.html>`__. 

* **Metrics**: during the inference phase the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is ``True``. In the case of image-to-image the **Peak signal-to-noise ratio** (`PSNR <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__) metric is calculated when the target image is reconstructed from individual patches.

.. _i2i_data_run:

Run
~~~

.. tabs::

   .. tab:: GUI

        Select image-to-image workflow during the creation of a new configuration file:

        .. image:: ../img/gui/biapy_gui_i2i.png
            :align: center 

   .. tab:: Google Colab

        Two different options depending on the image dimension: 

        .. |sr_2D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/image_to_image/BiaPy_2D_Image_to_Image.ipynb

        * 2D: |sr_2D_colablink|

        .. |sr_3D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/image_to_image/BiaPy_3D_Image_to_Image.ipynb

        * 3D: |sr_3D_colablink|

   .. tab:: Docker

        `Open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. For instance, using `2d_image-to-image.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/image-to-image/2d_image-to-image.yaml>`__ template file, the code can be run as follows:

        .. code-block:: bash                                                                                                    

            # Configuration file
            job_cfg_file=/home/user/2d_image-to-image.yaml
            # Path to the data directory
            data_dir=/home/user/data
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results
            # Just a name for the job
            job_name=my_2d_image_to_image
            # Number that should be increased when one need to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0

            sudo docker run --rm \
                --gpus "device=$gpu_number" \
                --mount type=bind,source=$job_cfg_file,target=$job_cfg_file \
                --mount type=bind,source=$result_dir,target=$result_dir \
                --mount type=bind,source=$data_dir,target=$data_dir \
                BiaPyX/biapy \
                    -cfg $job_cfg_file \
                    -rdir $result_dir \
                    -name $job_name \
                    -rid $job_counter \
                    -gpu "cuda:$gpu_number"

        .. note:: 
            Note that ``data_dir`` must contain all the paths ``DATA.*.PATH`` and ``DATA.*.GT_PATH`` so the container can find them. For instance, if you want to only train in this example ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` could be ``/home/user/data/train/x`` and ``/home/user/data/train/y`` respectively. 

   .. tab:: Command line

        `Open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. For instance, using `2d_image-to-image.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/image-to-image/2d_image-to-image.yaml>`__ template file, the code can be run as follows:

        .. code-block:: bash
            
            # Configuration file
            job_cfg_file=/home/user/2d_image-to-image.yaml       
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results  
            # Just a name for the job
            job_name=my_2d_image_to_image      
            # Number that should be increased when one need to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0                   

            # Load the environment
            conda activate BiaPy_env

            python -u main.py \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu "cuda:$gpu_number"  

        For multi-GPU training you can call BiaPy as follows:

        .. code-block:: bash
            
            # First check where is your biapy command (you need it in the below command)
            # $ which biapy
            # > /home/user/anaconda3/envs/BiaPy_env/bin/biapy

            gpu_number="0, 1, 2"
            python -u -m torch.distributed.run \
                --nproc_per_node=3 \
                /home/user/anaconda3/envs/BiaPy_env/bin/biapy \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu "cuda:$gpu_number"  

        ``nproc_per_node`` needs to be equal to the number of GPUs you are using (e.g. ``gpu_number`` length).
        
.. _image_to_image_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. list-table:: 
  :align: center
  :width: 680px

  * - .. figure:: ../img/i2i/i2i_pred.png
         :align: center
         :width: 300px

         Predicted image.

    - .. figure:: ../img/i2i/i2i_target2.png
         :align: center
         :width: 300px

         Target image.


Following the example, you should see that the directory ``/home/user/exp_results/my_2d_image_to_image`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: 

.. collapse:: Expand directory tree 

    .. code-block:: bash
        
      my_2d_image_to_image/
      ├── config_files
      │   └── 2d_image-to-image.yaml                                                                                                           
      ├── checkpoints
      │   └── my_2d_image-to-image_1-checkpoint-best.pth
      └── results
          ├── my_2d_image_to_image_1
          ├── . . .
          └── my_2d_image_to_image_5
              ├── aug
              │   └── .tif files
              ├── charts
              │   ├── my_2d_image_to_image_1_*.png
              │   └── my_2d_image_to_image_1_loss.png
              ├── per_image
              │   ├── .tif files
              │   └── .zarr files (or.h5)
              ├── full_image
              │   └── .tif files
              ├── train_logs
              └── tensorboard

\

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

  * ``2d_image-to-image.yaml``: YAML configuration file used (it will be overwrited every time the code is run)

* ``checkpoints``, *optional*: directory where model's weights are stored. Only created when ``TRAIN.ENABLE`` is ``True`` and the model is trained for at least one epoch. 

  * ``my_2d_image-to-image_1-checkpoint-best.pth``, *optional*: checkpoint file (best in validation) where the model's weights are stored among other information. Only created when the model is trained for at least one epoch. 

  * ``normalization_mean_value.npy``, *optional*: normalization mean value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.
  
  * ``normalization_std_value.npy``, *optional*: normalization std value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed. Can contain:

  * ``my_2d_image_to_image_1``: run 1 experiment folder. Can contain:

    * ``aug``, *optional*: image augmentation samples. Only created if ``AUGMENTOR.AUG_SAMPLES`` is ``True``.

    * ``charts``, *optional*. Only created when ``TRAIN.ENABLE`` is ``True`` and epochs trained are more or equal ``LOG.CHART_CREATION_FREQ``. Can contain:

      * ``my_2d_image_to_image_1_*.png``: plot of each metric used during training.

      * ``my_2d_image_to_image_1_loss.png``: loss over epochs plot. 

    * ``per_image``, *optional*: only created if ``TEST.FULL_IMG`` is ``False``. Can contain:

      * ``.tif files``: reconstructed images from patches.   

      * ``.zarr files (or.h5)``, *optional*: reconstructed images from patches. Created when ``TEST.BY_CHUNKS.ENABLE`` is ``True``.

    * ``full_image``, *optional*: only created if ``TEST.FULL_IMG`` is ``True``. Can contain:

      * ``.tif files``: full image predictions.

    * ``train_logs``: each row represents a summary of each epoch stats. Only avaialable if training was done.

    * ``tensorboard``: tensorboard logs.

.. note:: 
   Here, for visualization purposes, only ``my_2d_image_to_image_1`` has been described but ``my_2d_image_to_image_2``, ``my_2d_image_to_image_3``, ``my_2d_image_to_image_4`` and ``my_2d_image_to_image_5`` will follow the same structure.


