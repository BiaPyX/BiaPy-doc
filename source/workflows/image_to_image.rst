.. _image-to-image:

Image to Image
--------------

The goal of this workflow aims at translating/mapping input images into target images. This workflow is as the super-resolution one but with no upsampling, e.g. with the scaling factor to x1.

* **Input:** 
    
  * image (single-channel or multi-channel). E.g. image with shape ``(500, 500, 1)`` ``(y, x, channels)`` in ``2D`` or ``(100, 500, 500, 1)`` ``(z, y, x, channels)`` in ``3D``.  

* **Output:**

  * image. 

In the figure below an example of paired microscopy images (fluorescence) of lifeact-RFP (**input**) and SiR-DNA is depicted (**output**). The images were obtained from `ZeroCostDL4Mic pix2pix example training and test dataset <https://zenodo.org/records/3941889#.XxrkzWMzaV4>`__:

.. list-table:: 

  * - .. figure:: ../img/i2i/i2i_raw.png
         :align: center

         Input image.

    - .. figure:: ../img/i2i/i2i_target.png
         :align: center

         Target image.

.. _i2i2_data_prep:

Data preparation
~~~~~~~~~~~~~~~~

To ensure the proper operation of the library the data directory tree should be something like this: 

.. collapse:: Expand directory tree 

    .. code-block:: bash
  
      dataset/
      ├── train
      │   ├── x
      │   │   ├── training-0001.tif
      │   │   ├── training-0002.tif
      │   │   ├── . . .
      │   │   ├── training-9999.tif
      │   └── y
      │       ├── training_groundtruth-0001.tif
      │       ├── training_groundtruth-0002.tif
      │       ├── . . .
      │       ├── training_groundtruth-9999.tif
      └── test
          ├── x
          │   ├── testing-0001.tif
          │   ├── testing-0002.tif
          │   ├── . . .
          │   ├── testing-9999.tif
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

        .. image:: ../img/gui/biapy_gui_i2i.jpg
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

        ``nproc_per_node`` need to be equal to the number of GPUs you are using (e.g. ``gpu_number`` length).
        
.. _image_to_image_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. list-table:: 

  * - .. figure:: ../img/i2i/i2i_pred.png
         :align: center

         Predicted image.

    - .. figure:: ../img/i2i/i2i_target2.png
         :align: center

         Target image.


Following the example, you should see that the directory ``/home/user/exp_results/my_2d_image_to_image`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: 

.. collapse:: Expand directory tree 

    .. code-block:: bash
        
      my_2d_image_to_image/
      ├── config_files/
      │   └── 2d_image-to-image.yaml                                                                                                           
      ├── checkpoints
      │   └── my_2d_image-to-image_1-checkpoint-best.pth
      └── results
         ├── my_2d_image_to_image_1
          ├── . . .
          └── my_2d_image_to_image_5
              ├── aug
              │   └── .tif files
             ├── charts
              │   ├── my_2d_image_to_image_1_*.png
              │   ├── my_2d_image_to_image_1_loss.png
              │   └── model_plot_my_2d_image_to_image_1.png
             ├── per_image
              │   └── .tif files
              ├── train_logs
              └── tensorboard

\

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

  * ``2d_image-to-image.yaml``: YAML configuration file used (it will be overwrited every time the code is run)

* ``checkpoints``: directory where model's weights are stored.

  * ``my_2d_image-to-image_1-checkpoint-best.pth``: checkpoint file (best in validation) where the model's weights are stored among other information.

  * ``normalization_mean_value.npy``: normalization mean value (only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``). Is saved to not calculate it everytime and to use it in inference.  
  
  * ``normalization_std_value.npy``: normalization std value (only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``). Is saved to not calculate it everytime and to use it in inference. 
  
* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

  * ``my_2d_image_to_image_1``: run 1 experiment folder. 

    * ``aug``: image augmentation samples.

    * ``charts``:  

      * ``my_2d_image_to_image_1_*.png``: Plot of each metric used during training.

      * ``my_2d_image_to_image_1_loss.png``: Loss over epochs plot (when training is done). 

      * ``model_plot_my_2d_image_to_image_1.png``: plot of the model.

    * ``per_image``:

      * ``.tif files``: reconstructed images from patches.   

    * ``train_logs``: each row represents a summary of each epoch stats. Only avaialable if training was done.

    * ``tensorboard``: Tensorboard logs.

.. note:: 
   Here, for visualization purposes, only ``my_2d_image_to_image_1`` has been described but ``my_2d_image_to_image_2``, ``my_2d_image_to_image_3``, ``my_2d_image_to_image_4`` and ``my_2d_image_to_image_5`` will follow the same structure.


