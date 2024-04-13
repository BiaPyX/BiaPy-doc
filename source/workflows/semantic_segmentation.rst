.. _semantic_segmentation:

Semantic segmentation
---------------------

The goal of this workflow is assign a class to each pixel of the input image. 

* **Input:** 

  * Image (single-channel or multi-channel). E.g. image with shape ``(500, 500, 1)`` ``(y, x, channels)`` in ``2D`` or ``(100, 500, 500, 1)`` ``(z, y, x, channels)`` in ``3D``. 
  * Mask (single-channel), where each pixel is labeled with an integer representing a class. E.g. a mask with shape ``(500, 500, 1)`` ``(y, x, channels)`` in ``2D`` or ``(100, 500, 500, 1)`` ``(z, y, x, channels)`` in ``3D``.

* **Output:**

  * Image with each class represented by a unique integer.  

In the figure below an example of this workflow's **input** is depicted. There, only two labels are present in the mask: black pixels, with value ``0``, represent the background and white ones the mitochondria, labeled with ``1``. The number of classes is defined by ``MODEL.N_CLASSES`` variable.

.. list-table:: 

  * - .. figure:: ../img/lucchi_test_0.png
         :align: center
        
         Input image.

    - .. figure:: ../img/lucchi_test_0_gt.png
         :align: center

         Input mask. 

For multiclass case, the same rule applies: the expected mask is single-channel with each class labeled with a different integer. Below an example is depicted where ``0`` is background (black), ``1`` are outlines (pink) and ``2`` nuclei (light blue). 

.. list-table:: 

  * - .. figure:: ../img/semantic_seg/semantic_seg_multiclass_raw.png
         :align: center
        
         Input image.

    - .. figure:: ../img/semantic_seg/semantic_seg_multiclass_mask.png
         :align: center

         Input mask.

The **output** can be: 

- Single-channel image, when ``DATA.TEST.ARGMAX_TO_OUTPUT`` is ``True``, with each class labeled with an integer. 
- Multi-channel image, when ``DATA.TEST.ARGMAX_TO_OUTPUT`` is ``False``, with the same number of channels as classes, and the same pixel in each channel will be the probability (in ``[0-1]`` range) of being of the class that represents that channel number. For instance, with ``3`` classes, e.g. background, mitochondria and contours, the fist channel will represent background, the second mitochondria and the last the contours. 

.. _semantic_segmentation_data_prep:

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

Find in `templates/semantic_segmentation <https://github.com/BiaPyX/BiaPy/tree/master/templates/semantic_segmentation>`__ folder of BiaPy a few YAML configuration templates for this workflow. 

Special workflow configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data loading
************

If you want to select ``DATA.EXTRACT_RANDOM_PATCH`` you can also set ``DATA.PROBABILITY_MAP`` to create a probability map so the patches extracted will have a high probability of having an object in the middle of it. Useful to avoid extracting patches which no foreground class information. That map will be saved in ``PATHS.PROB_MAP_DIR``. Furthermore, in ``PATHS.DA_SAMPLES`` path, i.e. ``aug`` folder by default (see :ref:`semantic_segmentation_results`), two more images will be created so you can check how this probability map is working. These images will have painted a blue square and a red point in its middle, which correspond to the patch area extracted and the central point selected respectively. One image will be named as ``mark_x`` and the other one as ``mark_y``, which correspond to the input image and ground truth respectively.  

Metrics
*******

During the inference phase the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is ``True``. In the case of semantic segmentation the **Intersection over Union** (IoU) metrics is calculated after the network prediction. This metric, also referred as the Jaccard index, is essentially a method to quantify the percent of overlap between the target mask and the prediction output. Depending on the configuration different values are calculated (as explained in :ref:`config_test` and :ref:`config_metric`). This values can vary a lot as stated in :cite:p:`Franco-Barranco2021`.

* **Per patch**: IoU is calculated for each patch separately and then averaged. 
* **Reconstructed image**: IoU is calculated for each reconstructed image separately and then averaged. Notice that depending on the amount of overlap/padding selected the merged image can be different than just concatenating each patch. 
* **Full image**: IoU is calculated for each image separately and then averaged. The results may be slightly different from the reconstructed image.

Post-processing
***************

Only applied to ``3D`` images (e.g. ``PROBLEM.NDIM`` is ``2D`` or ``TEST.ANALIZE_2D_IMGS_AS_3D_STACK`` is ``True``). There are the following options:

* **Z-filtering**: to apply a median filtering in ``z`` axis. Useful to maintain class coherence across ``3D`` volumes. Enable it with ``TEST.POST_PROCESSING.Z_FILTERING`` and use ``TEST.POST_PROCESSING.Z_FILTERING_SIZE`` for the size of the median filter. 

* **YZ-filtering**: to apply a median filtering in ``y`` and ``z`` axes. Useful to maintain class coherence across ``3D`` volumes that can work slightly better than ``Z-filtering``. Enable it with ``TEST.POST_PROCESSING.YZ_FILTERING`` and use ``TEST.POST_PROCESSING.YZ_FILTERING_SIZE`` for the size of the median filter.  

.. _semantic_segmentation_data_run:

Run
~~~

.. tabs::

   .. tab:: GUI

        Select semantic segmentation workflow during the creation of a new configuration file:

        .. image:: https://raw.githubusercontent.com/BiaPyX/BiaPy-doc/master/source/img/gui/biapy_gui_semantic_seg.jpg
            :align: center 

   .. tab:: Google Colab

        Two different options depending on the image dimension: 

        .. |sem_seg_2D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/semantic_segmentation/BiaPy_2D_Semantic_Segmentation.ipynb

        * 2D: |sem_seg_2D_colablink|

        .. |sem_seg_3D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/semantic_segmentation/BiaPy_3D_Semantic_Segmentation.ipynb

        * 3D: |sem_seg_3D_colablink|

   .. tab:: Docker
            
        `Open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. For instance, using `2d_semantic_segmentation.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/semantic_segmentation/2d_semantic_segmentation.yaml>`__ template file, the code can be run as follows:

        .. code-block:: bash                                                                                                    

            # Configuration file
            job_cfg_file=/home/user/2d_semantic_segmentation.yaml
            # Path to the data directory
            data_dir=/home/user/data
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results
            # Just a name for the job
            job_name=my_2d_semantic_segmentation
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
                    -gpu $gpu_number

        .. note:: 

            Note that ``data_dir`` must contain all the paths ``DATA.*.PATH`` and ``DATA.*.GT_PATH`` so the container can find them. For instance, if you want to only train in this example ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` could be ``/home/user/data/train/x`` and ``/home/user/data/train/y`` respectively. 

   .. tab:: Command line

        `Open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. For instance, using `2d_semantic_segmentation.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/semantic_segmentation/2d_semantic_segmentation.yaml>`__ template file, the code can be run as follows:

        .. code-block:: bash
            
            # Configuration file
            job_cfg_file=/home/user/2d_semantic_segmentation.yaml       
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results  
            # Just a name for the job
            job_name=my_2d_semantic_segmentation      
            # Number that should be increased when one need to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0                   

            # Move where BiaPy installation resides
            cd BiaPy

            # Load the environment
            conda activate BiaPy_env
            source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
            
            python -u main.py \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu $gpu_number  

        For multi-GPU training you can call BiaPy as follows:

        .. code-block:: bash

            gpu_number="0, 1, 2"
            python -u -m torch.distributed.run \
                --nproc_per_node=3 \
                main.py \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu $gpu_number  

        ``nproc_per_node`` need to be equal to the number of GPUs you are using (e.g. ``gpu_number`` length).

      

.. _semantic_segmentation_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. figure:: ../img/unet2d_prediction.gif
   :align: center                  

   Example of semantic segmentation model predictions. From left to right: input image, its mask and the overlap between the mask and the model's output binarized. 


Following the example, you should see that the directory ``/home/user/exp_results/my_2d_semantic_segmentation`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: 

.. collapse:: Expand directory tree 

    .. code-block:: bash
        
      my_2d_semantic_segmentation/
      ├── config_files/
      │   └── my_2d_semantic_segmentation_1.yaml                                                                                                           
      ├── checkpoints
      │   └── my_2d_semantic_segmentation_1-checkpoint-best.pth
      └── results
         ├── my_2d_semantic_segmentation_1
          ├── . . .
          └── my_2d_semantic_segmentation_5
              ├── aug
              │   └── .tif files
             ├── charts
              │   ├── my_2d_semantic_segmentation_1_*.png
              │   ├── my_2d_semantic_segmentation_1_loss.png
              │   └── model_plot_my_2d_semantic_segmentation_1.png
             ├── full_image
              │   └── .tif files
             ├── full_image_binarized
              │   └── .tif files
             ├── full_post_processing
              │   └── .tif files
             ├── per_image
              │   └── .tif files
             ├── per_image_binarized
              │   └── .tif files
              ├── tensorboard
              └── train_logs

\

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

  * ``my_2d_semantic_segmentation.yaml``: YAML configuration file used (it will be overwrited every time the code is run)

* ``checkpoints``: directory where model's weights are stored.

  * ``my_2d_semantic_segmentation_1-checkpoint-best.pth``: checkpoint file (best in validation) where the model's weights are stored among other information.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

  * ``my_2d_semantic_segmentation_1``: run 1 experiment folder. 

    * ``aug``: image augmentation samples.

    * ``charts``:  

      * ``my_2d_semantic_segmentation_1_*.png``: Plot of each metric used during training.

      * ``my_2d_semantic_segmentation_1_loss.png``: Loss over epochs plot (when training is done). 

      * ``model_plot_my_2d_semantic_segmentation_1.png``: plot of the model.
        
    * ``full_image``: 

      * ``.tif files``: output of the model when feeding entire images (without patching). 

    * ``full_image_binarized``: 

      * ``.tif files``: Same as ``full_image`` but with the image binarized.

    * ``full_post_processing`` (optional if any post-processing was selected):

      * ``.tif files``: output of the model when feeding entire images (without patching) and applying post-processing, which in this case only `y` and `z` axes filtering was selected.

    * ``per_image``:

      * ``.tif files``: reconstructed images from patches.   

    * ``per_image_binarized``: 

      * ``.tif files``: Same as ``per_image`` but with the images binarized.
    
    * ``tensorboard``: Tensorboard logs.

    * ``train_logs``: each row represents a summary of each epoch stats. Only avaialable if training was done.
        
.. note:: 
   Here, for visualization purposes, only ``my_2d_semantic_segmentation_1`` has been described but ``my_2d_semantic_segmentation_2``, ``my_2d_semantic_segmentation_3``, ``my_2d_semantic_segmentation_4`` and ``my_2d_semantic_segmentation_5`` will follow the same structure.

