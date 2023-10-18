.. _self-supervision:

Self-supervision
----------------

The idea of this workflow is to pretrain the backbone model by solving a so-called pretext task without labels. This way, the model learns a representation that can be later transferred to solve a downstream task in a labeled (but smaller) dataset. In BiaPy we adopt two pretext tasks that you will need to choose with **pretext_task** variable below (controlled with ``PROBLEM.SELF_SUPERVISED.PRETEXT_TASK``). The pretext tasks implemented in BiaPy are these:

* ``crappify``: To recover a worstened version of the input image as in (:cite:p:`franco2022deep`).
* ``masking``: Random patches of the input image are masked and the network needs to reconstruct the missing pixels (:cite:p:`he2022masked`).

* **Input:** 
    * Image. 
* **Output:**
    * Pretrained model. 

In the figure below an example of ``crappify`` pretext task input images are depicted:

.. list-table::

  * - .. figure:: ../img/lucchi_train_0_crap.png
         :align: center

         Input image's worstened version.  

    - .. figure:: ../img/lucchi_train_0.png
         :align: center

         Input image. 

.. _self-supervision_data_prep:

Data preparation
~~~~~~~~~~~~~~~~

To ensure the proper operation of the library the data directory tree should be something like this: ::

    dataset/
    ├── train
    │   └── x
    │       ├── training-0001.tif
    │       ├── training-0002.tif
    │       ├── . . .
    │       └── training-9999.tif
    └── test
        └── x
            ├── testing-0001.tif
            ├── testing-0002.tif
            ├── . . .
            └── testing-9999.tif

.. _self-supervision_problem_resolution:

Problem resolution
~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Crappify
        Firstly, a **pre-processing** step is done where the input images are worstened by adding Gaussian noise and downsampling and upsampling them so the resolution gets worsen. This way, the images are stored in ``DATA.TRAIN.SSL_SOURCE_DIR``, ``DATA.VAL.SSL_SOURCE_DIR`` and ``DATA.TEST.SSL_SOURCE_DIR`` for train, validation and test data respectively. This way, the model will be input with the worstened version of images and will be trained to map it to its good version.  

   .. tab:: Masking

        The model undergoes training by acquiring the skill to restore a concealed input image. This occurs in real-time during training, where random portions of the images are automatically obscured.

After this training, the model should have learned some features of the images, which will be a good starting point in another training process. This way, if you re-train the model loading those learned model's weigths, which can be done enabling ``MODEL.LOAD_CHECKPOINT`` if you call BiaPy with the same ``--name`` option or setting ``PATHS.CHECKPOINT_FILE`` variable to point the file directly otherwise, the training process will be easier and faster than training from scratch. 

Configuration file
~~~~~~~~~~~~~~~~~~

Find in `templates/self-supervised <https://github.com/danifranco/BiaPy/tree/master/templates/self-supervised>`__ folder of BiaPy a few YAML configuration templates for this workflow. 


Special workflow configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here some special configuration options that can be selected in this workflow are described:

* **Metrics**: during the inference phase the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is enabled. In the case of super-resolution the **Peak signal-to-noise ratio** (`PSNR <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__) metrics is calculated when the worstened image is reconstructed from individual patches.

Run
~~~

.. tabs::

   .. tab:: Command line 

        Open a terminal as described in :ref:`installation`. For instance, using `2d_self-supervised.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/self-supervised/2d_self-supervised.yaml>`__ template file, the code can be run as follows:

        .. code-block:: bash
            
            # Configuration file
            job_cfg_file=/home/user/2d_self-supervised.yaml       
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results  
            # Just a name for the job
            job_name=my_2d_self-supervised     
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

   .. tab:: Docker 

        Open a terminal as described in :ref:`installation`. For instance, using `2d_self-supervised.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/self-supervision/2d_self-supervised.yaml>`__ template file, the code can be run as follows:

        .. code-block:: bash                                                                                                    

            # Configuration file
            job_cfg_file=/home/user/2d_self-supervised.yaml
            # Path to the data directory
            data_dir=/home/user/data
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results
            # Just a name for the job
            job_name=my_2d_self-supervised
            # Number that should be increased when one need to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0

            docker run --rm \
                --gpus "device=$gpu_number" \
                --mount type=bind,source=$job_cfg_file,target=$job_cfg_file \
                --mount type=bind,source=$result_dir,target=$result_dir \
                --mount type=bind,source=$data_dir,target=$data_dir \
                danifranco/biapy \
                    -cfg $job_cfg_file \
                    -rdir $result_dir \
                    -name $job_name \
                    -rid $job_counter \
                    -gpu $gpu_number

        .. note:: 
            Note that ``data_dir`` must contain the path ``DATA.*.PATH`` so the container can find it. For instance, if you want to only train in this example ``DATA.TRAIN.PATH`` could be ``/home/user/data/train/x``. 

   .. tab:: Google Colab 

        Two different options depending on the image dimension:

        .. |class_2D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/danifranco/BiaPy/blob/master/notebooks/self-supervised/BiaPy_2D_Self_Supervision.ipynb

        * 2D: |class_2D_colablink|

        .. |class_3D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/danifranco/BiaPy/blob/master/notebooks/self-supervised/BiaPy_3D_Self_Supervision.ipynb

        * 3D: |class_3D_colablink|

.. _self-supervision_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. list-table:: 

  * - .. figure:: ../img/pred_ssl.png
         :align: center

         Predicted image.

    - .. figure:: ../img/lucchi_train_0.png
         :align: center

         Original input image.


Following the example, you should see that the directory ``/home/user/exp_results/my_2d_self-supervised`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: ::

    my_2d_self-supervised/
    ├── config_files/
    │   └── my_2d_self-supervised.yaml                                                                                                           
    ├── checkpoints
    │   └── my_2d_self-supervised_1-checkpoint-best.pth
    └── results
        ├── my_2d_self-supervised_1
        ├── . . .
        └── my_2d_self-supervised_5
            ├── aug
            │   └── .tif files
            ├── charts
            │   ├── my_2d_self-supervised_1_*.png
            │   ├── my_2d_self-supervised_1_loss.png
            │   └── model_plot_my_2d_self-supervised_1.png
            ├── MAE_checks
            │   └── .tif files            
            ├── per_image
            │   └── .tif files
            ├── tensorboard
            └── train_logs

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

    * ``my_2d_self-supervised.yaml``: YAML configuration file used (it will be overwrited every time the code is run).

* ``checkpoints``: directory where model's weights are stored.

    * ``my_2d_self-supervised_1-checkpoint-best.pth``: checkpoint file (best in validation) where the model's weights are stored among other information.

    * ``normalization_mean_value.npy``: normalization mean value (only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``). Is saved to not calculate it everytime and to use it in inference.  
    
    * ``normalization_std_value.npy``: normalization std value (only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``). Is saved to not calculate it everytime and to use it in inference. 
    
* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

    * ``my_2d_self-supervised_1``: run 1 experiment folder. 

        * ``aug``: image augmentation samples.

        * ``charts``:  

             * ``my_2d_self-supervised_1_*.png``: Plot of each metric used during training.

             * ``my_2d_self-supervised_1_loss.png``: Loss over epochs plot (when training is done). 

             * ``model_plot_my_2d_self-supervised_1.png``: plot of the model.

        * ``MAE_checks``: MAE predictions. Only available if ``PROBLEM.SELF_SUPERVISED.PRETEXT_TASK`` is ``masking``.
        
            * ``*_original.tif``: Original image. 

            * ``*_masked.tif``: Masked image inputed to the model. 

            * ``*_reconstruction.tif``: Reconstructed image. 

            * ``*_reconstruction_and_visible.tif``: Reconstructed image with the visible parts copied. 

        * ``per_image``:

            * ``.tif files``: reconstructed images from patches.  

        * ``tensorboard``: Tensorboard logs.

        * ``train_logs``: each row represents a summary of each epoch stats. Only avaialable if training was done.
        
.. note:: 

  Here, for visualization purposes, only ``my_2d_self-supervised_1`` has been described but ``my_2d_self-supervised_2``, ``my_2d_self-supervised_3``, ``my_2d_self-supervised_4`` and ``my_2d_self-supervised_5`` will follow the same structure.



