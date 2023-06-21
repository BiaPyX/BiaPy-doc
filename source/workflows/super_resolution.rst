.. _super-resolution:

Super-resolution
----------------

The goal of this workflow aims at reconstructing high-resolution (HR) images from low-resolution (LR) ones. 

* **Input:** 
    * LR image. 
* **Output:**
    * HR image. 

In the figure below an example of this workflow's **input** is depicted to make a x2 upsampling. The images were obtained from `ZeroCostDL4Mic <https://github.com/HenriquesLab/ZeroCostDL4Mic>`__ project:

.. list-table:: 

  * - .. figure:: ../img/LR_sr.png
         :align: center

         Input LR image.

    - .. figure:: ../img/HR_sr.png
         :align: center

         Input HR image.

Notice that the LR image has been resized but actually that is ``502x502`` whereas the HR is ``1004x1004``. 

.. _super_resolution_data_prep:

Data preparation
~~~~~~~~~~~~~~~~

To ensure the proper operation of the library the data directory tree should be something like this: ::

    dataset/
    ├── train
    │   ├── LR
    │   │   ├── training-0001.tif
    │   │   ├── training-0002.tif
    │   │   ├── . . .
    │   │   ├── training-9999.tif
    │   └── HR
    │       ├── training_0001.tif
    │       ├── training_0002.tif
    │       ├── . . .
    │       ├── training_9999.tif
    └── test
        ├── LR
        │   ├── testing-0001.tif
        │   ├── testing-0002.tif
        │   ├── . . .
        │   ├── testing-9999.tif
        └── HR
            ├── testing_0001.tif
            ├── testing_0002.tif
            ├── . . .
            ├── testing_9999.tif

.. warning:: Ensure that images and their corresponding masks are sorted in the same way. A common approach is to fill with zeros the image number added to the filenames (as in the example). 

Configuration                                                                                                                 
~~~~~~~~~~~~~

Find in `templates/super-resolution <https://github.com/danifranco/BiaPy/tree/master/templates/super-resolution>`__ folder of BiaPy a few YAML configuration templates for this workflow. 


Special workflow configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here some special configuration options that can be selected in this workflow are described:

* **Upsampling** is the most important variable to be set via ``PROBLEM.SUPER_RESOLUTION.UPSCALING``. In the example above, its value is ``2``. 

* **Metrics**: during the inference phase the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is enabled. In the case of super-resolution the **Peak signal-to-noise ratio** (`PSNR <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__) metrics is calculated when the HR image is reconstructed from individual patches.

.. _super_resolution_data_run:

Run
~~~

**Command line**: Open a terminal as described in :ref:`installation`. For instance, using `edsr_super-resolution.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/super_resolution/edsr_super-resolution.yaml>`__ template file, the code can be run as follows:

.. code-block:: bash
    
    # Configuration file
    job_cfg_file=/home/user/edsr_super-resolution.yaml       
    # Where the experiment output directory should be created
    result_dir=/home/user/exp_results  
    # Just a name for the job
    job_name=edsr_2d      
    # Number that should be increased when one need to run the same job multiple times (reproducibility)
    job_counter=1
    # Number of the GPU to run the job in (according to 'nvidia-smi' command)
    gpu_number=0                   

    # Move where BiaPy installation resides
    cd BiaPy

    # Load the environment
    conda activate BiaPy_env
    
    python -u main.py \
           --config $job_cfg_file \
           --result_dir $result_dir  \ 
           --name $job_name    \
           --run_id $job_counter  \
           --gpu $gpu_number  


**Docker**: Open a terminal as described in :ref:`installation`. For instance, using `edsr_super-resolution.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/super_resolution/edsr_super-resolution.yaml>`__ template file, the code can be run as follows:

.. code-block:: bash                                                                                                    

    # Configuration file
    job_cfg_file=/home/user/edsr_super-resolution.yaml
    # Path to the data directory
    data_dir=/home/user/data
    # Where the experiment output directory should be created
    result_dir=/home/user/exp_results
    # Just a name for the job
    job_name=edsr_2d
    # Number that should be increased when one need to run the same job multiple times (reproducibility)
    job_counter=1
    # Number of the GPU to run the job in (according to 'nvidia-smi' command)
    gpu_number=0

    sudo docker run --rm \
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
    Note that ``data_dir`` must contain all the paths ``DATA.*.PATH`` and ``DATA.*.GT_PATH`` so the container can find them. For instance, if you want to only train in this example ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` could be ``/home/user/data/train/x`` and ``/home/user/data/train/y`` respectively. 

.. _super_resolution_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. list-table:: 

  * - .. figure:: ../img/pred_sr.png
         :align: center

         Predicted HR image.

    - .. figure:: ../img/HR_sr.png
         :align: center

         Input HR image.

Here both images are of size ``1004x1004``. 


Following the example, you should see that the directory ``/home/user/exp_results/edsr_2d`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: ::

    edsr_2d/
    ├── config_files/
    │   └── edsr_super-resolution.yaml                                                                                                           
    ├── checkpoints
    │   └── model_weights_edsr_2d_1.h5
    └── results
        ├── edsr_2d_1
        ├── . . .
        └── edsr_2d_5
            ├── aug
            │   └── .tif files
            ├── charts
            │   ├── edsr_2d_1_PSNR.png
            │   ├── edsr_2d_1_loss.png
            │   └── model_plot_edsr_2d_1.png
            └── per_image
                └── .tif files

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

    * ``edsr_super-resolution.yaml``: YAML configuration file used (it will be overwrited every time the code is run)

* ``checkpoints``: directory where model's weights are stored.

    * ``model_weights_edsr_2d_1.h5``: model's weights file.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

    * ``edsr_2d_1``: run 1 experiment folder. 

        * ``aug``: image augmentation samples.

        * ``charts``:  

             * ``edsr_2d_1_PSNR.png``: PNSR over epochs plot (when training is done).

             * ``edsr_2d_1_loss.png``: Loss over epochs plot (when training is done). 

             * ``model_plot_edsr_2d_1.png``: plot of the model.

        * ``per_image``:

            * ``.tif files``: reconstructed images from patches.   

.. note:: 
   Here, for visualization purposes, only ``edsr_2d_1`` has been described but ``edsr_2d_2``, ``edsr_2d_3``, ``edsr_2d_4``
   and ``edsr_2d_5`` will follow the same structure.


