.. _self-supervision:

Self-supervision
----------------

Description of the task
~~~~~~~~~~~~~~~~~~~~~~~

The idea of this workflow is to **pretrain** a deep learning model by solving a so-called **pretext task** (denoising, inpainting, etc.) without labels. This way, the model learns a representation that can be later transferred to solve a **downstream task** (segmentation, detection, etc.) in a labeled (and usually smaller) dataset. 

An example of a pretext task is depicted below, with an original image and its corresponding degradated version. The model will be then train using the *clean* original image as target and the degrated image as input:

.. role:: raw-html(raw)
    :format: html

.. list-table::
  :align: center
  :width: 680px

  * - .. figure:: ../img/lucchi_train_0.png
         :align: center
         :width: 300px
         :alt: Electron microscopy image to be used as target for pretraining

         Original electron microscopy image.

    - .. figure:: ../img/lucchi_train_0_crap.png
         :align: center
         :width: 300px
         :alt: Corresponding degrated version of the same image

         Corresponding degradated image.


Inputs and outputs
~~~~~~~~~~~~~~~~~~
The self-supervision workflows in BiaPy expect a **folder** as input:

* **Training Raw Images**: A folder that contains the unprocessed (single-channel or multi-channel) images that will be used to (pre-)train the model.
  
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Self-supervised learning*, twice *Continue*, under *General options* > *Train data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/GUI-general-options.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D self-supervision notebook, go to *Paths for Input Images and Output Files*, edit the field **train_data_path**:
        
        .. image:: ../img/self-supervised/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 95%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TRAIN.PATH`` with the absolute path to the folder with your training raw images.


Upon successful execution, a directory will be generated with the results of the pre-training. Therefore, you will need to define:

* **Output Folder**: A designated path to save the self-supervision outcomes.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, click on the *Browse* button of **Output folder to save the results**:

        .. image:: ../img/self-supervised/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D self-supervised learning notebook, go to *Paths for Input Images and Output Files*, edit the field **output_path**:
        
        .. image:: ../img/self-supervised/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 95%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--result_dir`` flag. See the *Command line* configuration of :ref:`self-supervision_data_run` for a full example.


.. list-table::
  :align: center

  * - .. figure:: ../img/self-supervised/Inputs-outputs.svg
         :align: center
         :width: 300
         :alt: Graphical description of minimal inputs and outputs in BiaPy for self-supervised learning.
        
         **BiaPy input and output folders for self-supervised learning.** Since this workflow :raw-html:`<br />` is self-supervised, no labels are needed in neither train nor test.
  

.. _self-supervision_data_prep:

Data structure
~~~~~~~~~~~~~~

To ensure the proper operation of the workflow, the data directory tree should be something like this: 

.. code-block::
    
  dataset/
  ├── train
  │   └── x
  │       ├── training-0001.tif
  │       ├── training-0002.tif
  │       ├── . . .
  │       └── training-9999.tif
  └── test
      └── x
          ├── testing-0001.tif
          ├── testing-0002.tif
          ├── . . .
          └── testing-9999.tif

\

In this example, the (pre-)training images are under ``dataset/train/x/``, while the test images are under ``dataset/test/x/``. **This is just an example**, you can name your folders as you wish as long as you set the paths correctly later.

Minimal configuration
~~~~~~~~~~~~~~~~~~~~~
Apart from the input and output folders, there are a few basic parameters that always need to be specified in order to run a self-supervised learning workflow in BiaPy. **These parameters can be introduced either directly in the GUI, the code-free notebooks or by editing the YAML configuration file**.

Experiment name
***************
Also known as "model name" or "job name", this will be the name of the current experiment you want to run, so it can be differenciated from other past and future experiments.

.. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, type the name you want for the job in the **Job name** field:

        .. image:: ../img/self-supervised/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D self-supervised learning notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **model_name**:
        
        .. image:: ../img/self-supervised/Notebooks-model-name-data-conf.png
          :align: center
          :width: 65%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--name`` flag. See the *Command line* configuration of :ref:`self-supervision_data_run` for a full example.


\

.. note:: Use only *my_model* -style, not *my-model* (Use "_" not "-"). Do not use spaces in the name. Avoid using the name of an existing experiment/model/job (saved in the same result folder) as it will be overwritten.

Data management
***************
Validation Set
""""""""""""""
With the goal to monitor the training process, it is common to use a third dataset called the "Validation Set". This is a subset of the whole dataset that is used to evaluate the model's performance and optimize training parameters. This subset will not be directly used for training the model, and thus, when applying the model to these images, we can see if the model is learning the training set's patterns too specifically or if it is generalizing properly.

.. list-table::
  :align: center

  * - .. figure:: ../img/self-supervised/data-partitions.png
         :align: center
         :width: 400
         :alt: Graphical description of data partitions in BiaPy for SSL
        
         **Graphical description of data partitions in BiaPy when using self-generated labels.**



To define such set, there are two options:
  
* **Validation proportion/percentage**: Select a proportion (or percentage) of your training dataset to be used to validate the network during the training. Usual values are 0.1 (10%) or 0.2 (20%), and the samples of that set will be selected at random.
  
  .. collapse:: Expand to see how to configure

      .. tabs::
        .. tab:: GUI

          Under *Workflow*, select *Self-supervised learning*, click twice on *Continue*, and under *General options* > *Advanced options* > *Validation data*, select "Extract from train (split training)" in **Validation type**, and introduce your value (between 0 and 1) in the **Train proportion for validation**:

          .. image:: ../img/GUI-validation-percentage.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          In either the 2D or the 3D denoising notebook, go to *Configure and train the DNN model* > *Select your parameters*, and under *Data management*, edit the field **percentage_validation** with a value between 0 and 100:
          
          .. image:: ../img/self-supervised/Notebooks-model-name-data-conf.png
            :align: center
            :width: 75%

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.SPLIT_TRAIN`` with a value between 0 and 1, representing the proportion of the training set that will be set apart for validation.


* **Validation path**: Similar to the training and test sets, you can select a folder that contains the unprocessed (single-channel or multi-channel) raw images that will be used to validate the current model during training.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image denoising*, click twice on *Continue*, and under *General options* > *Advanced options* > *Validation data*, select "Not extracted from train (path needed)" in **Validation type**, click on the *Browse* button of **Input raw image folder** and select the folder containing your validation raw images:

        .. image:: ../img/self-supervised/GUI-validation-paths.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        This option is currently not available in the notebooks.

      .. tab:: YAML configuration file
      
        Edit the variable ``DATA.VAL.PATH`` with the absolute path to your validation raw images.

 

Basic training parameters
*************************
At the core of each BiaPy workflow there is a deep learning model. Although we try to simplify the number of parameters to tune, these are the basic parameters that need to be defined for training a self-supervised learning workflow:

* **Pretext task**: The task to use to pretrain the model. Options: 'crappify' to recover a worstened version of the input image (as in :cite:p:`franco2022deep`), and 'masking', where random patches of the input image are masked and the network needs to reconstruct the missing pixels (as in :cite:p:`he2022masked`). Default value: 'masking'.

  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Self-supervised learning*, click twice on *Continue*, and under *Workflow specific options* > *Pretext task options*, edit the **Type of task** field by selecting "masking" or "crappify":

            .. image:: ../img/self-supervised/GUI-workflow-specific-options.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D self-supervised learning notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **pretext_task**:
            
            .. image:: ../img/self-supervised/Notebooks-basic-training-params.png
              :align: center

          .. tab:: YAML configuration file
          
            Edit the value of the variable ``DATA.SELF_SUPERVISED.PRETEXT_TASK`` with either ``"crappify"``or ``"masking"``.

* **Number of input channels**: The number of channels of your raw images (grayscale = 1, RGB = 3). Notice the dimensionality of your images (2D/3D) is set by default depending on the workflow template you select.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Self-supervised learning*, click twice on *Continue*, and under *General options* > *Train data*, edit the last value of the field **Data patch size** with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D:

            .. image:: ../img/GUI-general-options.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D self-supervised learning notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **input_channels**:
            
            .. image:: ../img/self-supervised/Notebooks-basic-training-params.png
              :align: center

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``DATA.PATCH_SIZE`` with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D.

* **Number of epochs**: This number indicates how many `rounds <https://machine-learning.paperspace.com/wiki/epoch>`_ the network will be trained. On each round, the network usually sees the full training set. The value of this parameter depends on the size and complexity of each dataset. You can start with something like 100 epochs and tune it depending on how fast the loss (error) is reduced.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Self-supervised learning*, click twice on *Continue*, and under *General options*, click on *Advanced options*, scroll down to *General training parameters*, and edit the field **Number of epochs**:

            .. image:: ../img/self-supervised/GUI-basic-training-params.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D self-supervised learning notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **number_of_epochs**:
            
            .. image:: ../img/self-supervised/Notebooks-basic-training-params.png
              :align: center

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.EPOCHS`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.

* **Patience**: This is a number that indicates how many epochs you want to wait without the model improving its results in the validation set to stop training. Again, this value depends on the data you're working on, but you can start with something like 20.
   
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Self-supervised learning*, click twice on *Continue*, and under *General options*, click on *Advanced options*, scroll down to *General training parameters*, and edit the field **Patience**:

            .. image:: ../img/self-supervised/GUI-basic-training-params.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D self-supervised notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **patience**:
            
            .. image:: ../img/self-supervised/Notebooks-basic-training-params.png
              :align: center

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.PATIENCE`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.


For improving performance, other advanced parameters can be optimized, for example, the model's architecture. The architecture assigned as default is usually the MAE, as it is a standard in self-supervision tasks. This architecture allows a strong baseline, but further exploration could potentially lead to better results.

.. note:: Once the parameters are correctly assigned, the training phase can be executed. Note that to train large models effectively the use of a GPU (Graphics Processing Unit) is essential. This hardware accelerator performs parallel computations and has larger RAM memory compared to the CPUs, which enables faster training times.

.. _self-supervision_data_run:

How to run
~~~~~~~~~~
BiaPy offers different options to run workflows depending on your degree of computer expertise. Select whichever is more approppriate for you:

.. tabs::
   .. tab:: GUI

        In the BiaPy GUI, navigate to *Workflow*, then select *Self-supervised learning* and follow the on-screen instructions:

        .. image:: ../img/gui/biapy_gui_ssl.png
            :align: center

        \
        
        .. tip:: If you need additional help, watch BiaPy's `GUI walk-through video <https://www.youtube.com/embed/vY7aBh5FUNk?si=yvVolBnu5APNeHwB>`__. 
   
   .. tab:: Google Colab 

        BiaPy offers two code-free notebooks in Google Colab to perform self-supervised learning:

        .. |class_2D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/self-supervised/BiaPy_2D_Self_Supervision.ipynb

        * For 2D images: |class_2D_colablink|

        .. |class_3D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/self-supervised/BiaPy_3D_Self_Supervision.ipynb

        * For 3D images: |class_3D_colablink|

   .. tab:: Docker 

        If you installed BiaPy via Docker, `open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. For instance, you can use the `2d_self-supervised.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/self-supervision/2d_self-supervised.yaml>`__ template file (or your own YAML file), and then run the workflow as follows:

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
                BiaPyX/biapy \
                    -cfg $job_cfg_file \
                    -rdir $result_dir \
                    -name $job_name \
                    -rid $job_counter \
                    -gpu "$gpu_number"

        .. note:: 
            Note that ``data_dir`` must contain the path ``DATA.*.PATH`` so the container can find it. For instance, if you want to only train in this example ``DATA.TRAIN.PATH`` could be ``/home/user/data/train/x``. 

   .. tab:: Command line 

        `From a terminal <../get_started/faq.html#opening-a-terminal>`__, you can use `2d_self-supervised.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/self-supervised/2d_self-supervised.yaml>`__ template file (or your own YAML file)to run the workflow as follows:

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

            # Load the environment
            conda activate BiaPy_env
            
            biapy \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu "$gpu_number"  


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
                --gpu "$gpu_number"  

        ``nproc_per_node`` needs to be equal to the number of GPUs you are using (e.g. ``gpu_number`` length).



.. _self-supervision_problem_resolution:

Templates                                                                                                                 
~~~~~~~~~

In the `templates/self-supervised <https://github.com/BiaPyX/BiaPy/tree/master/templates/self-supervised>`__ folder of BiaPy, you can find a few YAML configuration templates for this workflow. 

[Advanced] Special workflow configuration 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: This section is recommended for experienced users only to improve the performance of their workflows. When in doubt, do not hesitate to check our `FAQ & Troubleshooting <../get_started/faq.html>`__ or open a question in the `image.sc discussion forum <our FAQ & Troubleshooting section>`_.

Advanced Parameters 
*******************
Many of the parameters of our workflows are set by default to values that work commonly well. However, it may be needed to tune them to improve the results of the workflow. For instance, you may modify the following parameters:

* **Model architecture**:  Select the architecture of the DNN used as backbone of the pipeline. Options: MAE, EDSR, RCAN, WDSR, DFCAN, U-Net, Residual U-Net, Attention U-Net, SEUNet, MultiResUNet, ResUNet++, UNETR-Mini, UNETR-Small and UNETR-Base. Common option: MAE.
* **Batch size**: This parameter defines the number of patches seen in each training step. Reducing or increasing the batch size may slow or speed up your training, respectively, and can influence network performance. Common values are 4, 8, 16, etc.
* **Patch size**: Input the size of the patches use to train your model (length in pixels in X and Y). The value should be smaller or equal to the dimensions of the image. The default value is 64 in 2D, i.e. 64x64 pixels.
* **Optimizer**: Select the optimizer used to train your model. Options: ADAM, ADAMW, Stochastic Gradient Descent (SGD). ADAM usually converges faster, while ADAMW provides a balance between fast convergence and better handling of weight decay regularization. SGD is known for better generalization. Default value: ADAMW.
* **Initial learning rate**: Input the initial value to be used as learning rate. If you select ADAM as optimizer, this value should be around 10e-4. 

Problem resolution
******************

In BiaPy we adopt two pretext tasks that you will need to choose with **pretext_task** variable below (controlled with ``PROBLEM.SELF_SUPERVISED.PRETEXT_TASK``):

* ``crappify``: Firstly, a **pre-processing** step is done where the input images are worstened by adding Gaussian noise and downsampling and upsampling them so the resolution gets worsen. This way, the images are stored in ``DATA.TRAIN.SSL_SOURCE_DIR``, ``DATA.VAL.SSL_SOURCE_DIR`` and ``DATA.TEST.SSL_SOURCE_DIR`` for train, validation and test data respectively. This way, the model will be input with the worstened version of images and will be trained to map it to its good version (as in :cite:p:`franco2022deep`).

* ``masking``: The model undergoes training by acquiring the skill to restore a concealed input image. This occurs in real-time during training, where random portions of the images are automatically obscured (:cite:p:`he2022masked`).

After this training, the model should have learned some features of the images, which will be a good starting point in another training process. This way, if you re-train the model loading those learned model's weigths, which can be done enabling ``MODEL.LOAD_CHECKPOINT`` if you call BiaPy with the same ``--name`` option or setting ``PATHS.CHECKPOINT_FILE`` variable to point the file directly otherwise, the training process will be easier and faster than training from scratch. 

Metrics
*******

During the inference phase the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is ``True``. In the case of super-resolution the **Peak signal-to-noise ratio** (`PSNR <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__) metrics is calculated when the worstened image is reconstructed from individual patches.


.. _self-supervision_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. list-table:: 
  :align: center
  :width: 680px

  * - .. figure:: ../img/pred_ssl.png
         :align: center
         :width: 300px

         Predicted image.

    - .. figure:: ../img/lucchi_train_0.png
         :align: center
         :width: 300px

         Original image.


Following the example, you should see that the directory ``/home/user/exp_results/my_2d_self-supervised`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: 

.. collapse:: Expand directory tree 

    .. code-block:: bash
        
      my_2d_self-supervised/
      ├── config_files
      │   └── my_2d_self-supervised.yaml                                                                                                           
      ├── checkpoints
      │   └── my_2d_self-supervised_1-checkpoint-best.pth
      └── results
          ├── my_2d_self-supervised_1
          ├── . . .
          └── my_2d_self-supervised_5
              ├── aug
              │   └── .tif files
              ├── charts
              │   ├── my_2d_self-supervised_1_*.png
              │   └── my_2d_self-supervised_1_loss.png
              ├── MAE_checks
              │   └── .tif files            
              ├── per_image
              │   ├── .tif files
              │   └── .zarr files (or.h5)
              ├── tensorboard
              └── train_logs

\

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

  * ``my_2d_self-supervised.yaml``: YAML configuration file used (it will be overwrited every time the code is run).

* ``checkpoints``, *optional*: directory where model's weights are stored. Only created when ``TRAIN.ENABLE`` is ``True`` and the model is trained for at least one epoch. Can contain:

  * ``my_2d_self-supervised_1-checkpoint-best.pth``, *optional*: checkpoint file (best in validation) where the model's weights are stored among other information. Only created when the model is trained for at least one epoch. 

  * ``normalization_mean_value.npy``, *optional*: normalization mean value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.
  
  * ``normalization_std_value.npy``, *optional*: normalization std value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.
  
* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

  * ``my_2d_self-supervised_1``: run 1 experiment folder. Can contain:

    * ``aug``, *optional*: image augmentation samples. Only created if ``AUGMENTOR.AUG_SAMPLES`` is ``True``.

    * ``charts``, *optional*: only created when ``TRAIN.ENABLE`` is ``True`` and epochs trained are more or equal ``LOG.CHART_CREATION_FREQ``. Can contain:

      * ``my_2d_self-supervised_1_*.png``: Plot of each metric used during training.

      * ``my_2d_self-supervised_1_loss.png``: Loss over epochs plot. 

    * ``MAE_checks``, *optional*: MAE predictions. Only created if ``PROBLEM.SELF_SUPERVISED.PRETEXT_TASK`` is ``masking``.
      
      * ``*_original.tif``: Original image. 

      * ``*_masked.tif``: Masked image inputed to the model. 

      * ``*_reconstruction.tif``: Reconstructed image. 

      * ``*_reconstruction_and_visible.tif``: Reconstructed image with the visible parts copied. 

    * ``per_image``:

      * ``.tif files``: reconstructed images from patches.  

      * ``.zarr files (or.h5)``, *optional*: reconstructed images from patches. Created when ``TEST.BY_CHUNKS.ENABLE`` is ``True``.

    * ``tensorboard``: Tensorboard logs.

    * ``train_logs``: each row represents a summary of each epoch stats. Only avaialable if training was done.
      
.. note:: 

  Here, for visualization purposes, only ``my_2d_self-supervised_1`` has been described but ``my_2d_self-supervised_2``, ``my_2d_self-supervised_3``, ``my_2d_self-supervised_4`` and ``my_2d_self-supervised_5`` will follow the same structure.



