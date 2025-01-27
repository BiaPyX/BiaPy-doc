.. _super-resolution:

Super-resolution
----------------

Description of the task
~~~~~~~~~~~~~~~~~~~~~~~

The goal of this workflow is to reconstruct high-resolution (HR) images from low-resolution (LR) ones. If there is a difference in the size of the LR and HR images, typically determined by a scale factor (x2, x4), this task is known as **single-image super-resolution**. If the size of the LR and HR images is the same, this task is usually referred to as **image restoration**.

An example of this task is displayed in the figure below, with a LR fluorescence microscopy image used as input (left) and its corresponding HR image (x2 scale factor).

.. role:: raw-html(raw)
    :format: html


.. list-table:: 
  :align: center
  :width: 680px

  * - .. figure:: ../img/LR_sr.png
         :align: center
         :width: 300px
         
         LR fluorescence image from the :raw-html:`<br />` `F-actin dataset by Qiao et al <https://figshare.com/articles/dataset/BioSR/13264793>`_.

    - .. figure:: ../img/HR_sr.png
         :align: center
         :width: 300px

         Corresponding HR image :raw-html:`<br />` at x2 resolution.

Notice that the LR image has been resized but its actual size is 502x502 pixels, whereas the size of its HR counterpart is 1004x1004. 

Inputs and outputs
~~~~~~~~~~~~~~~~~~
The super-resolution workflows in BiaPy expect a series of **folders** as input:

* **Training LR Images**: A folder that contains the LR (single-channel or multi-channel) images that will be used to train the model.
  
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Super-resolution*, twice *Continue*, under *General options* > *Train data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/super-resolution/GUI-general-options.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D super-resolution notebook, go to *Paths for Input Images and Output Files*, edit the field **train_lr_data_path**:
        
        .. image:: ../img/super-resolution/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TRAIN.PATH`` with the absolute path to the folder with your training raw images.



* **Training HR Images**: A folder that contains the HR (single- or multi-channel) images for training. Ensure their number match that of the training LR images.
  
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Super-resolution*, twice *Continue*, under *General options* > *Train data*, click on the *Browse* button of **Input high-resolution image folder**:

        .. image:: ../img/super-resolution/GUI-general-options.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D super-resolution notebook, go to *Paths for Input Images and Output Files*, edit the field **train_hr_data_path**:
        
        .. image:: ../img/super-resolution/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TRAIN.GT_PATH`` with the absolute path to the folder with your training HR images.

* .. raw:: html

      <b><span style="color: darkgreen;">[Optional]</span> Test LR Images</b>: A folder that contains the images to evaluate the model's performance.
 
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Super-resolution*, three times *Continue*, under *General options* > *Test data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/super-resolution/GUI-test-data.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D super-resolution notebook, go to *Paths for Input Images and Output Files*, edit the field **test_lr_data_path**:
        
        .. image:: ../img/super-resolution/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TEST.PATH`` with the absolute path to the folder with your test LR images.

* .. raw:: html

      <b><span style="color: darkgreen;">[Optional]</span> Test HR Images</b>: A folder that contains the HR images for testing. Again, ensure their count and sizes align with the test raw images.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Super-resolution*, three times *Continue*, under *General options* > *Test data*, select "Yes" in the *Do you have high-resolution test data?* field, and then click on the *Browse* button of **Input high-resolution image folder**:

        .. image:: ../img/super-resolution/GUI-test-data-gt.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D super-resolution notebook, go to *Paths for Input Images and Output Files*, edit the field **test_hr_data_path**:
        
        .. image:: ../img/super-resolution/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TEST.GT_PATH`` with the absolute path to the folder with your test HR images.

Upon successful execution, a directory will be generated with the segmentation results. Therefore, you will need to define:

* **Output Folder**: A designated path to save the segmentation outcomes.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, click on the *Browse* button of **Output folder to save the results**:

        .. image:: ../img/super-resolution/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D super-resolution/ notebook, go to *Paths for Input Images and Output Files*, edit the field **output_path**:
        
        .. image:: ../img/super-resolution/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 75%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--result_dir`` flag. See the *Command line* configuration of :ref:`super_resolution_data_run` for a full example.


.. list-table::
  :align: center

  * - .. figure:: ../img/super-resolution/Inputs-outputs.svg
         :align: center
         :width: 500
         :alt: Graphical description of minimal inputs and outputs in BiaPy for super-resolution.
        
         **BiaPy input and output folders for super-resolution.**
  


.. _super_resolution_data_prep:

Data structure
**************

To ensure the proper operation of the workflow, the directory tree should be something like this: 



.. code-block::
    
  dataset/
  ├── train
  │   ├── LR
  │   │   ├── training_0001.tif
  │   │   ├── training_0002.tif
  │   │   ├── . . .
  │   │   └── training_9999.tif
  │   └── HR
  │       ├── training_0001.tif
  │       ├── training_0002.tif
  │       ├── . . .
  │       └── training_9999.tif
  └── test
      ├── LR
      │   ├── testing_0001.tif
      │   ├── testing_0002.tif
      │   ├── . . .
      │   └── testing_9999.tif
      └── HR
          ├── testing_0001.tif
          ├── testing_0002.tif
          ├── . . .
          └── testing_9999.tif

\

In this example, the LR training images are under ``dataset/train/LR/`` and their corresponding HR images are under ``dataset/train/HR/``, while the LR test images are under ``dataset/test/LR/`` and their corresponding HR are under ``dataset/test/HR/``. **This is just an example**, you can name your folders as you wish as long as you set the paths correctly later.

.. note:: Ensure that the LR and HR images are sorted in the same way. A common approach is to give the same name to each LR image and its corresponding HR image, or to fill with zeros the image number added to the filenames (as in the example). 

Example datasets
****************
Below is a list of publicly available datasets that are ready to be used in BiaPy for single image super-resolution:

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



Minimal configuration
~~~~~~~~~~~~~~~~~~~~~
Apart from the input and output folders, there are a few basic parameters that always need to be specified in order to run an super-resolution workflow in BiaPy. **These parameters can be introduced either directly in the GUI, the code-free notebooks or by editing the YAML configuration file**.

Experiment name
***************
Also known as "model name" or "job name", this will be the name of the current experiment you want to run, so it can be differenciated from other past and future experiments.

.. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, type the name you want for the job in the **Job name** field:

        .. image:: ../img/super-resolution/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D instance segmentation notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **model_name**:
        
        .. image:: ../img/super-resolution/Notebooks-model-name-data-conf.png
          :align: center
          :width: 65%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--name`` flag. See the *Command line* configuration of :ref:`super_resolution_data_run` for a full example.


\

.. note:: Use only *my_model* -style, not *my-model* (Use "_" not "-"). Do not use spaces in the name. Avoid using the name of an existing experiment/model/job (saved in the same result folder) as it will be overwritten.

Data management
***************
Validation Set
""""""""""""""
With the goal to monitor the training process, it is common to use a third dataset called the "Validation Set". This is a subset of the whole dataset that is used to evaluate the model's performance and optimize training parameters. This subset will not be directly used for training the model, and thus, when applying the model to these images, we can see if the model is learning the training set's patterns too specifically or if it is generalizing properly.

.. list-table::
  :align: center

  * - .. figure:: ../img/data-partitions.png
         :align: center
         :width: 400
         :alt: Graphical description of data partitions in BiaPy
        
         **Graphical description of data partitions in BiaPy.**



To define such set, there are two options:
  
* **Validation proportion/percentage**: Select a proportion (or percentage) of your training dataset to be used to validate the network during the training. Usual values are 0.1 (10%) or 0.2 (20%), and the samples of that set will be selected at random.
  
  .. collapse:: Expand to see how to configure

      .. tabs::
        .. tab:: GUI

          Under *Workflow*, select *Super-resolution*, click twice on *Continue*, and under *General options* > *Advanced options* > *Validation data*, select "Extract from train (split training)" in **Validation type**, and introduce your value (between 0 and 1) in the **Train proportion for validation**:

          .. image:: ../img/GUI-validation-percentage.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          In either the 2D or the 3D super-resolution notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **percentage_validation** with a value between 0 and 100:
          
          .. image:: ../img/super-resolution/Notebooks-model-name-data-conf.png
            :align: center
            :width: 50%

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.SPLIT_TRAIN`` with a value between 0 and 1, representing the proportion of the training set that will be set apart for validation.

* **Validation paths**: Similar to the training and test sets, you can select two folders with the validation LR and HR images:

  * **Validation LR Images**: A folder that contains the unprocessed (single-channel or multi-channel) LR images that will be used to select the best model during training.
  
    .. collapse:: Expand to see how to configure

      .. tabs::
        .. tab:: GUI

          Under *Workflow*, select *Super-resolution*, click twice on *Continue*, and under *General options* > *Advanced options* > *Validation data*, select "Not extracted from train (path needed)" in **Validation type**, click on the *Browse* button of **Input raw image folder** and select the folder containing your validation raw (LR) images:

          .. image:: ../img/super-resolution/GUI-validation-paths.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          This option is currently not available in the notebooks.

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.PATH`` with the absolute path to your validation raw images.

  * **Validation HR Images**: A folder that contains the instance (single-channel or multi-channel) HR images for validation. Ensure the number and ordering match those of the validation LR images.
  
    .. collapse:: Expand to see how to configure

      .. tabs::
        .. tab:: GUI

          Under *Workflow*, select *Super-resolution*, click twice on *Continue*, and under *General options* > *Advanced options* > *Validation data*, select "Not extracted from train (path needed)" in **Validation type**, click on the *Browse* button of **Input high-resolution image folder** and select the folder containing your validation HR images:

          .. image:: ../img/super-resolution/GUI-validation-paths.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          This option is currently not available in the notebooks.

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.GT_PATH`` with the absolute path to your validation HR images.



Test ground-truth
"""""""""""""""""
Do you have HR images for the test set? This is a key question so BiaPy knows if your test set will be used for evaluation in new data (unseen during training) or simply produce predictions on that new data. All workflows contain a parameter to specify this aspect.

.. collapse:: Expand to see how to configure

  .. tabs::
    .. tab:: GUI

      Under *Workflow*, select *Super-resolution*, three times *Continue*, under *General options* > *Test data*, select "Yes" or "No" in the **Do you have high-resolution test data?** field:

      .. image:: ../img/super-resolution/GUI-test-data.png
        :align: center

    .. tab:: Google Colab / Notebooks
      
      In either the 2D or the 3D instance segmentation notebook, go to *Configure and train the DNN model* > *Select your parameters*, and check or uncheck the **test_ground_truth** option:
      
      .. image:: ../img/super-resolution/Notebooks-model-name-data-conf.png
        :align: center
        :width: 50%


    .. tab:: YAML configuration file
      
      Set the variable ``DATA.TEST.LOAD_GT`` to ``True`` if you have test HR images, and ``False`` if you do not.


\

Basic training parameters
*************************
At the core of each BiaPy workflow there is a deep learning model. Although we try to simplify the number of parameters to tune, these are the basic parameters that need to be defined for training a super-resolution workflow:

* **Number of input channels**: The number of channels of your raw images (grayscale = 1, RGB = 3). Notice the dimensionality of your images (2D/3D) is set by default depending on the workflow template you select.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Super-resolution*, click twice on *Continue*, and under *General options* > *Train data*, edit the last value of the field **Data patch size** with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D:

            .. image:: ../img/super-resolution/GUI-general-options.png
              :align: center
              :width: 75%

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D super-resolution notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **input_channels**:
            
            .. image:: ../img/super-resolution/Notebooks-basic-training-params-2D.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``DATA.PATCH_SIZE`` with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D.

* **Scale factors**: Factors by which the images will be super-resolved in X, Y and, if the images are 3D, in Z. If set to 1, the model will perform image restoration.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Super-resolution*, click twice on *Continue*, and under *Workflow specific options*, and edit the values of the field **Upscaling** with the scale factor for each dimension. This variable follows a ``(y, x)`` notation in 2D and a ``(z, y, x)`` notation in 3D:

            .. image:: ../img/super-resolution/GUI-workflow-specific-options.png
              :align: center
              :width: 50%

          .. tab:: Google Colab / Notebooks
            
            In the 2D super-resolution notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **scale_factor** with the scale factor for X and Y (same value for both dimensions):
            
            .. image:: ../img/super-resolution/Notebooks-basic-training-params-2D.png
              :align: center
              :width: 75%

            In the 3D super-resolution notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **scale_factor_XY** with the scale factor for X and Y (same value for both dimensions) and the field **scale_factor_Z** with the scale factor for Z:
            
            .. image:: ../img/super-resolution/Notebooks-basic-training-params-3D.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``PROBLEM.SUPER_RESOLUTION.UPSCALING`` with the scale factor for each dimension. This variable follows a ``(y, x)`` notation in 2D and a ``(z, y, x)`` notation in 3D.


* **Number of epochs**: This number indicates how many `rounds <https://machine-learning.paperspace.com/wiki/epoch>`_ the network will be trained. On each round, the network usually sees the full training set. The value of this parameter depends on the size and complexity of each dataset. You can start with something like 100 epochs and tune it depending on how fast the loss (error) is reduced.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Super-resolution*, click twice on *Continue*, and under *General options*, click on *Advanced options*, scroll down to *General training parameters*, and edit the field **Number of epochs**:

            .. image:: ../img/super-resolution/GUI-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D super-resolution notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **number_of_epochs**:
            
            .. image:: ../img/super-resolution/Notebooks-basic-training-params-3D.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.EPOCHS`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.

* **Patience**: This is a number that indicates how many epochs you want to wait without the model improving its results in the validation set to stop training. Again, this value depends on the data you're working on, but you can start with something like 20.
   
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Super-resolution*, click twice on *Continue*, and under *General options*, click on *Advanced options*, scroll down to *General training parameters*, and edit the field **Patience**:

            .. image:: ../img/super-resolution/GUI-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D super-resolution notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **patience**:
            
            .. image:: ../img/super-resolution/Notebooks-basic-training-params-2D.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.PATIENCE`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.


For improving performance, other advanced parameters can be optimized, for example, the model's architecture. The architecture assigned as default is usually the RCAN, as it is effective in super-resolution tasks. This architecture allows a strong baseline, but further exploration could potentially lead to better results.

.. note:: Once the parameters are correctly assigned, the training phase can be executed. Note that to train large models effectively the use of a GPU (Graphics Processing Unit) is essential. This hardware accelerator performs parallel computations and has larger RAM memory compared to the CPUs, which enables faster training times.


.. _super_resolution_data_run:

How to run
~~~~~~~~~~
BiaPy offers different options to run workflows depending on your degree of computer expertise. Select whichever is more approppriate for you:

.. tabs::

   .. tab:: GUI

        In the BiaPy GUI, navigate to *Workflow*, then select *Super-resolution* and follow the on-screen instructions:

        .. image:: ../img/gui/biapy_gui_sr.png
            :align: center

        \

        .. note:: BiaPy's GUI requires that all data and configuration files reside on the same machine where the GUI is being executed.
        
        .. tip:: If you need additional help, watch BiaPy's `GUI walkthrough video <https://www.youtube.com/embed/vY7aBh5FUNk?si=yvVolBnu5APNeHwB>`__. 

   .. tab:: Google Colab

        BiaPy offers two code-free notebooks in Google Colab to perform super-resolution:

        .. |sr_2D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/super-resolution/BiaPy_2D_Super_Resolution.ipynb

        * For 2D images: |sr_2D_colablink|

        .. |sr_3D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/super-resolution/BiaPy_3D_Super_Resolution.ipynb

        * For 3D images: |sr_3D_colablink|
      
        \

        .. tip:: If you need additional help, watch BiaPy's `Notebook walkthrough video <https://youtu.be/KEqfio-EnYw>`__.

   .. tab:: Docker

        If you installed BiaPy via Docker, `open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. For instance, you can use the `2d_super-resolution.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/super-resolution/2d_super-resolution.yaml>`__ template file (or your own file), and run the workflow as follows:

        .. code-block:: bash                                                                                                    

            # Configuration file
            job_cfg_file=/home/user/2d_super-resolution.yaml
            # Path to the data directory
            data_dir=/home/user/data
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results
            # Just a name for the job
            job_name=my_2d_super_resolution
            # Number that should be increased when one need to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0

            sudo docker run --rm \
                --gpus "device=$gpu_number" \
                --mount type=bind,source=$job_cfg_file,target=$job_cfg_file \
                --mount type=bind,source=$result_dir,target=$result_dir \
                --mount type=bind,source=$data_dir,target=$data_dir \
                biapyx/biapy:latest-11.8 \
                    --config $job_cfg_file \
                    --result_dir $result_dir \
                    --name $job_name \
                    --run_id $job_counter \
                    --gpu "$gpu_number"

        .. note:: 
            Note that ``data_dir`` must contain all the paths ``DATA.*.PATH`` and ``DATA.*.GT_PATH`` so the container can find them. For instance, if you want to only train in this example ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` could be ``/home/user/data/train/x`` and ``/home/user/data/train/y`` respectively. 

   .. tab:: Command line

        `From a terminal <../get_started/faq.html#opening-a-terminal>`__, you can use the `2d_super-resolution.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/super-resolution/2d_super-resolution.yaml>`__ template file (or your own file), and run the workflow as follows:

        .. code-block:: bash
            
            # Configuration file
            job_cfg_file=/home/user/2d_super-resolution.yaml       
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results  
            # Just a name for the job
            job_name=my_2d_super_resolution      
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
    

Templates                                                                                                                 
~~~~~~~~~~

In the `templates/super-resolution <https://github.com/BiaPyX/BiaPy/tree/master/templates/super-resolution>`__ folder of BiaPy, you will find a few YAML configuration templates for this workflow. 


[Advanced] Special workflow configuration 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: This section is recommended for experienced users only to improve the performance of their workflows. When in doubt, do not hesitate to check our `FAQ & Troubleshooting <../get_started/faq.html>`__ or open a question in the `image.sc discussion forum <our FAQ & Troubleshooting section>`_.

Advanced Parameters 
*******************
Many of the parameters of our workflows are set by default to values that work commonly well. However, it may be needed to tune them to improve the results of the workflow. For instance, you may modify the following parameters:

* **Model architecture**: Select the architecture of the deep neural network used as backbone of the pipeline. Options: EDSR, RCAN, WDSR, DFCAN, U-Net, Residual U-Net, Attention U-Net, SEUNet, MultiResUNet, ResUNet++, ResUNet SE and U-NeXt V1. Safe option: RCAN.
* **Batch size**: This parameter defines the number of patches seen in each training step. Reducing or increasing the batch size may slow or speed up your training, respectively, and can influence network performance. Common values are 4, 8, 16, etc.
* **Patch size**: Input the size of the patches use to train your model (length in pixels in X and Y). The value should be smaller or equal to the dimensions of the image. The default value is 256 in 2D, i.e. 256x256 pixels.
* **Optimizer**: Select the optimizer used to train your model. Options: ADAM, ADAMW, Stochastic Gradient Descent (SGD). ADAM usually converges faster, while ADAMW provides a balance between fast convergence and better handling of weight decay regularization. SGD is known for better generalization. Default value: ADAMW.
* **Initial learning rate**: Input the initial value to be used as learning rate. If you select ADAM as optimizer, this value should be around 10e-4. 
* **Learning rate scheduler**: Select to adjust the learning rate between epochs. The current options are "Reduce on plateau", "One cycle", "Warm-up cosine decay" or no scheduler.
* **Test time augmentation (TTA)**: Select to apply augmentation (flips and rotations) at test time. It usually provides more robust results but uses more time to produce each result. By default, no TTA is peformed.

Metrics
*******
During the training and inference phases (if HR test images were provided, i.e. ground truth, and consequently, ``DATA.TEST.LOAD_GT`` is ``True``) the performance of the model is measured using different metrics. Those metrics can be defined programmatically using the ``TRAIN.METRICS`` and ``TEST.METRICS`` variables of the YAML configuration file (a list of them is possible).

During training and test, the following metrics are available:

* **Peak signal-to-noise ratio** (`PSNR <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__). Keyword: ``psnr``.
* **Mean absolute error** (`MAE <https://en.wikipedia.org/wiki/Mean_absolute_error>`__). Keyword: ``mae``.
* **Mean squared error** (`MSE <https://en.wikipedia.org/wiki/Mean_squared_error>`__). Keyword: ``mse``.
* **Structural similarity index measure** (`SSIM <https://en.wikipedia.org/wiki/Structural_similarity_index_measure>`__). Keyword: ``ssim``.

Additionally, during test, if the images are 2D, the following metrics are also available:

* **Fréchet inception distance** (`FID <https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance>`__). Keyword: ``fid``.
* **Inception score** (`IS <https://en.wikipedia.org/wiki/Inception_score>`__). Keyword: ``is``.
* **Learned perceptual image patch similarity**  (`LPIPS <https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html>`__). Keyword: ``lpips``.


.. _super_resolution_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. An example of this workflow is depicted below:

.. list-table:: 
  :align: center
  :width: 680px

  * - .. figure:: ../img/pred_sr.png
         :align: center
         :width: 300px

         Predicted HR image.

    - .. figure:: ../img/HR_sr.png
         :align: center
         :width: 300px

         Target HR image.

Here both images are of size ``1004x1004``. 


Following the example, you should see that the directory ``/home/user/exp_results/my_2d_super_resolution`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: 

.. collapse:: Expand directory tree 

    .. code-block:: bash
        
      my_2d_super_resolution/
      ├── config_files
      │   └── 2d_super-resolution.yaml                                                                                                           
      ├── checkpoints
      │   └── my_2d_super-resolution_1-checkpoint-best.pth
      └── results
          ├── my_2d_super_resolution_1
          ├── . . .
          └── my_2d_super_resolution_5
              ├── aug
              │   └── .tif files
              ├── charts
              │   ├── my_2d_super_resolution_1_*.png
              │   └── my_2d_super_resolution_1_loss.png
              ├── per_image
              │   ├── .tif files
              │   └── .zarr files (or.h5)
              ├── train_logs
              └── tensorboard

\

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

  * ``2d_super-resolution.yaml``: YAML configuration file used (it will be overwrited every time the code is run)

* ``checkpoints``, *optional*: directory where model's weights are stored. Only created when ``TRAIN.ENABLE`` is ``True`` and the model is trained for at least one epoch. 

  * ``my_2d_super-resolution_1-checkpoint-best.pth``, *optional*: checkpoint file (best in validation) where the model's weights are stored among other information. Only created when the model is trained for at least one epoch. 

  * ``normalization_mean_value.npy``, *optional*: normalization mean value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.
  
  * ``normalization_std_value.npy``, *optional*: normalization std value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

  * ``my_2d_super_resolution_1``: run 1 experiment folder. Can contain:

    * ``aug``, *optional*: image augmentation samples. Only created if ``AUGMENTOR.AUG_SAMPLES`` is ``True``.

    * ``charts``, *optional*: only created when ``TRAIN.ENABLE`` is ``True`` and epochs trained are more or equal ``LOG.CHART_CREATION_FREQ``. Can contain:

      * ``my_2d_super_resolution_1_*.png``: Plot of each metric used during training.

      * ``my_2d_super_resolution_1_loss.png``: Loss over epochs plot. 

    * ``per_image``:

      * ``.tif files``, *optional*: reconstructed images from patches. Created when ``TEST.BY_CHUNKS.ENABLE`` is ``False`` or when ``TEST.BY_CHUNKS.ENABLE`` is ``True`` but ``TEST.BY_CHUNKS.SAVE_OUT_TIF`` is ``True``. 

      * ``.zarr files (or.h5)``, *optional*: reconstructed images from patches. Created when ``TEST.BY_CHUNKS.ENABLE`` is ``True``.

    * ``train_logs``: each row represents a summary of each epoch stats. Only avaialable if training was done.

    * ``tensorboard``: Tensorboard logs.

.. note:: 
   Here, for visualization purposes, only ``my_2d_super_resolution_1`` has been described but ``my_2d_super_resolution_2``, ``my_2d_super_resolution_3``, ``my_2d_super_resolution_4`` and ``my_2d_super_resolution_5`` will follow the same structure.


