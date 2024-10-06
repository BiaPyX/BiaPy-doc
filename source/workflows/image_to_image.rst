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
        
        When calling BiaPy from command line, you can specify the output folder with the ``--result_dir`` flag. See the *Command line* configuration of :ref:`i2i_data_run` for a full example.


.. list-table::
  :align: center

  * - .. figure:: ../img/i2i/Inputs-outputs.svg
         :align: center
         :width: 500
         :alt: Graphical description of minimal inputs and outputs in BiaPy for image-to-image translation.
        
         **BiaPy input and output folders for image-to-image translation**.



.. _i2i2_data_prep:

Data structure
**************

To ensure the proper operation of the library, the data directory tree should be something like this: 

.. code-block::

  dataset/
  ├── train
  │   ├── x
  │   │   ├── training-0001.tif
  │   │   ├── training-0002.tif
  │   │   ├── . . .
  │   │   └── training-9999.tif
  │   └── y
  │       ├── training_groundtruth-0001.tif
  │       ├── training_groundtruth-0002.tif
  │       ├── . . .
  │       └── training_groundtruth-9999.tif
  └── test
      ├── x
      │   ├── testing-0001.tif
      │   ├── testing-0002.tif
      │   ├── . . .
      │   └── testing-9999.tif
      └── y
          ├── testing_groundtruth-0001.tif
          ├── testing_groundtruth-0002.tif
          ├── . . .
          └── testing_groundtruth-9999.tif

\

In this example, the raw training images are under ``dataset/train/x/`` and their corresponding target images are under ``dataset/train/y/``, while the raw test images are under ``dataset/test/x/`` and their corresponding target images are under ``dataset/test/y/``. **This is just an example**, you can name your folders as you wish as long as you set the paths correctly later.

.. note:: Make sure that raw and target images are sorted in the same way. A common approach is to fill with zeros the image number added to the filenames (as in the example).


Minimal configuration
~~~~~~~~~~~~~~~~~~~~~
Apart from the input and output folders, there are a few basic parameters that always need to be specified in order to run an image-to-image workflow in BiaPy. **These parameters can be introduced either directly in the GUI, the code-free notebooks or by editing the YAML configuration file**.

Experiment name
***************
Also known as "model name" or "job name", this will be the name of the current experiment you want to run, so it can be differenciated from other past and future experiments.

.. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, type the name you want for the job in the **Job name** field:

        .. image:: ../img/i2i/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image-to-image translation notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **model_name**:
        
        .. image:: ../img/i2i/Notebooks-model-name-data-conf.png
          :align: center
          :width: 75%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--name`` flag. See the *Command line* configuration of :ref:`i2i_data_run` for a full example.

\

.. note:: Use only *my_model* -style, not *my-model* (Use "_" not "-"). Do not use spaces in the name. Avoid using the name of an existing experiment/model/job (saved in the same result folder) as it will be overwritten..

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
        
         Graphical description of data partitions in BiaPy.



To define such set, there are two options:
  
* **Validation proportion/percentage**: Select a proportion (or percentage) of your training dataset to be used to validate the network during the training. Usual values are 0.1 (10%) or 0.2 (20%), and the samples of that set will be selected at random.
  
  .. collapse:: Expand to see how to configure

      .. tabs::
        .. tab:: GUI

          Under *Workflow*, select *Image to image*, click twice on *Continue*, and under *General options* > *Advanced options* > *Validation data*, select "Extract from train (split training)" in **Validation type**, and introduce your value (between 0 and 1) in the **Train proportion for validation**:

          .. image:: ../img/GUI-validation-percentage.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          In either the 2D or the 3D image-to-image translation notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **percentage_validation** with a value between 0 and 100:
          
          .. image:: ../img/i2i/Notebooks-model-name-data-conf.png
            :align: center
            :width: 75%

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.SPLIT_TRAIN`` with a value between 0 and 1, representing the proportion of the training set that will be set apart for validation.

* **Validation paths**: Similar to the training and test sets, you can select two folders with the validation raw and target images:

  * **Validation Raw Images**: A folder that contains the unprocessed (single-channel or multi-channel) images that will be used to select the best model during training.
  
    .. collapse:: Expand to see how to configure

      .. tabs::
        .. tab:: GUI

          Under *Workflow*, select *Image to image*, click twice on *Continue*, and under *General options* > *Advanced options* > *Validation data*, select "Not extracted from train (path needed)" in **Validation type**, click on the *Browse* button of **Input raw image folder** and select the folder containing your validation raw images:

          .. image:: ../img/i2i/GUI-validation-paths.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          This option is currently not available in the notebooks.

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.PATH`` with the absolute path to your validation raw images.

  * **Validation Target Images**: A folder that contains the semantic label (single-channel) images for validation. Ensure the number and dimensions match the validation raw images.
  
    .. collapse:: Expand to see how to configure

      .. tabs::
        .. tab:: GUI

          Under *Workflow*, select *Image to image*, click twice on *Continue*, and under *General options* > *Advanced options* > *Validation data*, select "Not extracted from train (path needed)" in **Validation type**, click on the *Browse* button of **Input label folder** and select the folder containing your validation label images:

          .. image:: ../img/i2i/GUI-validation-paths.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          This option is currently not available in the notebooks.

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.GT_PATH`` with the absolute path to your validation target images.


Test ground-truth
"""""""""""""""""
Do you have target images for the test set? This is a key question so BiaPy knows if your test set will be used for evaluation in new data (unseen during training) or simply produce predictions on that new data. All supervised workflows contain a parameter to specify this aspect.

.. collapse:: Expand to see how to configure

  .. tabs::
    .. tab:: GUI

      Under *Workflow*, select *Image to image*, three times *Continue*, under *General options* > *Test data*, select "No" or "Yes" in the **Do you have target test data?** field:

      .. image:: ../img/i2i/GUI-test-data.png
        :align: center

    .. tab:: Google Colab / Notebooks
      
      In either the 2D or the 3D image-to-image translation notebook, go to *Configure and train the DNN model* > *Select your parameters*, and check or uncheck the **test_ground_truth** option:
      
      .. image:: ../img/i2i/Notebooks-model-name-data-conf.png
        :align: center
        :width: 50%


    .. tab:: YAML configuration file
      
      Set the variable ``DATA.TEST.LOAD_GT`` to ``True`` if you do have target images in your test set, or ``False`` otherwise.


\

Basic training parameters
*************************
At the core of each BiaPy workflow there is a deep learning model. Although we try to simplify the number of parameters to tune, these are the basic parameters that need to be defined for training an image-to-image translation workflow:

* **Number of input channels**: The number of channels of your raw images (grayscale = 1, RGB = 3). Notice the dimensionality of your images (2D/3D) is set by default depending on the workflow template you select.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image to image*, click twice on *Continue*, and under *General options* > *Train data*, edit the last value of the field **Data patch size** with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D:

            .. image:: ../img/i2i/GUI-general-options.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D image-to-image translation notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **input_channels**:
            
            .. image:: ../img/i2i/Notebooks-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``DATA.PATCH_SIZE`` with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D.

* **Number of epochs**: This number indicates how many `rounds <https://machine-learning.paperspace.com/wiki/epoch>`_ the network will be trained. On each round, the network usually sees the full training set. The value of this parameter depends on the size and complexity of each dataset. You can start with something like 100 epochs and tune it depending on how fast the loss (error) is reduced.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image to image*, click twice on *Continue*, and under *General options*, click on *Advanced options*, scroll down to *General training parameters*, and edit the field **Number of epochs**:

            .. image:: ../img/i2i/GUI-basic-training-params.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D image-to-image translation notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **number_of_epochs**:
            
            .. image:: ../img/i2i/Notebooks-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.EPOCHS`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.

* **Patience**: This is a number that indicates how many epochs you want to wait without the model improving its results in the validation set to stop training. Again, this value depends on the data you're working on, but you can start with something like 20.
   
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image to image*, click twice on *Continue*, and under *General options*, click on *Advanced options*, scroll down to *General training parameters*, and edit the field **Patience**:

            .. image:: ../img/i2i/GUI-basic-training-params.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D image-to-image translation notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **patience**:
            
            .. image:: ../img/i2i/Notebooks-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.PATIENCE`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.


For improving performance, other advanced parameters can be optimized, for example, the model's architecture. A common choice is the U-Net, as it is effective in image-to-image translation tasks. This architecture allows a strong baseline, but further exploration could potentially lead to better results.

.. note:: Once the parameters are correctly assigned, the training phase can be executed. Note that to train large models effectively the use of a GPU (Graphics Processing Unit) is essential. This hardware accelerator performs parallel computations and has larger RAM memory compared to the CPUs, which enables faster training times.

.. _i2i_data_run:

How to run
~~~~~~~~~~
BiaPy offers different options to run workflows depending on your degree of computer expertise. Select whichever is more approppriate for you:

.. tabs::

   .. tab:: GUI

        In the BiaPy GUI, navigate to *Workflow*, then select *Image to image* and follow the on-screen instructions:

        .. image:: ../img/gui/biapy_gui_i2i.png
            :align: center 
        
        \
        
        **Tip**: If you need additional help, watch BiaPy's `GUI walk-through video <https://www.youtube.com/embed/vY7aBh5FUNk?si=yvVolBnu5APNeHwB>`__.


   .. tab:: Google Colab

        BiaPy offers two code-free notebooks in Google Colab to perform image to image translation: 

        .. |sr_2D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/image_to_image/BiaPy_2D_Image_to_Image.ipynb

        * For 2D images: |sr_2D_colablink|

        .. |sr_3D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/image_to_image/BiaPy_3D_Image_to_Image.ipynb

        * For 3D images: |sr_3D_colablink|

   .. tab:: Docker

        If you installed BiaPy via Docker, `open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. . Then, you can use the `2d_image-to-image.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/image-to-image/2d_image-to-image.yaml>`__ template file (or your own file), and run the workflow as follows:

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

        `From a terminal <../get_started/faq.html#opening-a-terminal>`__, you can use the `2d_image-to-image.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/image-to-image/2d_image-to-image.yaml>`__ template file (or your own file), and run the workflow as follows:

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
      


Templates                                                                                                                 
~~~~~~~~~

In the `templates/image-to-image <https://github.com/BiaPyX/BiaPy/tree/master/templates/image-to-image>`__ folder of BiaPy, you can find a few YAML configuration templates for this workflow. 

[Advanced] Special workflow configuration 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: This section is recommended for experienced users only to improve the performance of their workflows. When in doubt, do not hesitate to check our `FAQ & Troubleshooting <../get_started/faq.html>`__ or open a question in the `image.sc discussion forum <our FAQ & Troubleshooting section>`_.

Advanced Parameters 
*******************
Many of the parameters of our workflows are set by default to values that work commonly well. However, it may be needed to tune them to improve the results of the workflow. For instance, you may modify the following parameters

* **Model architecture**: Select the architecture of the deep neural network used as backbone of the pipeline. Options: EDSR, RCAN, WDSR, DFCAN, U-Net, Residual U-Net, Attention U-Net, SEUNet, MultiResUNet, ResUNet++, UNETR-Mini, UNETR-Small, UNETR-Base, ResUNet SE and U-NeXt V1. Safe choice: U-Net.
* **Batch size**: This parameter defines the number of patches seen in each training step. Reducing or increasing the batch size may slow or speed up your training, respectively, and can influence network performance. Common values are 4, 8, 16, etc.
* **Patch size**: Input the size of the patches use to train your model (length in pixels in X and Y). The value should be smaller or equal to the dimensions of the image. The default value is 256 in 2D, i.e. 256x256 pixels.
* **Optimizer**: Select the optimizer used to train your model. Options: ADAM, ADAMW, Stochastic Gradient Descent (SGD). ADAM usually converges faster, while ADAMW provides a balance between fast convergence and better handling of weight decay regularization. SGD is known for better generalization. Default value: ADAMW.
* **Initial learning rate**: Input the initial value to be used as learning rate. If you select ADAM as optimizer, this value should be around 10e-4. 
* **Learning rate scheduler**: Select to adjust the learning rate between epochs. The current options are "Reduce on plateau", "One cycle", "Warm-up cosine decay" or no scheduler.
* **Test time augmentation (TTA)**: Select to apply augmentation (flips and rotations) at test time. It usually provides more robust results but uses more time to produce each result. By default, no TTA is peformed.
* **Multiple raw inputs**. If each training sample is composed by several images, e.g. transformed versions of the sample, you need to set ``PROBLEM.IMAGE_TO_IMAGE.MULTIPLE_RAW_ONE_TARGET_LOADER``. Find an example of this configuration in the `LightMyCells tutorial <https://biapy.readthedocs.io/en/latest/tutorials/image-to-image/lightmycells.html>`__. 


Metrics
*******
During the inference phase, the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is ``True``. In the case of image-to-image the **Peak signal-to-noise ratio** (`PSNR <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__) metric is calculated when the target image is reconstructed from individual patches.

  
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


