.. _denoising:

Image denoising
---------------

Description of the task
~~~~~~~~~~~~~~~~~~~~~~~

The goal of this workflow is to remove noise from the input images. BiaPy makes use of Noise2Void :cite:p:`krull2019noise2void` with any of the U-Net-like models provided. The main advantage of Noise2Void is neither relying on noise image pairs nor clean target images since frequently clean images are simply unavailable.

An example of this task is displayed in the figure below, with an noisy fluorescence image and its corresponding denoised output:

.. role:: raw-html(raw)
    :format: html

.. figure:: ../img/denosing_overview.svg
   :align: center
   :width: 75%                

   **Example of denoising task**. Left: original (noisy) fluorescence image. Right: denoised  :raw-html:`<br />` output version of the same image. Blue squares show zoommed areas of both images. 


Inputs and outputs
~~~~~~~~~~~~~~~~~~
The denoising workflows in BiaPy expect a series of **folders** as input:

* **Training Raw Images**: A folder that contains the unprocessed (single-channel or multi-channel) images that will be used to train the model.
  
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image denoising*, twice *Continue*, under *General options* > *Train data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/denoising/GUI-train-general-options.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D denoising notebook, go to *Paths for Input Images and Output Files*, edit the field **train_data_path**:
        
        .. image:: ../img/denoising/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 95%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TRAIN.PATH`` with the absolute path to the folder with your training raw images.

* .. raw:: html

      <b><span style="color: darkgreen;">[Optional]</span> Test Raw Images</b>: A folder that contains the images to evaluate the model's performance.
 
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image denoising*, three times *Continue*, under *General options* > *Test data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/denoising/GUI-test-data.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D denoising notebook, go to *Paths for Input Images and Output Files*, edit the field **test_data_path**:
        
        .. image:: ../img/denoising/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 95%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TEST.PATH`` with the absolute path to the folder with your test raw images.

Upon successful execution, a directory will be generated with the denoising results. Therefore, you will need to define:

* **Output Folder**: A designated path to save the denoising outcomes.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, click on the *Browse* button of **Output folder to save the results**:

        .. image:: ../img/denoising/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D denoising notebook, go to *Paths for Input Images and Output Files*, edit the field **output_path**:
        
        .. image:: ../img/denoising/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 95%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--result_dir`` flag. See the *Command line* configuration of :ref:`denoising_data_run` for a full example.


.. list-table::
  :align: center

  * - .. figure:: ../img/denoising/Inputs-outputs.svg
         :align: center
         :width: 500
         :alt: Graphical description of minimal inputs and outputs in BiaPy for image denoising.
        
         **BiaPy input and output folders for image denoising.** Since this workflow is self-supervised, :raw-html:`<br />` no labels are needed in neither train nor test.
  


.. tip:: To denoise the entire dataset with Noise2Void:

    (1) **Train on the full dataset**: Since Noise2Void learns from the noise patterns within the images themselves, every image in your dataset contributes to the training.
    (2) **Test on all images as well**: After training, you can apply the model to denoise the entire dataset.



.. _denoising_data_prep:

Data structure
**************

To ensure the proper operation of the workflow, the directory tree should be something like this: 

.. code-block::
    
  dataset/
  ├── train
  │   ├── training-0001.tif
  │   ├── training-0002.tif
  │   ├── . . .
  │   └── training-9999.tif   
  └── test
      ├── testing-0001.tif
      ├── testing-0002.tif
      ├── . . .
      └── testing-9999.tif

\

In this example, the training images are under ``dataset/train/``, while the test images are under ``dataset/test/``. **This is just an example**, you can name your folders as you wish as long as you set the paths correctly later.

Example datasets
****************
Below is a list of publicly available datasets that are ready to be used in BiaPy for image denoising:

.. list-table::
  :widths: auto
  :header-rows: 1
  :align: center

  * - Example dataset
    - Image dimensions
    - Link to data
  * - `Noise2void Convallaria 2D (by B. Schroth-Diez) <https://github.com/juglab/n2v>`__
    - 2D
    - `convallaria2D.zip <https://drive.google.com/file/d/1TFvOySOiIgVIv9p4pbHdEbai-d2YGDvV/view?usp=drive_link>`__
  * - `Noise2void Flywing 3D (by R. Piscitello) <https://github.com/juglab/n2v>`__
    - 3D
    - `flywing3D.zip <https://drive.google.com/file/d/1OIjnUoJKdnbClBlpzk7V5R8wtoLont-r/view?usp=drive_link>`__


Minimal configuration
~~~~~~~~~~~~~~~~~~~~~
Apart from the input and output folders, there are a few basic parameters that always need to be specified in order to run an denoising workflow in BiaPy. **These parameters can be introduced either directly in the GUI, the code-free notebooks or by editing the YAML configuration file**.

Experiment name
***************
Also known as "model name" or "job name", this will be the name of the current experiment you want to run, so it can be differenciated from other past and future experiments.

.. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, type the name you want for the job in the **Job name** field:

        .. image:: ../img/denoising/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D denoising notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **model_name**:
        
        .. image:: ../img/denoising/Notebooks-model-name-data-conf.png
          :align: center
          :width: 65%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--name`` flag. See the *Command line* configuration of :ref:`denoising_data_run` for a full example.


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

          Under *Workflow*, select *Image denoising*, click twice on *Continue*, and under *Advanced options* > *Validation data*, select "Extract from train (split training)" in **Validation type**, and introduce your value (between 0 and 1) in the **Train prop. for validation**:

          .. image:: ../img/GUI-validation-percentage.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          In either the 2D or the 3D denoising notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **percentage_validation** with a value between 0 and 100:
          
          .. image:: ../img/denoising/Notebooks-model-name-data-conf.png
            :align: center
            :width: 75%

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.SPLIT_TRAIN`` with a value between 0 and 1, representing the proportion of the training set that will be set apart for validation.


* **Validation path**: Similar to the training and test sets, you can select a folder that contains the unprocessed (single-channel or multi-channel) raw images that will be used to validate the current model during training.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image denoising*, click twice on *Continue*, and under *Advanced otions* > *Validation data*, select "Not extracted from train (path needed)" in **Validation type**, click on the *Browse* button of **Input raw image folder** and select the folder containing your validation raw images:

        .. image:: ../img/denoising/GUI-validation-paths.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        This option is currently not available in the notebooks.

      .. tab:: YAML configuration file
      
        Edit the variable ``DATA.VAL.PATH`` with the absolute path to your validation raw images.

 

Basic training parameters
*************************
At the core of each BiaPy workflow there is a deep learning model. Although we try to simplify the number of parameters to tune, these are the basic parameters that need to be defined for training a denoising workflow:

* **Number of input channels**: The number of channels of your raw images (grayscale = 1, RGB = 3). Notice the dimensionality of your images (2D/3D) is set by default depending on the workflow template you select.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image denoising*, click twice on *Continue*, and under *General options* > *Train data*, edit the last value of the field **Data patch size** with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D:

            .. image:: ../img/denoising/GUI-general-options.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D denoising notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **input_channels**:
            
            .. image:: ../img/denoising/Notebooks-basic-training-params.png
              :align: center

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``DATA.PATCH_SIZE`` with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D.

* **Number of epochs**: This number indicates how many `rounds <https://machine-learning.paperspace.com/wiki/epoch>`_ the network will be trained. On each round, the network usually sees the full training set. The value of this parameter depends on the size and complexity of each dataset. You can start with something like 100 epochs and tune it depending on how fast the loss (error) is reduced.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image denoising*, click twice on *Continue*, and under *Advanced options*, scroll down to *General training parameters*, and edit the field **Number of epochs**:

            .. image:: ../img/denoising/GUI-basic-training-params.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D denoising notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **number_of_epochs**:
            
            .. image:: ../img/denoising/Notebooks-basic-training-params.png
              :align: center

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.EPOCHS`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.

* **Patience**: This is a number that indicates how many epochs you want to wait without the model improving its results in the validation set to stop training. Again, this value depends on the data you're working on, but you can start with something like 20.
   
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image denoising*, click twice on *Continue*, and under *Advanced options*, scroll down to *General training parameters*, and edit the field **Patience**:

            .. image:: ../img/denoising/GUI-basic-training-params.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D denoising notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **patience**:
            
            .. image:: ../img/denoising/Notebooks-basic-training-params.png
              :align: center

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.PATIENCE`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.


For improving performance, other advanced parameters can be optimized, for example, the model's architecture. The architecture assigned as default is usually the U-Net, as it is effective in denoising tasks. This architecture allows a strong baseline, but further exploration could potentially lead to better results.

.. note:: Once the parameters are correctly assigned, the training phase can be executed. Note that to train large models effectively the use of a GPU (Graphics Processing Unit) is essential. This hardware accelerator performs parallel computations and has larger RAM memory compared to the CPUs, which enables faster training times.

.. _denoising_data_run:

How to run
~~~~~~~~~~
BiaPy offers different options to run workflows depending on your degree of computer expertise. Select whichever is more approppriate for you:

.. tabs::
   .. tab:: GUI

        In the BiaPy GUI, navigate to *Workflow*, then select *Image denoising* and follow the on-screen instructions:

        .. image:: ../img/gui/biapy_gui_denoising.png
            :align: center

        \
        
        .. note:: BiaPy's GUI requires that all data and configuration files reside on the same machine where the GUI is being executed.

        .. tip:: If you need additional help, watch BiaPy's `GUI walkthrough video <https://www.youtube.com/embed/vY7aBh5FUNk?si=yvVolBnu5APNeHwB>`__.

   .. tab:: Google Colab

        BiaPy offers two code-free notebooks in Google Colab to perform image denoising:

        .. |denoising_2D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/denoising/BiaPy_2D_Denoising.ipynb

        * For 2D images: |denoising_2D_colablink|

        .. |denoising_3D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/denoising/BiaPy_3D_Denoising.ipynb

        * For 3D images: |denoising_3D_colablink|
      
        \

        .. tip:: If you need additional help, watch BiaPy's `Notebook walkthrough video <https://youtu.be/KEqfio-EnYw>`__.

   .. tab:: Docker

        If you installed BiaPy via Docker, `open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. For instance, you can use the `2d_denoising.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/denoising/2d_denoising.yaml>`__ template file (or your own YAML file), and then run the workflow as follows:

        .. code-block:: bash                                                                                                    

            # Configuration file
            job_cfg_file=/home/user/2d_denoising.yaml
            # Path to the data directory
            data_dir=/home/user/data
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results
            # Just a name for the job
            job_name=my_2d_denoising
            # Number that should be increased when one need to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0

            docker run --rm \
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

        `From a terminal <../get_started/faq.html#opening-a-terminal>`__, you can for instance use the `2d_denoising.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/denoising/2d_denoising.yaml>`__ template (or your own YAML file)to run the workflow as follows:

        .. code-block:: bash
            
            # Configuration file
            job_cfg_file=/home/user/2d_denoising.yaml       
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results  
            # Just a name for the job
            job_name=2d_denoising      
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

   


Templates                                                                                                                 
~~~~~~~~~~

In the `templates/denoising <https://github.com/BiaPyX/BiaPy/tree/master/templates/denoising>`__ folder of BiaPy, you can find a few YAML configuration templates for this workflow. 


[Advanced] Special workflow configuration 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: This section is recommended for experienced users only to improve the performance of their workflows. When in doubt, do not hesitate to check our `FAQ & Troubleshooting <../get_started/faq.html>`__ or open a question in the `image.sc discussion forum <our FAQ & Troubleshooting section>`_.

Advanced Parameters 
*******************
Many of the parameters of our workflows are set by default to values that work commonly well. However, it may be needed to tune them to improve the results of the workflow. For instance, you may modify the following parameters:

* **Model architecture**: Select the architecture of the deep neural network used as backbone of the pipeline. Options: U-Net, Residual U-Net, Attention U-Net, SEUNet, MultiResUNet, ResUNet++, UNETR-Mini, UNETR-Small, UNETR-Base, ResUNet SE and U-NeXt V1. Safe option: U-Net.
* **Batch size**: This parameter defines the number of patches seen in each training step. Reducing or increasing the batch size may slow or speed up your training, respectively, and can influence network performance. Common values are 4, 8, 16, etc.
* **Patch size**: Input the size of the patches use to train your model (length in pixels in X and Y). The value should be smaller or equal to the dimensions of the image. The default value is 64 in 2D, i.e. 64x64 pixels.
* **Optimizer**: Select the optimizer used to train your model. Options: ADAM, ADAMW, Stochastic Gradient Descent (SGD). ADAM usually converges faster, while ADAMW provides a balance between fast convergence and better handling of weight decay regularization. SGD is known for better generalization. Default value: ADAMW.
* **Initial learning rate**: Input the initial value to be used as learning rate. If you select ADAM as optimizer, this value should be around 10e-4. 
* **Learning rate scheduler**: Select to adjust the learning rate between epochs. The current options are "Reduce on plateau", "One cycle", "Warm-up cosine decay" or no scheduler.
* **Test time augmentation (TTA)**: Select to apply augmentation (flips and rotations) at test time. It usually provides more robust results but uses more time to produce each result. By default, no TTA is peformed.


Noise2Void Parameters 
*********************
Please refer to `Noise2Void <https://arxiv.org/abs/1811.10980>`__  to understand the method functionality. These variables can be set:

* ``PROBLEM.DENOISING.N2V_PERC_PIX`` controls the percentage of pixels per input patch to be manipulated. This is the ``n2v_perc_pix`` in their code. 

* ``PROBLEM.DENOISING.N2V_MANIPULATOR`` controls how the pixels will be replaced. This is the ``n2v_manipulator`` in their code. 

* ``PROBLEM.DENOISING.N2V_NEIGHBORHOOD_RADIUS`` controls the radius of the neighborhood. This is the ``n2v_neighborhood_radius`` in their code. 

* ``PROBLEM.DENOISING.N2V_STRUCTMASK`` whether to use `Struct Noise2Void <https://github.com/juglab/n2v/blob/main/examples/2D/structN2V_2D_convallaria/>`__. 



.. _denoising_results:

Results                                                                                                                 
~~~~~~~  

The results are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. You should see that the directory ``/home/user/exp_results/my_2d_denoising`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: 

.. collapse:: Expand directory tree 

    .. code-block:: bash

      my_2d_denoising/
      ├── config_files
      │   └── my_2d_denoising.yaml                                                                                                           
      ├── checkpoints
      |   ├── my_2d_denoising_1-checkpoint-best.pth
      |   ├── normalization_mean_value.npy
      │   └── normalization_std_value.npy
      └── results
          ├── my_2d_denoising
          ├── . . .
          └── my_2d_denoising
              ├── cell_counter.csv
              ├── aug
              │   └── .tif files
              ├── charts
              │   ├── my_2d_denoising_1_n2v_mse.png
              │   └── my_2d_denoising_1_loss.png
              ├── per_image
              │   ├── .tif files
              │   └── .zarr files (or.h5)
              ├── train_logs
              └── tensorboard

\

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

  * ``my_2d_denoising.yaml``: YAML configuration file used (it will be overwrited every time the code is run).

* ``checkpoints``, *optional*: directory where model's weights are stored. Only created when ``TRAIN.ENABLE`` is ``True`` and the model is trained for at least one epoch. Can contain:

  * ``my_2d_denoising_1-checkpoint-best.pth``, *optional*: checkpoint file (best in validation) where the model's weights are stored among other information. Only created when the model is trained for at least one epoch. 

  * ``normalization_mean_value.npy``, *optional*: normalization mean value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.
  
  * ``normalization_std_value.npy``, *optional*: normalization std value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed. Can contain:

  * ``my_2d_denoising_1``: run 1 experiment folder. Can contain:

    * ``aug``, *optional*: image augmentation samples. Only created if ``AUGMENTOR.AUG_SAMPLES`` is ``True``.

    * ``charts``, *optional*: only created when ``TRAIN.ENABLE`` is ``True`` and epochs trained are more or equal ``LOG.CHART_CREATION_FREQ``. Can contain:  

      * ``my_2d_denoising_1_*.png``: plot of each metric used during training.

      * ``my_2d_denoising_1_loss.png``: loss over epochs plot. 

    * ``per_image``, *optional*: only created if ``TEST.FULL_IMG`` is ``False``. Can contain:

      * ``.tif files``, *optional*: reconstructed images from patches. Created when ``TEST.BY_CHUNKS.ENABLE`` is ``False`` or when ``TEST.BY_CHUNKS.ENABLE`` is ``True`` but ``TEST.BY_CHUNKS.SAVE_OUT_TIF`` is ``True``.

      * ``.zarr files (or.h5)``, *optional*: reconstructed images from patches. Created when ``TEST.BY_CHUNKS.ENABLE`` is ``True``.

    * ``train_logs``: each row represents a summary of each epoch stats. Only avaialable if training was done.

    * ``tensorboard``: tensorboard logs.

.. note:: 

  Here, for visualization purposes, only ``my_2d_denoising_1`` has been described but ``my_2d_denoising_2``, ``my_2d_denoising_3``, ``my_2d_denoising_4`` and ``my_2d_denoising_5`` will follow the same structure.



