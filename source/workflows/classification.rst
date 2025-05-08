.. _classification:

Image classification
--------------------

Description of the task
~~~~~~~~~~~~~~~~~~~~~~~

The goal of this workflow is to assign a **category (or class)** to every input image. 

In the figure below a few examples of this workflow's **input** are depicted:

.. list-table::
  :align: center
  :width: 680px
  
  * - .. figure:: ../img/classification/MedMNIST_DermaMNIST_test1008_0.png
         :align: center
         :width: 50

    - .. figure:: ../img/classification/MedMNIST_DermaMNIST_test10_1.png
         :align: center
         :width: 50
         
    - .. figure:: ../img/classification/MedMNIST_DermaMNIST_test1002_2.png
         :align: center
         :width: 50

    - .. figure:: ../img/classification/MedMNIST_DermaMNIST_test1030_3.png
         :align: center
         :width: 50

    - .. figure:: ../img/classification/MedMNIST_DermaMNIST_test1003_4.png
         :align: center
         :width: 50

    - .. figure:: ../img/classification/MedMNIST_DermaMNIST_test0_5.png
         :align: center
         :width: 50

    - .. figure:: ../img/classification/MedMNIST_DermaMNIST_test1021_6.png
         :align: center
         :width: 50

Each of these examples are of a different class and were obtained from `MedMNIST v2 <https://medmnist.com/>`__ (:cite:p:`yang2021medmnist`), concretely from DermaMNIST dataset which is a large collection of multi-source dermatoscopic images of common
pigmented skin lesions.

Inputs and outputs
~~~~~~~~~~~~~~~~~~
The image classification workflows in BiaPy expect a series of **folders** as input:

* **Training Raw Images**: A folder that contains the unprocessed (single-channel or multi-channel) images that will be used to train the model. As explained later, all images of the same category are expected to be in the same sub-folder.
  
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image classification*, twice *Continue*, under *General options* > *Train data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/classification/GUI-train-general-options.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D self-supervision notebook, go to *Paths for Input Images and Output Files*, edit the field **train_data_path**:
        
        .. image:: ../img/classification/Notebooks-Inputs-Outputs.png
          :align: center

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TRAIN.PATH`` with the absolute path to the folder with your training raw images.

* .. raw:: html

      <b><span style="color: darkgreen;">[Optional]</span> Test Raw Images</b>: A folder that contains the images to evaluate the model's performance. Optionaly, if the category of each test image is known, all images of the same category are expected to be in the same sub-folder.
 
  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image classification*, three times *Continue*, under *General options* > *Test data*, click on the *Browse* button of **Input raw image folder**:

        .. image:: ../img/classification/GUI-test-data.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image classification notebook, go to *Paths for Input Images and Output Files*, edit the field **test_data_path**:
        
        .. image:: ../img/classification/Notebooks-Inputs-Outputs.png
          :align: center
          :width: 95%

      .. tab:: YAML configuration file
        
        Edit the variable ``DATA.TEST.PATH`` with the absolute path to the folder with your test raw images.


Upon successful execution, a directory will be generated with the results of the classification. Therefore, you will need to define:

* **Output Folder**: A designated path to save the classification outcomes.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, click on the *Browse* button of **Output folder to save the results**:

        .. image:: ../img/classification/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image classification notebook, go to *Paths for Input Images and Output Files*, edit the field **output_path**:
        
        .. image:: ../img/classification/Notebooks-Inputs-Outputs.png
          :align: center

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--result_dir`` flag. See the *Command line* configuration of :ref:`classification_data_run` for a full example.


.. role:: raw-html(raw)
    :format: html


.. list-table::
  :align: center

  * - .. figure:: ../img/classification/Inputs-outputs.svg
         :align: center
         :width: 500
         :alt: Graphical description of minimal inputs and outputs in BiaPy for image classification.
        
         **BiaPy input and output folders for image classification.** Notice the test folder :raw-html:`<br />` and its sub-folders are optional.


.. _classification_data_prep:

Data structure
**************

To ensure the proper operation of the workflow, the directory tree should be something like this: 
 
.. code-block::
    
  dataset/
  ├── train
  │   ├── class_0
  │   │   ├── train0_0.png
  │   │   ├── train1013_0.png
  │   │   ├── . . .
  │   │   └── train932_0.png
  │   ├── class_1
  │   │   ├── train104_1.png
  │   │   ├── train1049_1.png
  │   │   ├── . . .
  │   │   └── train964_1.png
  | . . .
  │   └── class_6
  │       ├── train1105_6.png
  │       ├── train1148_6.png
  │       ├── . . .
  │       └── train98_6.png
  └── test
      ├── class_0
      │   ├── test1008_0.png
      │   ├── test1084_0.png
      │   ├── . . .
      │   └── test914_0.png
      ├── class_1
      │   ├── test10_1.png
      │   ├── test1034_1.png
      │   ├── . . .
      │   └── test984_1.png
    . . .
      └── class_6
          ├── test1021_6.png
          ├── test1069_6.png
          ├── . . .
          └── test806_6.png

\

Each image category is obtained from the sub-folder name in which that image resides. That is why is so important to follow the directory tree as described above. If you have a .csv file with each image category, as is provided by `MedMNIST v2 <https://medmnist.com/>`__, you can use our script `from_class_csv_to_folders.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/utils/scripts/from_class_csv_to_folders.py>`__ to create such directory tree.

The **sub-folder names can be any number or string**. They will be considered as the class names. Regarding the test, if you have no classes it doesn't matter if the images are separated in several folders or are all in one folder.

Example datasets
****************
Below is a list of publicly available datasets that are ready to be used in BiaPy for image classification:

.. list-table::
  :widths: auto
  :header-rows: 1
  :align: center

  * - Example dataset
    - Image dimensions
    - Link to data
  * - `DermaMNIST <https://www.nature.com/articles/s41597-022-01721-8>`__
    - 2D
    - `DermaMNIST.zip <https://drive.google.com/file/d/15_pnH4_tJcwhOhNqFsm26NQuJbNbFSIN/view?usp=drive_link>`__
  * - `OrganMNIST3D <https://medmnist.com/>`__
    - 3D
    - `organMNIST3D.zip <https://drive.google.com/file/d/1pypWJ4Z9sRLPlVHbG6zpwmS6COkm3wUg/view?usp=drive_link>`__
  * - `Butterfly Image Classification <https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification>`__
    - 2D
    - `butterfly_data.zip <https://drive.google.com/file/d/1m4_3UAgUsZ8FDjB4HyfA50Sht7_XkfdB/view?usp=drive_link>`__


Minimal configuration
~~~~~~~~~~~~~~~~~~~~~
Apart from the input and output folders, there are a few basic parameters that always need to be specified in order to run an image classification workflow in BiaPy. **These parameters can be introduced either directly in the GUI, the code-free notebooks or by editing the YAML configuration file**.

Experiment name
***************
Also known as "model name" or "job name", this will be the name of the current experiment you want to run, so it can be differenciated from other past and future experiments.

.. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Run Workflow*, type the name you want for the job in the **Job name** field:

        .. image:: ../img/classification/GUI-run-workflow.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        In either the 2D or the 3D image classification notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **model_name**:
        
        .. image:: ../img/classification/Notebooks-model-name-data-conf.png
          :align: center
          :width: 75%

      .. tab:: Command line
        
        When calling BiaPy from command line, you can specify the output folder with the ``--name`` flag. See the *Command line* configuration of :ref:`classification_data_run` for a full example.


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

          Under *Workflow*, select *Image classification*, click twice on *Continue*, and under *Advanced options* > *Validation data*, select "Extract from train (split training)" in **Validation type**, and introduce your value (between 0 and 1) in the **Train prop. for validation**:

          .. image:: ../img/GUI-validation-percentage.png
            :align: center

        .. tab:: Google Colab / Notebooks
          
          In either the 2D or the 3D image classification notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **percentage_validation** with a value between 0 and 100:
          
          .. image:: ../img/classification/Notebooks-model-name-data-conf.png
            :align: center
            :width: 75%

        .. tab:: YAML configuration file
        
          Edit the variable ``DATA.VAL.SPLIT_TRAIN`` with a value between 0 and 1, representing the proportion of the training set that will be set apart for validation.


* **Validation path**: Similar to the training set, you can select a folder that contains the unprocessed (single-channel or multi-channel) raw images that will be used to validate the current model during training. As it happened with the training images, **all images of the same category are expected to be in the same sub-folder**.

  .. collapse:: Expand to see how to configure

    .. tabs::
      .. tab:: GUI

        Under *Workflow*, select *Image classification*, click twice on *Continue*, and under *Advanced options* > *Validation data*, select "Not extracted from train (path needed)" in **Validation type**, click on the *Browse* button of **Input raw image folder** and select the folder containing your validation raw images:

        .. image:: ../img/classification/GUI-validation-paths.png
          :align: center

      .. tab:: Google Colab / Notebooks
        
        This option is currently not available in the notebooks.

      .. tab:: YAML configuration file
      
        Edit the variable ``DATA.VAL.PATH`` with the absolute path to your validation raw images.

 
Test ground-truth
"""""""""""""""""
Do you have labels (classes) for the test set? This is a key question so BiaPy knows if your test set will be used for evaluation in new data (unseen during training) or simply produce predictions on that new data. All supervised workflows contain a parameter to specify this aspect.

.. collapse:: Expand to see how to configure

  .. tabs::
    .. tab:: GUI

      Under *Workflow*, select *Image Classification*, three times *Continue*, under *General options* > *Test data*, select "No" or "Yes" in the **Is the test separated in classes?** field:

      .. image:: ../img/classification/GUI-test-data.png
        :align: center

    .. tab:: Google Colab / Notebooks
      
      In either the 2D or the 3D image classification notebook, go to *Configure and train the DNN model* > *Select your parameters*, and check or uncheck the **test_ground_truth** option:
      
      .. image:: ../img/classification/Notebooks-model-name-data-conf.png
        :align: center
        :width: 75%


    .. tab:: YAML configuration file
      
      Set the variable ``DATA.TEST.LOAD_GT`` to ``True`` if you do have labels for your test images, or ``False`` otherwise.


\


Basic training parameters
*************************
At the core of each BiaPy workflow there is a deep learning model. Although we try to simplify the number of parameters to tune, these are the basic parameters that need to be defined for training an image classification workflow:

* **Number of classes**: The number of classes present in the problem. It must be equal to the number of subfolders in the training folder.

  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image classification*, click twice on *Continue*, and under *Workflow specific options* > *Extra options*, and edit the field **Number of classes**:

            .. image:: ../img/classification/GUI-workflow-specific-options.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D image classification notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **number_of_classes**:
            
            .. image:: ../img/classification/Notebooks-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the variable ``MODEL.N_CLASSES`` with the number of classes.

* **Number of input channels**: The number of channels of your raw images (grayscale = 1, RGB = 3). Notice the dimensionality of your images (2D/3D) is set by default depending on the workflow template you select.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image classification*, click once on *Continue*, and under *General options*, edit the last value of the field **Patch size** with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D:

            .. image:: ../img/classification/GUI-general-options.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D image classification notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **input_channels**:
            
            .. image:: ../img/classification/Notebooks-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``DATA.PATCH_SIZE`` with the number of channels. This variable follows a ``(y, x, channels)`` notation in 2D and a ``(z, y, x, channels)`` notation in 3D.

* **Number of epochs**: This number indicates how many `rounds <https://machine-learning.paperspace.com/wiki/epoch>`_ the network will be trained. On each round, the network usually sees the full training set. The value of this parameter depends on the size and complexity of each dataset. You can start with something like 100 epochs and tune it depending on how fast the loss (error) is reduced.
  
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image classification*, click twice on *Continue*, and under *Advanced options*, scroll down to *General training parameters*, and edit the field **Number of epochs**:

            .. image:: ../img/classification/GUI-basic-training-params.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D image classification notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **number_of_epochs**:
            
            .. image:: ../img/classification/Notebooks-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.EPOCHS`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.

* **Patience**: This is a number that indicates how many epochs you want to wait without the model improving its results in the validation set to stop training. Again, this value depends on the data you're working on, but you can start with something like 20.
   
  .. collapse:: Expand to see how to configure

        .. tabs::
          .. tab:: GUI

            Under *Workflow*, select *Image classification*, click twice on *Continue*, and under *Advanced options*, scroll down to *General training parameters*, and edit the field **Patience**:

            .. image:: ../img/classification/GUI-basic-training-params.png
              :align: center

          .. tab:: Google Colab / Notebooks
            
            In either the 2D or the 3D image classification notebook, go to *Configure and train the DNN model* > *Select your parameters*, and edit the field **patience**:
            
            .. image:: ../img/classification/Notebooks-basic-training-params.png
              :align: center
              :width: 75%

          .. tab:: YAML configuration file
          
            Edit the last value of the variable ``TRAIN.PATIENCE`` with the number of epochs. For this to have effect, the variable ``TRAIN.ENABLE`` should also be set to ``True``.


For improving performance, other advanced parameters can be optimized, for example, the model's architecture. The architecture assigned as default is the ViT, as it is effective in image classification tasks. This architecture allows a strong baseline, but further exploration could potentially lead to better results.

.. note:: Once the parameters are correctly assigned, the training phase can be executed. Note that to train large models effectively the use of a GPU (Graphics Processing Unit) is essential. This hardware accelerator performs parallel computations and has larger RAM memory compared to the CPUs, which enables faster training times.


.. _classification_data_run:

How to run
~~~~~~~~~~
BiaPy offers different options to run workflows depending on your degree of computer expertise. Select whichever is more approppriate for you:

.. tabs::
   .. tab:: GUI

        In the BiaPy GUI, navigate to *Workflow*, then select *Image classification* and follow the on-screen instructions:

        .. image:: ../img/gui/biapy_gui_classification.png
            :align: center
        
        \

        .. note:: BiaPy's GUI requires that all data and configuration files reside on the same machine where the GUI is being executed.

        .. tip:: If you need additional help, watch BiaPy's `GUI walkthrough video <https://www.youtube.com/embed/vY7aBh5FUNk?si=yvVolBnu5APNeHwB>`__.


   .. tab:: Google Colab 

        BiaPy offers two code-free notebooks in Google Colab to perform image classification: 

        .. |class_2D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/classification/BiaPy_2D_Classification.ipynb

        * For 2D images: |class_2D_colablink|

        .. |class_3D_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/classification/BiaPy_3D_Classification.ipynb

        * For 3D images: |class_3D_colablink|
      
        \

        .. tip:: If you need additional help, watch BiaPy's `Notebook walkthrough video <https://youtu.be/KEqfio-EnYw>`__.

   .. tab:: Docker

        If you installed BiaPy via Docker, `open a terminal <../get_started/faq.html#opening-a-terminal>`__ as described in :ref:`installation`. Then, you can use the `2d_classification.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/classification/2d_classification.yaml>`__ template file (or your own file), and run the workflow as follows:

        .. code-block:: bash                                                                                                    

            # Configuration file
            job_cfg_file=/home/user/2d_classification.yaml
            # Path to the data directory
            data_dir=/home/user/data
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results
            # Just a name for the job
            job_name=classification
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
            Note that ``data_dir`` must contain the path ``DATA.*.PATH`` so the container can find it. For instance, if you want to only train in this example ``DATA.TRAIN.PATH`` could be ``/home/user/data/train/``. 

   .. tab:: Command line 

        `From a terminal <../get_started/faq.html#opening-a-terminal>`__, you can use the `2d_classification.yaml <https://github.com/BiaPyX/BiaPy/blob/master/templates/classification/2d_classification.yaml>`__ template file (or your own file), and run the workflow as follows:

        .. tabs::

          .. tab:: Linux (bash)
              
              .. code-block:: bash
                    
                  # Configuration file
                  job_cfg_file=/home/user/2d_classification.yaml       
                  # Where the experiment output directory should be created
                  result_dir=/home/user/exp_results  
                  # Just a name for the job
                  job_name=my_2d_classification      
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

          .. tab:: Windows (batch)
              
              .. code-block:: bat
                  
                  REM Configuration file
                  set job_cfg_file=C:\home\user\2d_classification.yaml       
                  REM Where the experiment output directory should be created
                  set result_dir=C:\home\user\exp_results  
                  REM Just a name for the job
                  setjob_name=my_2d_classification      
                  REM Number that should be increased when one need to run the same job multiple times (reproducibility)
                  set job_counter=1
                  REM Number of the GPU to run the job in (according to 'nvidia-smi' command)
                  set gpu_number=0                   

                  REM Load the environment
                  call conda activate BiaPy_env
                  
                  biapy ^
                      --config %job_cfg_file% ^
                      --result_dir %result_dir%  ^
                      --name %job_name%    ^
                      --run_id %job_counter%  ^
                      --gpu "%gpu_number%"  

              For multi-GPU training you can call BiaPy as follows:

              .. code-block:: bat
                  
                  REM First check where is your biapy command (you need it in the below command)
                  REM $ where biapy
                  REM > C:\home\user\anaconda3\envs\BiaPy_env\bin\biapy

                  set gpu_number="0, 1, 2"
                  python -u -m torch.distributed.run ^
                      --nproc_per_node=3 ^
                      C:\home\user\anaconda3\envs\BiaPy_env\bin\biapy ^
                      --config %job_cfg_file% ^
                      --result_dir %result_dir%  ^
                      --name %job_name%    ^
                      --run_id %job_counter%  ^
                      --gpu "%gpu_number%"  

              ``nproc_per_node`` needs to be equal to the number of GPUs you are using (e.g. ``gpu_number`` length).



.. _classification_problem_resolution:

Templates                                                                                                                 
~~~~~~~~~

In the `templates/classification <https://github.com/BiaPyX/BiaPy/tree/master/templates/classification>`__ folder of BiaPy, you can find a few YAML configuration templates for this workflow. 


[Advanced] Special workflow configuration 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: This section is recommended for experienced users only to improve the performance of their workflows. When in doubt, do not hesitate to check our `FAQ & Troubleshooting <../get_started/faq.html>`__ or open a question in the `image.sc discussion forum <our FAQ & Troubleshooting section>`_.

Advanced Parameters 
*******************
Many of the parameters of our workflows are set by default to values that work commonly well. However, it may be needed to tune them to improve the results of the workflow. For instance, you may modify the following parameters

* **Model architecture**: Select the architecture of the deep neural network used as backbone of the pipeline. ViT, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7 and simple CNN. Default value: ViT.
* **Batch size**: This parameter defines the number of patches seen in each training step. Reducing or increasing the batch size may slow or speed up your training, respectively, and can influence network performance. Common values are 4, 8, 16, etc.
* **Patch size**: Input the size of the patches use to train your model (length in pixels in X and Y). The value should be smaller or equal to the dimensions of the image. The default value is 256 in 2D, i.e. 256x256 pixels.
* **Optimizer**: Select the optimizer used to train your model. Options: ADAM, ADAMW, Stochastic Gradient Descent (SGD). ADAM usually converges faster, while ADAMW provides a balance between fast convergence and better handling of weight decay regularization. SGD is known for better generalization. Default value: ADAMW.
* **Initial learning rate**: Input the initial value to be used as learning rate. If you select ADAM as optimizer, this value should be around 10e-4. 
* **Learning rate scheduler**: Select to adjust the learning rate between epochs. The current options are "Reduce on plateau", "One cycle", "Warm-up cosine decay" or no scheduler.
* **Test time augmentation (TTA)**: Select to apply augmentation (flips and rotations) at test time. It usually provides more robust results but uses more time to produce each result. By default, no TTA is peformed.

Metrics
*******

During the inference phase the performance of the test data is measured using different metrics if test masks were provided (i.e. ground truth) and, consequently, ``DATA.TEST.LOAD_GT`` is ``True``. In the case of classification the **accuracy**, **precision**, **recall**, and **F1** are calculated. Apart from that, the `confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`__ is also printed.


.. _classification_results:

Results                                                                                                                 
~~~~~~~  

The main output of this workflow will be a file named ``predictions.csv`` that will contain the predicted image class:

.. figure:: ../img/classification/classification_csv_output.svg
    :align: center
    :width: 150

    Classification workflow output

All files are placed in ``results`` folder under ``--result_dir`` directory with the ``--name`` given. Following the example, you should see that the directory ``/home/user/exp_results/classification`` has been created. If the same experiment is run 5 times, varying ``--run_id`` argument only, you should find the following directory tree: 

.. collapse:: Expand directory tree 

    .. code-block:: bash
        
      my_2d_classification/
      ├── config_files
      │   └── 2d_classification.yaml                                                                                                           
      ├── checkpoints
      │   └── model_weights_classification_1.h5
      └── results
          ├── my_2d_classification_1
          ├── . . .
          └── my_2d_classification_5
              ├── predictions.csv
              ├── aug
              │   └── .tif files
              ├── charts
              │   ├── my_2d_classification_1_*.png
              │   └── my_2d_classification_1_loss.png
              ├── train_logs
              └── tensorboard

\

* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

  * ``2d_classification.yaml``: YAML configuration file used (it will be overwrited every time the code is run).

* ``checkpoints``, *optional*: directory where model's weights are stored. Only created when ``TRAIN.ENABLE`` is ``True`` and the model is trained for at least one epoch. 

  * ``model_weights_my_2d_classification_1.h5``, *optional*: checkpoint file (best in validation) where the model's weights are stored among other information. Only created when the model is trained for at least one epoch. 
  
  * ``normalization_mean_value.npy``, *optional*: normalization mean value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.
  
  * ``normalization_std_value.npy``, *optional*: normalization std value. Is saved to not calculate it everytime and to use it in inference. Only created if ``DATA.NORMALIZATION.TYPE`` is ``custom``.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

  * ``my_2d_classification_1``: run 1 experiment folder. Can contain:

    * ``predictions.csv``: list of assigned class per test image.

    * ``aug``, *optional*: image augmentation samples. Only created if ``AUGMENTOR.AUG_SAMPLES`` is ``True``.

    * ``charts``, *optional*. Only created when ``TRAIN.ENABLE`` is ``True`` and epochs trained are more or equal ``LOG.CHART_CREATION_FREQ``:  

      * ``my_2d_classification_1_*.png``: plot of each metric used during training. 

      * ``my_2d_classification_1_loss.png``: loss over epochs plot. 

    * ``train_logs``: each row represents a summary of each epoch stats. Only avaialable if training was done.

    * ``tensorboard``: tensorboard logs.

.. note:: 

  Here, for visualization purposes, only ``my_2d_classification_1`` has been described but ``my_2d_classification_2``, ``my_2d_classification_3``, ``my_2d_classification_4`` and ``my_2d_classification_5`` directories will follow the same structure.



