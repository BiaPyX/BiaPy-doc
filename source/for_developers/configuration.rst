YAML configuration
------------------
In order to use BiaPy, a plain text YAML configuration file must be created using `YACS <https://github.com/rbgirshick/yacs>`_. This configuration file includes information about the hardware to be used, such as the number of CPUs or GPUs, the specific task or workflow, the model to be used, optional hyperparameters, the optimizer, and the paths for loading and storing data.

As an example, a full pipeline for semantic segmentation can be created using this configuration file. This file would include information on the specific model to be used, any necessary hyperparameters, the optimizer to be used during training, and the paths for loading and storing data. This configuration file is an essential component of BiaPy and is used to streamline the execution of the pipeline and ensure reproducibility of the results.

.. code-block:: yaml

  PROBLEM:
    TYPE: SEMANTIC SEG
    NDIM: 2D
  DATA:
    PATCH_SIZE: (256, 256, 1)
    TRAIN:
      PATH: /TRAIN_PATH
      GT_PATH: /TRAIN_GT_PATH
    VAL:
      SPLIT_TRAIN: 0.1
    TEST:
      PATH: /TEST_PATH
  AUGMENTOR:
    ENABLE: True
    RANDOM_ROT: True
  MODEL:
    ARCHITECTURE: unet
  TRAIN:
    OPTIMIZER: ADAMW 
    LR: 1.E−4
    BATCH_SIZE: 6
    EPOCHS: 360
  TEST:
    POST_PROCESSING:
      YZ_FILTERING: True
            

In order to run BiaPy, a YAML configuration file must be created. Examples for each workflow can be found in the `templates <https://github.com/BiaPyX/BiaPy/tree/master/templates>`__ folder on the BiaPy GitHub repository. If you are unsure about which workflow is most suitable for your data, you can refer to the `Select Workflow <select_workflow.html>`__ page for guidance.

The options for the configuration file can be found in the `config.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/config/config.py>`_ file on the BiaPy GitHub repository. However, some of the most commonly used options are explained below:

System
~~~~~~

To limit the number of CPUs used by the program, use the ``SYSTEM.NUM_WORKERS`` option. 

Problem specification
~~~~~~~~~~~~~~~~~~~~~

To specify the type of workflow, use the ``PROBLEM.TYPE`` option and select one of the following options: ``SEMANTIC_SEG``, ``INSTANCE_SEG``, ``DETECTION``, ``DENOISING``, ``SUPER_RESOLUTION``, ``SELF_SUPERVISED``, or ``CLASSIFICATION``.

To specify whether the data is 2D or 3D, use the ``PROBLEM.NDIM`` option and select either 2D or 3D.

.. _data_management:

Data management
~~~~~~~~~~~~~~~

The ``DATA.PATCH_SIZE`` variable is used to specify the shape of the images that will be used in the workflow. The order of the dimensions for 2D images is ``(y,x,c)`` and for 3D images it is ``(z,y,x,c)``. To ensure all images have a minimum size of ``DATA.PATCH_SIZE`` you can use ``DATA.REFLECT_TO_COMPLETE_SHAPE`` to ``True`` and those images smaller in any dimension will be padded with reflect. 

.. tabs::

  .. tab:: Train data

        The paths for the training data are set using the ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` variables. 

        There are two ways to work with the training data:

        * In the default setting, each image is divided into patches of size ``DATA.PATCH_SIZE`` using ``DATA.TRAIN.OVERLAP`` and ``DATA.TRAIN.PADDING``. By default, the minimum overlap is used, and the patches will always cover the entire image. On each epoch all these patches are visited. 

        * A random patch (of ``DATA.PATCH_SIZE`` size) from each image can be extracted if ``DATA.EXTRACT_RANDOM_PATCH`` is ``True``. This way, each epoch will only visit a patch within each training image, so it will be faster (but the amount of data seen by the network will be reduced too).

        The training data can be loaded into memory using ``DATA.TRAIN.IN_MEMORY`` to ``True``. In general, loading the data in memory is the fastest approach, but it relies on having enough memory available on the computer.

  .. tab:: Validation data
        
        There are two options to create the validation data:
          
        * Extract validation data from the training, where ``DATA.VAL.FROM_TRAIN`` must be set to ``True``. There are two options for doing it:
        
        .. tabs::

          .. tab:: Percentage split

                Create validation from the training data by setting ``DATA.VAL.SPLIT_TRAIN`` to the desired percentage of the training data to be used as validation.
          
          .. tab:: Cross validation      
                
                `Cross validation strategy <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#:~:text=Cross%2Dvalidation%20includes%20resampling%20and,model%20will%20perform%20in%20practice.>`__ by setting ``DATA.VAL.CROSS_VAL`` to ``True``. Use ``DATA.VAL.CROSS_VAL_NFOLD`` to set the number of folds and ``DATA.VAL.CROSS_VAL`` to set the number of the fold to choose as validation.

        * Create it by setting ``DATA.VAL.PATH`` and ``DATA.VAL.GT_PATH`` so the images and target can be read from the defined folders. For this, ``DATA.VAL.FROM_TRAIN`` must be set to ``False``. In this settings, as with the training data, two options are available:

          * Each image is divided into patches of size ``DATA.PATCH_SIZE`` using ``DATA.VAL.OVERLAP`` and ``DATA.VAL.PADDING``. By default, the minimum overlap is used, and the patches will always cover the entire image. On each epoch all these patches are visited. 

          * A fixed patch, starting from the origin (0,0), of ``DATA.PATCH_SIZE`` size will be extracted from each image if ``DATA.EXTRACT_RANDOM_PATCH`` is ``True``. This way, each epoch will only visit a patch within each validation image, so it will be faster (but the amount of data seen by the network will be reduced too). 

        The validation data can be loaded into memory using ``DATA.VAL.IN_MEMORY`` to ``True``. In general, loading the data in memory is the fastest approach, but it relies on having enough memory available on the computer.

  .. tab:: Test data
        
        The paths for the test data are set using the ``DATA.TEST.PATH`` and ``DATA.TEST.GT_PATH`` variables. If this last is present and ``DATA.TEST.LOAD_GT`` is ``True`` the model prediction will be compared with this target/ground truth and some metrics calculated to evaluate the performance of the model. 

        For more information regarding the test data management go to :ref:`config_test`.

Data filtering
**************

For all data types (training, validation, and test), the parameters ``DATA.TRAIN.FILTER_SAMPLES``, ``DATA.VAL.FILTER_SAMPLES``, and ``DATA.TEST.FILTER_SAMPLES`` can be used to specify which samples should be included. In each case, the option ``DATA.*.FILTER_SAMPLES.ENABLE`` must be set to ``True``. After enabling, you need to configure ``DATA.*.FILTER_SAMPLES.PROPS``, ``DATA.*.FILTER_SAMPLES.VALUES``, and ``DATA.*.FILTER_SAMPLES.SIGNS`` to define the filtering criteria. 

With ``DATA.*.FILTER_SAMPLES.PROPS``, we define the property to look at to establish the condition. Currently, the available properties for filtering are: 

* ``'foreground'`` is defined as the percentage of pixels/voxels corresponding to the foreground mask. This option is only valid for ``SEMANTIC_SEG``, ``INSTANCE_SEG`` and ``DETECTION``.

* ``'mean'`` is defined as the mean intensity value of the raw image inputs.

* ``'min'`` is defined as the min intensity value of the raw image inputs.

* '``max'`` is defined as the max intensity value of the raw image inputs.

* ``'diff'`` is defined as the difference between ground truth and raw images. Available for all workflows but ``SELF_SUPERVISED`` and ``DENOISING``. 

* ``'diff_by_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the ratio between raw image max and min. Available for all workflows but ``SELF_SUPERVISED`` and ``DENOISING``. 

* ``'target_mean'`` is defined as the mean intensity value of the raw image targets. Available for all workflows but ``SELF_SUPERVISED`` and ``DENOISING``.

* ``'target_min'`` is defined as the min intensity value of the raw image targets. Available for all workflows but ``SELF_SUPERVISED`` and ``DENOISING``. 

* ``'target_max'`` is defined as the max intensity value of the raw image targets. Available for all workflows but ``SELF_SUPERVISED`` and ``DENOISING``.  

* ``'diff_by_target_min_max_ratio'`` is defined as the difference between ground truth and raw images multiplied by the ratio between ground truth image max and min. Available for all workflows but ``SELF_SUPERVISED`` and ``DENOISING``. 

With ``DATA.*.FILTER_SAMPLES.VALUES`` and ``DATA.*.FILTER_SAMPLES.SIGNS``, we define the specific values and the comparison operators of each property, respectively. The available operators are: ``'gt'``, ``'ge'``, ``'lt'`` and ``'le'``, that corresponds to "greather than" (or ">"), "greather equal" (or ">="), "less than" (or "<"), and "less equal"  (or "<=").
          
For example, if you want to remove those samples that have intensity values lower than ``0.00001`` and a mean average greater than ``100`` you should declare the above three variables as follows (notice you need to know the image data type in advance):

.. code-block:: bash

  DATA.TRAIN.FILTER_SAMPLES.PROPS = [['foreground','mean']]
  DATA.TRAIN.FILTER_SAMPLES.VALUES = [[0.00001, 100]]
  DATA.TRAIN.FILTER_SAMPLES.SIGNS = [['lt', 'gt']]

You can also concatenate more restrictions and they will be applied in order. For instance, if you want to filter those
samples with a maximum intensity value greater than ``1000``, and do that before the condition described above, you can define the
variables this way:

.. code-block:: bash

  DATA.TRAIN.FILTER_SAMPLES.PROPS = [['max'], ['foreground','mean']]
  DATA.TRAIN.FILTER_SAMPLES.VALUES = [[1000], [0.00001, 100]]
  DATA.TRAIN.FILTER_SAMPLES.SIGNS = [['gt'], ['lt', 'gt']]

The ``DATA.FILTER_BY_IMAGE`` parameter determines how the filtering is applied: if set to ``True``, the entire image is processed (this is always the case if ``DATA.EXTRACT_RANDOM_PATCH`` is ``True``); if set to ``False``, the filtering is performed on a patch-by-patch basis.

.. seealso::

    For test data, even if ``DATA.FILTER_BY_IMAGE`` is set to ``False``, indicating that filtering will be applied on a patch-by-patch basis, no patches are discarded to ensure the complete image can be reconstructed. These patches are flagged and are not processed by the model, resulting in a black patch prediction.

.. seealso::

    You can use ``DATA.TRAIN.FILTER_SAMPLES.NORM_BEFORE`` to control whether the normalization is applied before the filtering, which can help you deciding the values for the filtering. 

.. _data_norm:

Data normalization
~~~~~~~~~~~~~~~~~~

Previous to normalization, you can choose to do a percentile clipping to remove outliers (by setting ``DATA.NORMALIZATION.PERC_CLIP`` to ``True``). Lower and upper bound for percentile clip are set with  ``DATA.NORMALIZATION.PERC_LOWER`` and ``DATA.NORMALIZATION.PERC_UPPER`` respectively. 

The data normalization type is controlled by ``DATA.NORMALIZATION.TYPE`` and a few options are available:

* ``'div'`` (default): normalizes the data to ``[0-1]`` range. The division is done using the maximum value of the data type. i.e. 255 for uint8 or 65535 if uint16.
* ``'zero_mean_unit_variance'``: normalization substracting the mean and divide by std. The mean and std can be specified with ``DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL`` and ``DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL`` respectively.
* ``'scale_range'``: normalizes the data to ``[0-1]`` range but, instead of dividing by the maximum value of the data type as in ``'div'``, it divides by the maximum value of each image.

The normalization or clipping values can be derived either from the entire image or from individual patches. This behavior is controlled by the variable ``DATA.NORMALIZATION.MEASURE_BY``, which accepts either ``'image'`` or ``'patch'`` as its value.

Pre-processing
~~~~~~~~~~~~~~

There are a few pre-processing functions  (controlled by ``DATA.PREPROCESS``) that can be applied to the train (``DATA.PREPROCESS.TRAIN``), validation (``DATA.PREPROCESS.VAL``) or test data (``DATA.PREPROCESS.TEST``). So they can be applied the images need to be loaded in memory (``DATA.*.IN_MEMORY`` to ``True``). The pre-processing is done right after loading the images, when no normalization has been done yet. These is the list of available functions:

* **Resize** (controlled by ``DATA.PREPROCESS.RESIZE``): to resize images to the desired shape. 

* **Gaussian blur** (controlled by ``DATA.PREPROCESS.GAUSSIAN_BLUR``): to add gaussian blur.

* **Median blur** (controlled by ``DATA.PREPROCESS.MEDIAN_BLUR``): to add median blur.

* **CLAHE** (controlled by ``DATA.PREPROCESS.CLAHE``): to apply a `contrast limited adaptive histogram equalization <https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE>`__.

* **Canny** (controlled by ``DATA.PREPROCESS.CANNY``): to apply `Canny <https://en.wikipedia.org/wiki/Canny_edge_detector>`__ or edge detection (only for 2D images, grayscale or RGB).

Check out our pre-processing notebook showcasing all these transformations that can be applied to the data: |preprocessing_notebook_colablink|

.. |preprocessing_notebook_colablink| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/master/notebooks/Data_Preprocessing.ipynb
    
Data augmentation
~~~~~~~~~~~~~~~~~

The ``AUGMENTOR.ENABLE`` variable must be set to ``True`` to enable data augmentation (DA). The probability of each transformation is set using the ``AUGMENTOR.DA_PROB`` variable. BiaPy offers a wide range of transformations, which can be found in the `config.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/config/config.py>`__ file in the BiaPy repository on GitHub.

Images generated using data augmentation will be saved in the ``PATHS.DA_SAMPLES`` directory (which is ``aug`` by default). This allows you to check the data augmentation applied to the images. If you want a more exhaustive check, you can save all the augmented training data by enabling ``DATA.CHECK_GENERATORS``. The images will be saved in ``PATHS.GEN_CHECKS`` and ``PATHS.GEN_MASK_CHECKS``. Be aware that this option can consume a large amount of disk space as the training data will be entirely copied.

Model definition
~~~~~~~~~~~~~~~~
BiaPy offers three different backends to be used to choose a model (controlled by ``MODEL.SOURCE``):


- ``biapy``, which uses BiaPy as the backend for the model definition. Use ``MODEL.ARCHITECTURE`` to select the model. Different models for each workflow are implemented:

  * Semantic segmentation: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet``, ``resunet_se``, ``unext_v1``, ``multiresunet`` and ``unetr``. 

  * Instance segmentation: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet``, ``resunet_se``, ``unext_v1``, ``multiresunet`` and ``unetr``. 

  * Detection: ``unet``, ``resunet``, ``resunet++``, ``attention_unet`` and ``seunet``.

  * Denoising: ``unet``, ``resunet``, ``resunet++``, ``attention_unet`` and ``seunet``.

  * Super-resolution: ``edsr``, ``rcan``, ``dfcan``, ``wdsr``, ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet`` and ``multiresunet``. 

  * Self-supervision: ``edsr``, ``rcan``, ``dfcan``, ``wdsr``, ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet``, ``resunet_se``, ``unext_v1``, ``multiresunet``, ``unetr``, ``vit`` and ``mae``.

  * Classification: ``simple_cnn``, ``efficientnet_b0``, ``efficientnet_b1``, ``efficientnet_b2``, ``efficientnet_b3``, ``efficientnet_b4``, ``efficientnet_b5``, ``efficientnet_b6``, ``efficientnet_b7``, ``vit``. 

  * Image to image: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet``, ``resunet_se``, ``unext_v1``, ``multiresunet``,  ``unetr``, ``edsr``, ``rcan``, ``dfcan``, ``wdsr``, ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet`` and ``multiresunet``. 

  For ``unet``, ``resunet``, ``resunet++``, ``resunet_se``, ``attention_unet`` and ``seunet`` architectures you can set ``MODEL.FEATURE_MAPS`` to determine the feature maps to use on each network level. In the same way, ``MODEL.DROPOUT_VALUES`` can be set for each level in those networks. For ``unetr`` and ``vit`` networks only the first value of those variables will be taken into account.

  The ``MODEL.BATCH_NORMALIZATION`` variable can be used to enable batch normalization on the ``unet``, ``resunet``, ``resunet++``, ``resunet_se``, ``attention_unet``, ``seunet`` and ``unetr`` models. For the 3D versions of these networks (except for ``unetr``), the ``MODEL.Z_DOWN`` option can also be used to avoid downsampling in the z-axis, which is typically beneficial for anisotropic data.

  The ``MODEL.N_CLASSES`` variable can be used to specify the number of classes for the classification problem, excluding the background class (labeled as ``0``). If the number of classes is set to ``1`` or ``2``, the problem is considered binary, and the behavior is the same. For more than ``2`` classes, the problem is considered multi-class, and the output of the models will have the corresponding number of channels.

  Finally, the ``MODEL.LOAD_CHECKPOINT`` variable can be used to load a pre-trained checkpoint of the network (for finetunning).  

- ``torchvision``, which uses models defined in `TorchVision <https://pytorch.org/vision/stable/models.html>`__. Use ``MODEL.TORCHVISION_MODEL_NAME`` to select the model. All the models load pretrained their corresponding default weights. Currently, BiaPy supports the following  models for each workflow: 

  * Semantic segmentation (defined `here <https://pytorch.org/vision/stable/models.html#semantic-segmentation>`__): ``deeplabv3_mobilenet_v3_large``, ``deeplabv3_resnet101``, ``deeplabv3_resnet50``, ``fcn_resnet101``, ``fcn_resnet50`` and ``lraspp_mobilenet_v3_large``. 

  * Instance segmentation (defined `here <https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection>`__): ``maskrcnn_resnet50_fpn`` and ``maskrcnn_resnet50_fpn_v2``. 

  * Detection (defined `here <https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection>`__): ``fasterrcnn_mobilenet_v3_large_320_fpn``, ``fasterrcnn_mobilenet_v3_large_fpn``, ``fasterrcnn_resnet50_fpn``, ``fasterrcnn_resnet50_fpn_v2``, ``fcos_resnet50_fpn``, ``ssd300_vgg16``, ``ssdlite320_mobilenet_v3_large``, ``retinanet_resnet50_fpn``, ``retinanet_resnet50_fpn_v2``.

  * Denoising: no model available with TorchVision backend.

  * Super-resolution: no model available with TorchVision backend. 

  * Self-supervision: no model available with TorchVision backend.

  * Classification (defined `here <https://pytorch.org/vision/stable/models.html#classification>`__): ``alexnet``, ``convnext_base``, ``convnext_large``, ``convnext_small``, ``convnext_tiny``, ``densenet121``, ``densenet161``, ``densenet169``, ``densenet201``, ``efficientnet_b0``, ``efficientnet_b1``, ``efficientnet_b2``, ``efficientnet_b3``, ``efficientnet_b4``, ``efficientnet_b5``, ``efficientnet_b6``, ``efficientnet_b7``, ``efficientnet_v2_l``, ``efficientnet_v2_m``, ``efficientnet_v2_s``, ``googlenet``, ``inception_v3``, ``maxvit_t``, ``mnasnet0_5``, ``mnasnet0_75``, ``mnasnet1_0``, ``mnasnet1_3``, ``mobilenet_v2``, ``mobilenet_v3_large``, ``mobilenet_v3_small``, ``quantized_googlenet``, ``quantized_inception_v3``, ``quantized_mobilenet_v2``, ``quantized_mobilenet_v3_large``, ``quantized_resnet18``, ``quantized_resnet50``, ``quantized_resnext101_32x8d``, ``quantized_resnext101_64x4d``, ``quantized_shufflenet_v2_x0_5``, ``quantized_shufflenet_v2_x1_0``, ``quantized_shufflenet_v2_x1_5``, ``quantized_shufflenet_v2_x2_0``, ``regnet_x_16gf``, ``regnet_x_1_6gf``, ``regnet_x_32gf``, ``regnet_x_3_2gf``, ``regnet_x_400mf``, ``regnet_x_800mf``, ``regnet_x_8gf``, ``regnet_y_128gf``, ``regnet_y_16gf``, ``regnet_y_1_6gf``, ``regnet_y_32gf``, ``regnet_y_3_2gf``, ``regnet_y_400mf``, ``regnet_y_800mf``, ``regnet_y_8gf``, ``resnet101``, ``resnet152``, ``resnet18``, ``resnet34``, ``resnet50``, ``resnext101_32x8d``, ``resnext101_64x4d``, ``resnext50_32x4d``, ``retinanet_resnet50_fpn``, ``shufflenet_v2_x0_5``, ``shufflenet_v2_x1_0``, ``shufflenet_v2_x1_5``, ``shufflenet_v2_x2_0``, ``squeezenet1_0``, ``squeezenet1_1``, ``swin_b``, ``swin_s``, ``swin_t``, ``swin_v2_b``, ``swin_v2_s``, ``swin_v2_t``, ``vgg11``, ``vgg11_bn``, ``vgg13``, ``vgg13_bn``, ``vgg16``, ``vgg16_bn``, ``vgg19``, ``vit_b_16``, ``vit_b_32``, ``vit_h_14``, ``vit_l_16``, ``vit_l_32``, ``wide_resnet101_2`` and ``wide_resnet50_2``.

- ``bmz``, which uses `Bioimage Model Zoo (bioimage.io) <https://bioimage.io/#/>`__ pretrained models. Use ``MODEL.BMZ.SOURCE_MODEL_ID`` to select the model. More a more models are added to the zoo so please check `Bioimage Model Zoo page <https://bioimage.io/#/>`__ to see available models. BiaPy can only consume models exported with `Pytorch state dict <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html#:~:text=A%20state_dict%20is%20an%20integral,to%20PyTorch%20models%20and%20optimizers.>`__. 

Training phase
~~~~~~~~~~~~~~

To activate the training phase, set the ``TRAIN.ENABLE`` variable to ``True``. The ``TRAIN.OPTIMIZER`` variable can be set to either ``SGD``, ``ADAM`` or ``ADAMW``, and the learning rate can be set using the ``TRAIN.LR`` variable. If you do not have much expertise in choosing these settings, you can use ``ADAMW`` and ``1.E-4`` as a starting point. It is also possible to use a learning rate scheduler with ``TRAIN.LR_SCHEDULER`` variable.

Additionally, you need to specify how many images will be fed into the network at the same time using the ``TRAIN.BATCH_SIZE`` variable. For example, if you have ``100`` training samples and you select a batch size of ``6``, this means that ``17`` batches (``100/6 = 16.6``) are needed to input all the training data to the network, after which one epoch is completed.

To train the network, you need to specify the number of epochs using the ``TRAIN.EPOCHS`` variable. You can also set the patience using ``TRAIN.PATIENCE``, which will stop the training process if no improvement is made on the validation data for that number of epochs.

.. seealso::

    Set ``DATA.TRAIN.RESOLUTION`` to let the model know the resolution of training data. This information will be taken into account for some data augmentations.

Loss types 
~~~~~~~~~~

Different loss functions can be set depending on the workflow: 

* Semantic segmentation:

    * ``"CE"`` (default): `Cross entropy loss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`__. 
    * ``"DICE"``: `Dice loss <https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch>`__.
    * ``"W_CE_DICE"``: ``CE`` and ``Dice`` (with a weight term on each one that must sum ``1``). With ``LOSS.WEIGHTS`` the weights for each of the losses can be configured. `Reference link <https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch>`__.

* Instance segmentation: automatically set depending on the channels selected (``PROBLEM.INSTANCE_SEG.DATA_CHANNELS``). There is no need to set it.

* Detection:

    * ``"CE"`` (default): `Cross entropy loss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`__.
    * ``"DICE"``: `Dice loss <https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch>`__.
    * ``"W_CE_DICE"``: ``CE`` and ``Dice`` (with a weight term on each one that must sum ``1``). With ``LOSS.WEIGHTS`` the weights for each of the losses can be configured. `Reference link <https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch>`__. 

* Denoising:

    * ``"MSE"`` (default): `Mean Square Error <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`__. 

* Super-resolution:

    * ``"MAE"`` (default): `Mean Absolute Error (MAE) <https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss>`__. 
    * ``"MSE"``: `Mean Square Error (MSE) <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`__. 
    * ``"SSIM"``: `structural similarity index measure (SSIM) <https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html#torchmetrics.image.StructuralSimilarityIndexMeasure>`__.
    * ``"W_MAE_SSIM_loss"``: ``MAE`` and ``SSIM`` (with a weight term on each one that must sum ``1``). The weights are set with ``LOSS.WEIGHTS``. 
    * ``"W_MSE_SSIM_loss"``: ``MSE`` and ``SSIM`` (with a weight term on each one that must sum ``1``). The weights are set with ``LOSS.WEIGHTS``. 

* Self-supervision. These losses can only be set when ``PROBLEM.SELF_SUPERVISED.PRETEXT_TASK`` is ``"crappify"``. Otherwise it will be automatically set to ``"MSE"``, i.e when ``PROBLEM.SELF_SUPERVISED.PRETEXT_TASK`` is ``"masking"``. The options are:

    * ``"MAE"`` (default): `Mean Absolute Error (MAE) <https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss>`__.
    * ``"MSE"``: `Mean Square Error (MSE) <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`__. 
    * ``"SSIM"``: `structural similarity index measure (SSIM) <https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html#torchmetrics.image.StructuralSimilarityIndexMeasure>`__.
    * ``"W_MAE_SSIM_loss"``: ``MAE`` and ``SSIM`` (with a weight term on each one that must sum ``1``). The weights are set with ``LOSS.WEIGHTS``. 
    * ``"W_MSE_SSIM_loss"``: ``MSE`` and ``SSIM`` (with a weight term on each one that must sum ``1``). The weights are set with ``LOSS.WEIGHTS``. 

* Classification:

    * ``"CE"`` (default): `Cross entropy loss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`__.

* Image to image:

    * ``"MAE"`` (default): `Mean Absolute Error (MAE) <https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss>`__.
    * ``"MSE"``: `Mean Square Error (MSE) <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`__. 
    * ``"SSIM"``: `structural similarity index measure (SSIM) <https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html#torchmetrics.image.StructuralSimilarityIndexMeasure>`__.
    * ``"W_MAE_SSIM_loss"``: ``MAE`` and ``SSIM`` (with a weight term on each one that must sum ``1``). The weights are set with ``LOSS.WEIGHTS``. 
    * ``"W_MSE_SSIM_loss"``: ``MSE`` and ``SSIM`` (with a weight term on each one that must sum ``1``). The weights are set with ``LOSS.WEIGHTS``. 

``LOSS.CLASS_REBALANCE`` can be used to adjust the loss function based on the imbalance between classes. This can be used when ``LOSS.TYPE`` is ``"CE"`` detection and semantic segmentation, or if using ``'B'``, ``'C'``, ``'M'``, ``'P'`` or ``'A'`` channels in instance segmentation workflow, as those are are binary channels.

.. _config_test:

Test phase
~~~~~~~~~~

To initiate the testing phase, also referred to as inference or prediction, one must set the variable ``TEST.ENABLE`` to ``True`` within the BiaPy framework. BiaPy provides two distinct prediction options contingent upon the dimensions of the test images to be predicted. It is essential to consider that not only must the test image fit into memory, but also the model's prediction, characterized by a data type of ``float32`` (or ``float16`` if ``TEST.REDUCE_MEMORY`` is activated). Moreover, if the test image cannot be accommodated within the GPU memory, a cropping procedure becomes necessary. Typically, this entails cropping into patches with overlap and/or padding to circumvent border effects during the reconstruction of the original shape, albeit at the expense of increased memory usage. Given these considerations, two alternative procedures are available for predicting a test image:

- When each test image can be fit in memory the procedure is the following. In this scenario there are two options:
  
  - First option, and the default, is where each test image is divided into patches of size ``DATA.PATCH_SIZE`` and passed through the network individually. Then, the original image will be reconstructed. Apart from this, it will automatically calculate performance metrics per patch and per reconstructed image if the ground truth is available (enabled by ``DATA.TEST.LOAD_GT``).

  - Second option is to enable ``TEST.FULL_IMG``, to pass entire images through the model without cropping them. This option requires enough GPU memory to fit the images into, so to prevent possible errors it is only available for 2D images.

  In both options described above you can also use test-time augmentation by setting ``TEST.AUGMENTATION`` to ``True``, which will create multiple augmented copies of each patch, or image if ``TEST.FULL_IMG`` selected, by all possible rotations (``8`` copies in 2D and ``16`` in 3D). This will slow down the inference process, but it will return more robust predictions.

  You can use also use ``DATA.REFLECT_TO_COMPLETE_SHAPE`` to ensure that the patches can be made as pointed out in :ref:`data_management`. 

  .. seealso::

    If the test images are large and you experience memory issues during the testing phase, you can set the ``TEST.REDUCE_MEMORY`` variable to ``True``. This will reduce memory usage as much as possible, but it may slow down the inference process.

- When each test image can not fit into memory (scalable solution):

  BiaPy offers to use `H5 <https://docs.h5py.org/en/stable/#:~:text=HDF5%20lets%20you%20store%20huge,they%20were%20real%20NumPy%20arrays.>`__ or `Zarr <https://zarr.readthedocs.io/en/stable/>`__ files to generate predictions by configuring ``TEST.BY_CHUNKS`` variable. In this setting, ``TEST.BY_CHUNKS.FORMAT`` decides which files are you working with and ``DATA.TEST.INPUT_IMG_AXES_ORDER`` sets the axis order (all the test images need to be order in the same way). This way, BiaPy enables multi-GPU processing per image by chunking large images into patches with overlap and padding to mitigate artifacts at the edges. Each GPU processes a chunk of the large image, storing the patch in its designated location using Zarr or H5 file formats. This is possible because these file formats facilitate reading and storing data chunks without requiring the entire file to be loaded into memory. Consequently, our approach allows the generation of predictions for large images, overcoming potential memory bottlenecks.
  
  .. warning::

    There is also an option to generate a TIFF file from the predictions with ``TEST.BY_CHUNKS.SAVE_OUT_TIF``. However, take into account that this option require to load the entire data into memory, which is sometimes not fleasible. 

  After the prediction is generated the variable ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.ENABLE`` controls whether the rest of the workflow process is going to be done or not (as may require large memory consumption depending on the workflow). If enabled, the prediction can be processed in two different ways (controlled by ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE``):

  - ``chunk_by_chunk`` : prediction will be processed by chunks, where each chunk will be considered as an individual image. Select this operation if you have not enough memory to process the entire prediction image with ``entire_pred``.
  - ``entire_pred``: the predicted image will be loaded in memory at once and processed entirely (be aware of your memory budget).

  The option ``chunk_by_chunk`` is not trivial depending on the workflow, e.g. in instance segmentation different instances on each chunk need to be merged into one. Three workflows need to post-process the predictions to have a final result, semantic segmentation, instance segmentation and detection. Currently, ``chunk_by_chunk`` is only supported in detection workflow. 

.. _config_metric:

Metric measurement
~~~~~~~~~~~~~~~~~~

You can configure the metrics to be measured during train and test with ``TRAIN.METRICS`` and ``TEST.METRICS`` variables, respectively. Each workflow have different type of metrics that can be configured. If empty, some default metrics will be configured automatically.

During training these ones can be applied (all of them on each case are set by default):

* Semantic segmentation: ``"iou"`` (called also Jaccard index).
* Instance segmentation: automatically set depending on the channels selected (``PROBLEM.INSTANCE_SEG.DATA_CHANNELS``).
* Detection: ``"iou"`` (called also Jaccard index).
* Denoising: ``"mae"``, ``"mse"``.
* Super-resolution: ``"psnr"``, ``"mae"``, ``"mse"``, ``"ssim"``.
* Self-supervision: ``"psnr"``, ``"mae"``, ``"mse"``, ``"ssim"``.
* Classification: ``'accuracy'``, ``"top-5-accuracy"``.
* Image to image: ``"psnr"``, ``"mae"``, ``"mse"``, ``"ssim"``.

During test these ones can be applied (all of them on each case are set by default):

* Semantic segmentation: ``"iou"`` (called also Jaccard index).
* Instance segmentation: automatically set depending on the channels selected (``PROBLEM.INSTANCE_SEG.DATA_CHANNELS``). Instance metrics will be always calculated.
* Detection: ``"iou"`` (called also Jaccard index).
* Denoising: ``"mae"``, ``"mse"``.
* Super-resolution: ``"psnr"``, ``"mae"``, ``"mse"``, ``"ssim"``. Additionally, if only if ``PROBLEM.NDIM`` is ``'2D'``, these can also be selected: ``"fid"``, ``"is"``, ``"lpips"``.
* Self-supervision: ``"psnr"``, ``"mae"``, ``"mse"``, ``"ssim"``. Additionally, if only if ``PROBLEM.NDIM`` is ``'2D'``, these can also be selected: ``"fid"``, ``"is"``, ``"lpips"``.
* Classification: ``'accuracy'``, ``"top-5-accuracy"``.
* Image to image: ``"psnr"``, ``"mae"``, ``"mse"``, ``"ssim"``. Additionally, if only if ``PROBLEM.NDIM`` is ``'2D'``, these can also be selected: ``"fid"``, ``"is"``, ``"lpips"``.


Post-processing
~~~~~~~~~~~~~~~

BiaPy is equipped with several post-processing methods that are primarily applied in two distinct stages:

1. After the network's prediction. These post-processing methods are common among workflows that return probabilities from their models, e.g. semantic/instance segmentation and detection. These post-processing methods aim to improve the resulting probabilities. Currently, these post-processing methods are only avaialable for 3D images (e.g. ``PROBLEM.NDIM`` is 3D or ``PROBLEM.NDIM`` is 2D but ``TEST.ANALIZE_2D_IMGS_AS_3D_STACK`` is ``True``):

  * ``TEST.POST_PROCESSING.APPLY_MASK``: a binary mask is applied to remove anything not contained within the mask. For this, the ``DATA.TEST.BINARY_MASKS`` path needs to be set.
  * ``TEST.POST_PROCESSING.MEDIAN_FILTER``: to apply a median filtering. This variable expects a list of median filters to apply. They are going to be applied in the list order. This can only be used in ``'SEMANTIC_SEG'``, ``'INSTANCE_SEG'`` and ``'DETECTION'`` workflows. There are multiple options to compose the list:

    * ``'xy'`` or ``'yx'``: to apply the filter in x and y axes together.
    * ``'zy'`` or ``'yz'``: to apply the filter in y and z axes together.
    * ``'zx'`` or ``'xz'``: to apply the filter in x and z axes together.
    * ``'z'``: to apply the filter only in z axis.

2.  After each workflow main process is done there is another post-processing step on some of the workflows to achieve the final results, i.e. workflow-specific post-processing methods. Find a full description of each method inside the workflow description:

  * Instance segmentation:

    * Big instance repair
    * Filter instances by morphological features

  * Detection:

    * Remove close points
    * Create instances from points