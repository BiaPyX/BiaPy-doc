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

The options for the configuration file can be found in the `config.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/config/config.py>`__ file on `the BiaPy GitHub repository <https://github.com/BiaPyX/BiaPy>`__. However, some of the most commonly used options are explained below:

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

.. code-block:: yaml

  DATA:
    TRAIN:
      FILTER_SAMPLES.PROPS: [['foreground','mean']]
      FILTER_SAMPLES.VALUES: [[0.00001, 100]]
      FILTER_SAMPLES.SIGNS: [['lt', 'gt']]

You can also concatenate more restrictions and they will be applied in order. For instance, if you want to filter those
samples with a maximum intensity value greater than ``1000``, and do that before the condition described above, you can define the
variables this way:

.. code-block:: yaml

  DATA:
    TRAIN:
      FILTER_SAMPLES.PROPS: [['max'], ['foreground','mean']]
      FILTER_SAMPLES.VALUES: [[1000], [0.00001, 100]]
      FILTER_SAMPLES.SIGNS: [['gt'], ['lt', 'gt']]

The ``DATA.FILTER_BY_IMAGE`` parameter determines how the filtering is applied: if set to ``True``, the entire image is processed (this is always the case if ``DATA.EXTRACT_RANDOM_PATCH`` is ``True``); if set to ``False``, the filtering is performed on a patch-by-patch basis.

.. seealso::

    For test data, even if ``DATA.FILTER_BY_IMAGE`` is set to ``False``, indicating that filtering will be applied on a patch-by-patch basis, no patches are discarded to ensure the complete image can be reconstructed. These patches are flagged and are not processed by the model, resulting in a black patch prediction.

.. seealso::

    You can use ``DATA.TRAIN.FILTER_SAMPLES.NORM_BEFORE`` to control whether the normalization is applied before the filtering, which can help you deciding the values for the filtering. 

.. _data_norm:

Data normalization
~~~~~~~~~~~~~~~~~~

Previous to normalization, you can choose to do a percentile clipping to remove outliers (by setting ``DATA.NORMALIZATION.PERC_CLIP.ENABLE`` to ``True``). Lower and upper bound for percentile clip are set with  ``DATA.NORMALIZATION.PERC_LOWER`` and ``DATA.NORMALIZATION.PERC_UPPER`` respectively. 

The data normalization type is controlled by ``DATA.NORMALIZATION.TYPE`` and a few options are available:

* ``'div'`` (default): normalizes the data to ``[0-1]`` range. The division is done using the maximum value of the data type. i.e. 255 for uint8 or 65535 if uint16.
* ``'zero_mean_unit_variance'``: normalization substracting the mean and divide by std. The mean and std can be specified with ``DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL`` and ``DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL`` respectively.
* ``'scale_range'``: normalizes the data to ``[0-1]`` range but, instead of dividing by the maximum value of the data type as in ``'div'``, it divides by the maximum value of each image.

The normalization or clipping values can be derived either from the entire image or from individual patches. This behavior is controlled by the variable ``DATA.NORMALIZATION.MEASURE_BY``, which accepts either ``'image'`` or ``'patch'`` as its value. A very common configuration for normalization can be as follows:

.. code-block:: yaml

  DATA:
    NORMALIZATION:
      TYPE: zero_mean_unit_variance
      PERC_CLIP:
        ENABLE: True
        LOWER: 0.1
        UPPER: 99.8

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

An example of a common data augmentation configuration is as follows:

.. code-block:: yaml

  AUGMENTOR:
    ENABLE: True
    AUG_SAMPLES: True
    RANDOM_ROT: True
    VFLIP: True
    HFLIP: True
    ZFLIP: True
    BRIGHTNESS: True
    BRIGHTNESS_FACTOR: (-0.2, 0.2)
    CONTRAST: True
    CONTRAST_FACTOR: (-0.2, 0.2)
    ELASTIC: True
    AFFINE_MODE: 'reflect'

Model definition
~~~~~~~~~~~~~~~~
BiaPy offers three different backends to be used to choose a model (controlled by ``MODEL.SOURCE``):

- ``biapy``, which uses BiaPy as the backend for the model definition. Use ``MODEL.ARCHITECTURE`` to select the model. Different models for each workflow are implemented:

  * Semantic segmentation: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``multiresunet``, ``seunet``, ``resunet_se``, ``unetr``, ``unext_v1``, ``unext_v2``, ``hrnet`` and ``stunet``.

  * Instance segmentation: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``multiresunet``, ``seunet``, ``resunet_se``, ``unetr``, ``unext_v1``, ``unext_v2``, ``hrnet`` and ``stunet``. 

  * Detection: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``multiresunet``, ``seunet``, ``resunet_se``, ``unetr``, ``unext_v1``, ``unext_v2``, ``hrnet`` and ``stunet``.

  * Denoising: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``multiresunet``, ``seunet``, ``resunet_se``, ``unetr``, ``unext_v1``, ``unext_v2``, ``hrnet`` and ``stunet``.

  * Super-resolution: ``edsr``, ``rcan``, ``dfcan``, ``wdsr``, ``unet``, ``resunet``, ``resunet++``, ``seunet``, ``resunet_se``, ``attention_unet``, ``multiresunet``, ``unext_v1`` and ``unext_v2``

  * Self-supervision: ``edsr``, ``rcan``, ``dfcan``, ``wdsr``, ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet``, ``resunet_se``, ``unext_v1``, ``unext_v2``, ``multiresunet``, ``unetr``, ``vit`` and ``mae``.

  * Classification: ``simple_cnn``, ``efficientnet_b0``, ``efficientnet_b1``, ``efficientnet_b2``, ``efficientnet_b3``, ``efficientnet_b4``, ``efficientnet_b5``, ``efficientnet_b6``, ``efficientnet_b7``, ``vit``. 

  * Image to image: ``edsr``, ``rcan``, ``dfcan``, ``wdsr``, ``unet``, ``resunet``, ``resunet++``, ``seunet``, ``resunet_se``, ``attention_unet``, ``unetr``, ``multiresunet``, ``unext_v1``, ``unext_v2``, ``hrnet`` and ``stunet``  

  An example of a U-Net-like model configuration is as follows:

  .. code-block:: yaml

    MODEL:
      SOURCE: biapy
      ARCHITECTURE: resunet
      FEATURE_MAPS: [32, 64, 128, 256]
      Z_DOWN: [2, 2, 2]
      NORMALIZATION: "in"
      KERNEL_SIZE: 3

  An example of a STU-Net model configuration is as follows:

  .. code-block:: yaml

    MODEL:
      SOURCE: biapy
      ARCHITECTURE: stunet
      STUNET:
        VARIANT: 'base'
        PRETRAINED: True

- ``bmz``, which uses `Bioimage Model Zoo (bioimage.io) <https://bioimage.io/#/>`__ pretrained models. Use ``MODEL.BMZ.SOURCE_MODEL_ID`` to select the model. More a more models are added to the zoo so please check `Bioimage Model Zoo page <https://bioimage.io/#/>`__ to see available models. BiaPy can only consume models exported with `Pytorch state dict <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html#:~:text=A%20state_dict%20is%20an%20integral,to%20PyTorch%20models%20and%20optimizers.>`__. 

- ``torchvision``, which uses models defined in `TorchVision <https://pytorch.org/vision/stable/models.html>`__. Use ``MODEL.TORCHVISION_MODEL_NAME`` to select the model. Notice that most of the models were trained in natural images (not biomedical) and most of them are for classification. On top of that, some use bounding-box-annotations, which are not supported in BiaPy so only inference/prediction/test can only be done. All the models will load by default the best pretrained weights offered. Currently, BiaPy supports the following models for each workflow: 

  * Semantic segmentation (defined `here <https://pytorch.org/vision/stable/models.html#semantic-segmentation>`__): ``deeplabv3_mobilenet_v3_large``, ``deeplabv3_resnet101``, ``deeplabv3_resnet50``, ``fcn_resnet101``, ``fcn_resnet50`` and ``lraspp_mobilenet_v3_large``. 

  * Instance segmentation (defined `here <https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection>`__) but only for inference: ``maskrcnn_resnet50_fpn`` and ``maskrcnn_resnet50_fpn_v2``. 

  * Detection (defined `here <https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection>`__) but only for inference: ``fasterrcnn_mobilenet_v3_large_320_fpn``, ``fasterrcnn_mobilenet_v3_large_fpn``, ``fasterrcnn_resnet50_fpn``, ``fasterrcnn_resnet50_fpn_v2``, ``fcos_resnet50_fpn``, ``ssd300_vgg16``, ``ssdlite320_mobilenet_v3_large``, ``retinanet_resnet50_fpn`` and ``retinanet_resnet50_fpn_v2``.

  * Classification (defined `here <https://pytorch.org/vision/stable/models.html#classification>`__): ``alexnet``, ``convnext_base``, ``convnext_large``, ``convnext_small``, ``convnext_tiny``, ``densenet121``, ``densenet161``, ``densenet169``, ``densenet201``, ``efficientnet_b0``, ``efficientnet_b1``, ``efficientnet_b2``, ``efficientnet_b3``, ``efficientnet_b4``, ``efficientnet_b5``, ``efficientnet_b6``, ``efficientnet_b7``, ``efficientnet_v2_l``, ``efficientnet_v2_m``, ``efficientnet_v2_s``, ``googlenet``, ``inception_v3``, ``maxvit_t``, ``mnasnet0_5``, ``mnasnet0_75``, ``mnasnet1_0``, ``mnasnet1_3``, ``mobilenet_v2``, ``mobilenet_v3_large``, ``mobilenet_v3_small``, ``quantized_googlenet``, ``quantized_inception_v3``, ``quantized_mobilenet_v2``, ``quantized_mobilenet_v3_large``, ``quantized_resnet18``, ``quantized_resnet50``, ``quantized_resnext101_32x8d``, ``quantized_resnext101_64x4d``, ``quantized_shufflenet_v2_x0_5``, ``quantized_shufflenet_v2_x1_0``, ``quantized_shufflenet_v2_x1_5``, ``quantized_shufflenet_v2_x2_0``, ``regnet_x_16gf``, ``regnet_x_1_6gf``, ``regnet_x_32gf``, ``regnet_x_3_2gf``, ``regnet_x_400mf``, ``regnet_x_800mf``, ``regnet_x_8gf``, ``regnet_y_128gf``, ``regnet_y_16gf``, ``regnet_y_1_6gf``, ``regnet_y_32gf``, ``regnet_y_3_2gf``, ``regnet_y_400mf``, ``regnet_y_800mf``, ``regnet_y_8gf``, ``resnet101``, ``resnet152``, ``resnet18``, ``resnet34``, ``resnet50``, ``resnext101_32x8d``, ``resnext101_64x4d``, ``resnext50_32x4d``, ``retinanet_resnet50_fpn``, ``shufflenet_v2_x0_5``, ``shufflenet_v2_x1_0``, ``shufflenet_v2_x1_5``, ``shufflenet_v2_x2_0``, ``squeezenet1_0``, ``squeezenet1_1``, ``swin_b``, ``swin_s``, ``swin_t``, ``swin_v2_b``, ``swin_v2_s``, ``swin_v2_t``, ``vgg11``, ``vgg11_bn``, ``vgg13``, ``vgg13_bn``, ``vgg16``, ``vgg16_bn``, ``vgg19``, ``vit_b_16``, ``vit_b_32``, ``vit_h_14``, ``vit_l_16``, ``vit_l_32``, ``wide_resnet101_2`` and ``wide_resnet50_2``.


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

* Detection: ``"CE"`` always used (`Cross entropy loss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`__). No other value can be set.

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

.. _config_test:


Weighting options
~~~~~~~~~~~~~~~~~

This section outlines how to configure weighting across different tasks. Understanding the hierarchy of these variables is essential, as they can be applied at the loss function level, the data channel level, or the class level.

**Global Loss Weighting**. These variables represent the most basic level of configuration, typically introduced in the Semantic Segmentation workflow. They control how individual loss components are merged and how specific pixel values are handled:

  * Combining multiple losses (``LOSS.WEIGHTS``): When using a combined loss (e.g., ``LOSS.TYPE = "W_CE_DICE"``), the library computes a weighted sum of the components. ``LOSS.WEIGHTS`` defines a multiplier for each specific loss. The length of the list must be equal to the number of losses being combined. In the case of ``LOSS.TYPE = "W_CE_DICE"``, the final loss will be: ``LOSS.WEIGHTS[0] * CE + LOSS.WEIGHTS[1] * DICE``.
  * Excluding data (``LOSS.IGNORE_INDEX``): To exclude certain pixel values from contributing to the loss, you can set ``LOSS.IGNORE_INDEX`` to the desired value. This is particularly useful for ignoring unlabeled regions in semantic segmentation tasks. For instance, if your dataset uses a specific value (e.g., ``-1`` or ``255``) to denote unlabeled pixels, setting ``LOSS.IGNORE_INDEX`` to that value will ensure those pixels do not affect the loss calculation or the evaluation metrics like IoU.
  * Class rebalancing (``LOSS.CLASS_REBALANCE``): This variable controls how the loss function handles class imbalance. It can be set to ``'none'`` for no rebalancing or ``'manual'`` to use custom weights defined in ``LOSS.CLASS_WEIGHTS``. This is particularly important in scenarios where certain classes are underrepresented in the dataset, as it helps the model learn to focus on those classes.

All of the above options are available mainly for semantic segmentation, but they can also be used in instance segmentation and detection workflows, when they predict a class channel apart from the other data channels. However, in these workflows, there are additional weighting options that can be applied at the data channel level and within channels to address specific challenges associated with these tasks as described below.

**Data Channel Weighting (Instance & Detection)**. In workflows like Instance Segmentation and Detection, the model predicts multiple data channels. A data channel is a specific type of information predicted, such as, the binary mask of the object (denoted as ``F``, Foreground in instance segmentation), the contours of the object (denoted as ``C``), the distance transform (denoted as ``Db``) etc. On top of that, a class channel can also be predicted to determine the class of the object (actually one channel per class). For detection workflow instead, there will be one channel predicting the center of the object, and optionally, one channel per class predicting the class of the object. In these workflows, you can set different weights for each of the output channels of the model with the variable ``PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS`` or ``PROBLEM.DETECTION.DATA_CHANNEL_WEIGHTS``. This allows you to prioritize the accuracy of one type of information over another. For example, in instance segmentation, you might want to prioritize the accuracy of the foreground mask (``F``) over the distance transform (``Db``) by assigning a higher weight to ``F`` in the loss calculation.

**Intra-Channel Balancing**. Within a specific channel, like "Contours" (``C``), the background often heavily outweighs the feature pixels. To address this imbalance, you can apply class rebalancing with ``PROBLEM.INSTANCE_SEG.CLASS_REBALANCE_WITHIN_CHANNELS`` and ``PROBLEM.DETECTION.CLASS_REBALANCE_WITHIN_CHANNELS``, which assigns different weights to the foreground and background classes within that channel. This is particularly useful in scenarios where the foreground objects are significantly smaller than the background, as it helps the model learn to focus on the relevant features. This option is available for the channels that have a binary mask as target, e.g. ``F`` and ``C`` channels in instance segmentation, and the center channel in detection.

Taking into account the above considerations a common configuration for instance segmentation workflow, including class prediction on top of instances, can be as follows:

.. code-block:: yaml

  PROBLEM:
    INSTANCE_SEG:
      DATA_CHANNELS: ['F', 'C', 'Db']
      DATA_CHANNEL_WEIGHTS: [0.6, 0.3, 0.1]
      CLASS_REBALANCE_WITHIN_CHANNELS: True

  LOSS:
    IGNORE_INDEX: -1
    CLASS_REBALANCE: 'manual'
    CLASS_WEIGHTS: [1, 1.3, 1.8]

Test phase
~~~~~~~~~~

The test phase, also referred to as inference or prediction, is the stage in which a trained model is applied to unseen images in order to generate the final output predictions. In BiaPy, this phase is enabled by setting ``TEST.ENABLE = True``.

During inference, memory usage must be considered carefully. There are two different types of memory constraints that can affect the prediction process:

* **Machine/workstation RAM memory**: this is the system memory used to load the input test image, store intermediate arrays, keep the model prediction, and reconstruct the final output. Even if the image can be processed by the GPU, the complete image and its corresponding prediction must still fit in RAM at some point. This is especially relevant for large 2D images, 3D volumes, or workflows that require storing probability maps. By default, predictions are stored as ``float32`` arrays, although memory usage can be reduced by enabling ``TEST.REDUCE_MEMORY = True``, which stores predictions as ``float16``.

* **GPU memory**: this is the memory available in the graphics card during the forward pass of the model. The GPU must be able to store the input patch, the model activations, and the output prediction for that patch. If the complete test image does not fit in GPU memory, it cannot be inferred in a single forward pass. In that case, the image must be divided into smaller patches.

In this phase you can enable test-time augmentation by setting ``TEST.AUGMENTATION = True``, which will create multiple augmented copies of each patch, or image if ``TEST.FULL_IMG = True``, by all possible rotations (``8`` copies in 2D and ``16`` in 3D). This will slow down the inference process, but it will return more robust predictions. Apart from that, you can use also use ``DATA.REFLECT_TO_COMPLETE_SHAPE = True`` to ensure that the patches can be made as pointed out in :ref:`data_management`. 

BiaPy provides two main inference strategies depending on these memory constraints and on the size of the test images.

Inference entire image
**********************

This option is used when the complete test image can be loaded into the machine/workstation RAM and can also be processed by the GPU in a single forward
pass (``TEST.FULL_IMG = True``).

In this mode, the full image is provided directly to the model, and the complete prediction is generated at once. This is the simplest and fastest inference mode
because no patch extraction or reconstruction step is required. 

However, this strategy is only possible when both of the following conditions are met:

* The input image and its corresponding output prediction fit in the machine/workstation RAM.
* The complete image fits in GPU memory during model inference.

This mode is usually suitable for small or medium-sized images, or for models with low memory requirements. It avoids stitching artifacts because the model sees the entire image at once.

Prediction fits in RAM but not in GPU memory
********************************************

This is the most common situation when working with large images or 3D volumes (``TEST.FULL_IMG = False``). The full test image can be loaded into the machine/workstation RAM, and the final prediction can also be stored and reconstructed in RAM. However, the complete image cannot be sent to the GPU at once because the GPU memory is not large enough to store the input image, the model activations, and the output prediction during the forward pass.

To overcome this limitation, BiaPy crops the image into smaller patches. Each patch is processed independently by the model, reducing the amount of GPU memory required at any given moment. Once all patches have been predicted, their outputs are merged to reconstruct the prediction with the same spatial shape as the original image.

Usually, patches are extracted with overlap and/or padding. This is done to reduce border artifacts, since predictions near the borders of a patch may be less accurate than predictions near the center. During reconstruction, overlapping regions are combined to produce a smoother final prediction.

This strategy addresses a **GPU memory limitation**, not a **machine/workstation RAM limitation**. The full image and the final reconstructed prediction are still kept in RAM, so they must fit in the available system memory. If they do not fit in RAM, a different chunked or out-of-memory strategy is needed.

If a CUDA out-of-memory error occurs during this mode, the crop or patch size should be reduced. If GPU memory usage is low, the crop size can be increased to improve inference speed.

Inference by chunks
*******************

When dealing with volumes that are too large to fit in GPU memory at once, BiaPy can process them in overlapping patches using ``TEST.BY_CHUNKS.ENABLE = True``. It splits the volume into patches, processes each patch independently (optionally across multiple GPUs or cluster jobs), and writes results into a shared Zarr store.

The pipeline is split into explicit phases, controlled by the ``TEST.BY_CHUNKS.PHASES`` list. Each cluster job specifies which phases it runs:

.. list-table:: Test-time phases
   :widths: 25 75
   :header-rows: 1

   * - Phase name
     - What it does
   * - ``prediction``
     - Runs the model over all patches and writes raw predictions to a Zarr file.
   * - ``instance_creation``
     - Post-processes raw predictions patch-by-patch to create per-chunk instance labels. This applies to instance segmentation only.
   * - ``instance_merging``
     - Merges per-chunk instance labels across the full volume into globally consistent IDs. This applies to instance segmentation only.

The default is ``PHASES = ["prediction", "instance_creation", "instance_merging"]``, which runs all phases in one job. For large volumes, split phases across jobs (see `cluster_parallel_execution`_ below).


Phase 1 — Raw model prediction
==============================

**Activated when:** ``"prediction"`` in ``TEST.BY_CHUNKS.PHASES`` and ``TEST.REUSE_PREDICTIONS = False``. If either condition is not met, Phase 1 is skipped entirely.

**Setup.** The data file (Zarr or HDF5) is opened lazily. A patch grid is computed over the volume with step size ``step_z = crop_shape[0] - 2*padding[0]`` (and analogously for Y and X), so adjacent patches overlap by ``2*padding`` voxels. The total number of patches is ``ceil(Z/step_z) * ceil(Y/step_y) * ceil(X/step_x)``.

**Z sub-range.** Set ``TEST.BY_CHUNKS.Z_START`` and ``TEST.BY_CHUNKS.Z_END`` to restrict Phase 1 to a contiguous block of Z slices. Both values are in voxel coordinates (0-indexed, ``Z_END`` is exclusive). Internally the generator converts them to chunk indices using **ceiling division**:

.. code-block:: python

  z_vol_start = ceil(Z_START / step_z)
  z_vol_end   = ceil(Z_END   / step_z)

Using ``ceil`` for both boundaries guarantees that adjacent jobs assign their chunks at exactly the same index with no overlap and no gap.

**Patch distribution.** A ``DistributedSampler`` divides the patch indices across all GPUs and dataloader workers, so each patch is processed by exactly one worker.

**Per-patch loop.** For each patch:

* The input patch (plus padding) is loaded from the source Zarr/HDF5.
* The model runs inference and returns logits or probabilities.
* The padding region is stripped from the output.
* The output patch is written into a shared prediction Zarr whose shape always matches the full volume (not just the Z sub-range), so multiple cluster jobs can write concurrently without conflict. Zarr chunk boundaries are aligned to ``(step_z, step_y, step_x, C)``, guaranteeing each write tile is owned by exactly one job.

**Sync and close.** After all patches are processed, a distributed barrier ensures all workers have finished writing before the main process continues. Open file handles are closed.

**TIF export.** If ``TEST.BY_CHUNKS.SAVE_OUT_TIF = True``, the main process converts the prediction Zarr to a TIFF file. This is a single-threaded operation that can be time-consuming for large volumes, so it is optional. Also, be aware that the TIFF format does not support chunking, so the entire prediction must be loaded into memory during export. **If the prediction is too large to fit in RAM, this step will fail.**

Phase 2 — Workflow post-processing
==================================

**Activated when:** ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.ENABLE = True``. Post-processing runs after Phase 1 (or independently when Phase 1 is skipped). Its behaviour depends on the workflow type. What happens in this phase depends on the workflow and on ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE``.

Semantic segmentation
#####################

The prediction Zarr is read patch-by-patch and each patch is binarized (or argmaxed for multi-class). The results are written to a new output Zarr aligned to the same tile grid. No cross-patch communication is needed.

Detection
#########

Peak detection runs per chunk; detected point coordinates are accumulated and written to a global CSV file.

Instance segmentation — ``chunk_by_chunk`` mode
###############################################

This is the most complex post-processing path. It runs a 5-pass algorithm to produce globally consistent instance IDs across the full volume. Requires ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE = "chunk_by_chunk"``.

The passes are split into two phase groups:

**"instance_creation" phase — Pass A (per-chunk watershed)**

Activated when ``"instance_creation"`` in ``TEST.BY_CHUNKS.PHASES``.

If this phase is absent the algorithm assumes a completed instance label Zarr already exists on disk (written by a prior cluster job) and skips directly to the merging phases.

For each tile in the prediction Zarr:

#. A halo-extended tile is loaded: the tile is expanded by ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.INSTANCE_SEG_HALO`` voxels on each side (clamped to volume boundaries) to provide boundary context.
#. Watershed (or the configured instance method) is run on the halo-extended tile.
#. The halo region is stripped and the core tile is written to the instance label Zarr.

The ``TEST.BY_CHUNKS.Z_START``/``TEST.BY_CHUNKS.Z_END`` restriction applies here too, using the same ``ceil``-based chunk-index mapping. The tile step for Pass A is ``step_z = crop_shape[0]`` (no padding subtracted), so the chunk grid is coarser than Phase 1's.

**"instance_merging" phase — Passes B–E (global ID stitching)**

Activated when ``"instance_merging"`` in ``TEST.BY_CHUNKS.PHASES``.

Requires the full instance label Zarr to be complete (all Z sub-ranges written). ``TEST.BY_CHUNKS.Z_START``/``TEST.BY_CHUNKS.Z_END`` are not applied to the merging passes — they always operate over the entire volume.

* Pass B — Global ID offset (prefix-sum). Each chunk's labels are shifted by the cumulative count of instances in all preceding chunks, making all instance IDs globally unique.
* Pass C — Boundary-edge IoU extraction. For every pair of adjacent chunks, the touching faces are loaded and intersection-over-union is computed between instances that span the boundary. Pairs exceeding ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.INSTANCE_SEG_MERGE_IOU_TH`` are recorded as merge candidates.
* Pass D — Union-Find (rank 0 only). All merge candidates are fed into a Union-Find structure on the main process to resolve transitive merges.
* Pass E — Relabelling. Each chunk is relabelled according to the Union-Find result, producing the final globally consistent instance map.

After Pass E, if ``TEST.BY_CHUNKS.SAVE_OUT_TIF = True``, the completed instance Zarr is converted to TIFF. This is a single-threaded operation that can be time-consuming for large volumes, so it is optional. Also, be aware that the TIFF format does not support chunking, so the entire prediction must be loaded into memory during export. **If the prediction is too large to fit in RAM, this step will fail.**

Instance segmentation — ``entire_pred`` mode
############################################

The entire prediction volume is loaded into memory after Phase 1 and instance segmentation is run as a single operation. Only suitable for volumes that fit in RAM. The ``TEST.BY_CHUNKS.PHASES`` mechanism has no effect on this mode.

.. _cluster_parallel_execution:

Cluster-parallel execution
==========================

For very large volumes, split the pipeline across multiple cluster jobs using ``TEST.BY_CHUNKS.Z_START``, ``TEST.BY_CHUNKS.Z_END``, and ``TEST.BY_CHUNKS.PHASES``.

Example: two prediction+creation jobs, one merging job.

.. list-table:: Example split of test-time phases across jobs
   :widths: 20 20 20 40
   :header-rows: 1

   * - Job
     - ``Z_START``
     - ``Z_END``
     - ``PHASES``
   * - Job 1
     - ``0``
     - ``K``
     - ``["prediction", "instance_creation"]``
   * - Job 2
     - ``K``
     - ``end of volume``
     - ``["prediction", "instance_creation"]``
   * - Job 3
     - ``not set``
     - ``not set``
     - ``["instance_merging"]``

Jobs 1 and 2 can run **in parallel** (they write to non-overlapping Zarr chunks). Job 3 must run **after both** are complete.

**Choosing K.** For fully parallel execution of Phase 1 and Pass A simultaneously (within the same job), K must be a multiple of **crop_shape[0]**. This ensures Phase 1's finer tile grid covers exactly the same Z extent as Pass A's coarser tile grid at the boundary.

If Jobs 1 and 2 run sequentially (one finishes before the other starts), any value of K is safe because Phase 1's smaller step size means it always produces data beyond what Pass A reads up to ``TEST.BY_CHUNKS.Z_END``.

**No-overlap guarantee.** Because both generators use ``ceil(Z_boundary / step_z)`` to convert voxel boundaries to chunk indices, adjacent jobs always split at the same chunk index:

.. code-block:: python
  
  job1.z_vol_end   = ceil(K / step_z)
  job2.z_vol_start = ceil(K / step_z)

There is no duplicated chunk and no missing chunk regardless of whether K falls on a step boundary. 

**Reuse predictions.** Setting ``TEST.REUSE_PREDICTIONS = True`` also skips Phase 1 (equivalent to omitting "prediction" from  ``TEST.BY_CHUNKS.PHASES``). The two mechanisms can be combined: ``TEST.BY_CHUNKS.PHASES = ["instance_creation"]`` with ``REUSE_PREDICTIONS = True`` skips prediction and reruns only Pass A on existing prediction Zarrs.

**Validation.** BiaPy raises a ``ValueError`` at config load time if:

* ``TEST.BY_CHUNKS.PHASES`` is empty or contains an unknown phase name.
* ``"instance_creation"`` or ``"instance_merging"`` appears in ``TEST.BY_CHUNKS.PHASES`` for a non-instance-segmentation workflow.
* ``"instance_creation"`` or ``"instance_merging"`` appears in ``TEST.BY_CHUNKS.PHASES`` but ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.ENABLE = False`` or ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE`` is not ``"chunk_by_chunk"``. 
* ``TEST.BY_CHUNKS.Z_START >= TEST.BY_CHUNKS.Z_END`` when both are set.


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