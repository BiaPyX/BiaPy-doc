Configuration
-------------
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
        OPTIMIZER: SGD 
        LR: 1.E−3
        BATCH_SIZE: 6
        EPOCHS: 360
    TEST:
        POST_PROCESSING:
            YZ_FILTERING: True
            

In order to run BiaPy, a YAML configuration file must be created. Examples for each workflow can be found in the `templates <https://github.com/BiaPyX/BiaPy/tree/master/templates>`__ folder on the BiaPy GitHub repository. If you are unsure about which workflow is most suitable for your data, you can refer to the `Select Workflow <select_workflow.html>`__ page for guidance.

The options for the configuration file can be found in the `config.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/config/config.py>`_ file on the BiaPy GitHub repository. However, some of the most commonly used options are explained below:

System
~~~~~~

To limit the number of CPUs used by the program, use the ``SYSTEM.NUM_CPUS`` option. 

Problem specification
~~~~~~~~~~~~~~~~~~~~~

To specify the type of workflow, use the ``PROBLEM.TYPE`` option and select one of the following options: ``SEMANTIC_SEG``, ``INSTANCE_SEG``, ``DETECTION``, ``DENOISING``, ``SUPER_RESOLUTION``, ``SELF_SUPERVISED``, or ``CLASSIFICATION``.

To specify whether the data is 2D or 3D, use the ``PROBLEM.NDIM`` option and select either ``2D`` or ``3D``.

.. _data_management:

Data management
~~~~~~~~~~~~~~~

The ``DATA.PATCH_SIZE`` variable is used to specify the shape of the images that will be used in the workflow. The order of the dimensions for 2D images is ``(y,x,c)`` and for 3D images it is ``(z,y,x,c)``.

The paths for the training data are set using the ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` variables (if necessary, depending on the specific workflow). Similarly, the paths for the validation data can be set using ``DATA.VAL.PATH`` and ``DATA.VAL.GT_PATH`` unless ``DATA.VAL.FROM_TRAIN`` is set, in which case these variables do not need to be defined. For test data, the ``DATA.TEST.PATH`` variable should be set if ``TEST.ENABLE`` is enabled. However, ``DATA.TEST.GT_PATH`` is not used when ``DATA.TEST.LOAD_GT`` is disabled, as there is usually no ground truth for test data.

There are two ways to handle the data during the workflow: 1) loading all images into memory at once, or 2) loading each image individually as it is needed. This behavior can be set for the training, validation, and test data using the ``DATA.TRAIN.IN_MEMORY``, ``DATA.VAL.IN_MEMORY``, and ``DATA.TEST.IN_MEMORY`` variables, respectively.

When loading training data into memory, i.e. setting ``DATA.TRAIN.IN_MEMORY`` to ``True``, all the images will be loaded into memory only once. During this process, each image will be divided into patches of size ``DATA.PATCH_SIZE`` using ``DATA.TRAIN.OVERLAP`` and ``DATA.TRAIN.PADDING``. By default, the minimum overlap is used, and the patches will always cover the entire image. In this configuration, the validation data can be created from the training data by setting ``DATA.VAL.SPLIT_TRAIN`` to the desired percentage of the training data to be used as validation. For this, ``DATA.VAL.FROM_TRAIN`` and ``DATA.VAL.IN_MEMORY`` must be set to ``True``. In general, loading training data in memory is the fastest approach, but it relies on having enough memory available on the computer.

Alternatively, when data is not loaded into memory, i.e. ``DATA.TRAIN.IN_MEMORY`` is set to False, a number of images equal to ``TRAIN.BATCH_SIZE`` will be loaded from the disk for each training epoch. If an image does not match the selected shape, i.e. ``DATA.PATCH_SIZE``, you can use ``DATA.EXTRACT_RANDOM_PATCH`` to extract a random patch from the image. As this approach requires loading each image multiple times, it is slower than the first approach but it saves memory.

.. seealso::

    In general, if for some reason the images loaded are smaller than the given patch size, i.e. ``DATA.PATCH_SIZE``, there will be no option to extract a patch from it. For that purpose the variable ``DATA.REFLECT_TO_COMPLETE_SHAPE`` was created so the image can be reshaped in those dimensions to complete ``DATA.PATCH_SIZE`` shape when needed.  

In the case of test data, even if ``DATA.TEST.IN_MEMORY`` is selected or not, each image is cropped to ``DATA.PATCH_SIZE`` using ``DATA.TEST.OVERLAP`` and ``DATA.TEST.PADDING``. Minimum overlap is made by default and the patches always cover the entire image. If ground truth is available you can set ``DATA.TEST.LOAD_GT`` to load it and measure the performance of the model. The metrics used depends on the workflow selected.

.. seealso::

    Set ``DATA.TRAIN.RESOLUTION`` and ``DATA.TEST.RESOLUTION`` to let the model know the resolution of training and test data respectively. In training, that information will be taken into account for some data augmentations. In test, that information will be used when the user selects to remove points from predictions in detection workflow. 

.. _data_norm:

Data normalization
~~~~~~~~~~~~~~~~~~

Two options are available for normalizing the data:

* Adjusting it to the ``[0-1]`` range, which is the default option. This can be done by setting ``DATA.NORMALIZATION.TYPE`` to ``div``.
* Custom normalization using a specified mean (``DATA.NORMALIZATION.CUSTOM_MEAN``) and standard deviation (``DATA.NORMALIZATION.CUSTOM_STD``). This can be done by setting ``DATA.NORMALIZATION.TYPE`` to ``custom``. If the mean and standard deviation are both set to ``-1``, which is the default, they will be calculated based on the training data. These values will be stored in the job's folder to be used at the inference phase, so that the test images are normalized using the same values. If specific values for mean and standard deviation are provided, those values will be used for normalization.

Data augmentation
~~~~~~~~~~~~~~~~~

The ``AUGMENTOR.ENABLE`` variable must be set to ``True`` to enable data augmentation (DA). The probability of each transformation is set using the ``AUGMENTOR.DA_PROB`` variable. BiaPy offers a wide range of transformations, which can be found in the `config.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/config/config.py>`__ file in the BiaPy repository on GitHub.

Images generated using data augmentation will be saved in the ``PATHS.DA_SAMPLES`` directory (which is ``aug`` by default). This allows you to check the data augmentation applied to the images. If you want a more exhaustive check, you can save all the augmented training data by enabling ``DATA.CHECK_GENERATORS``. The images will be saved in ``PATHS.GEN_CHECKS`` and ``PATHS.GEN_MASK_CHECKS``. Be aware that this option can consume a large amount of disk space as the training data will be entirely copied.

Model definition
~~~~~~~~~~~~~~~~
BiaPy offers three different backends to be used to choose a model (controlled by ``MODEL.SOURCE``):


- ``biapy``, which uses BiaPy as the backend for the model definition. Use ``MODEL.ARCHITECTURE`` to select the model. Different models for each workflow are implemented:

    * Semantic segmentation: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet``, ``multiresunet`` and ``unetr``. 

    * Instance segmentation: ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet``, ``multiresunet`` and ``unetr``. 

    * Detection: ``unet``, ``resunet``, ``resunet++``, ``attention_unet`` and ``seunet``.

    * Denoising: ``unet``, ``resunet``, ``resunet++``, ``attention_unet`` and ``seunet``.

    * Super-resolution: ``edsr``, ``rcan``, ``dfcan``, ``wdsr``, ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet`` and ``multiresunet``. 

    * Self-supervision: ``edsr``, ``rcan``, ``dfcan``, ``wdsr``, ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet``, ``multiresunet``, ``unetr``, ``vit`` and ``mae``.

    * Classification: ``simple_cnn``, ``efficientnet_b0``, ``efficientnet_b1``, ``efficientnet_b2``, ``efficientnet_b3``, ``efficientnet_b4``, ``efficientnet_b5``, ``efficientnet_b6``, ``efficientnet_b7``, ``vit``. 

    For ``unet``, ``resunet``, ``resunet++``, ``attention_unet`` and ``seunet`` architectures you can set ``MODEL.FEATURE_MAPS`` to determine the feature maps to use on each network level. In the same way, ``MODEL.DROPOUT_VALUES`` can be set for each level in those networks. For ``unetr`` and ``vit`` networks only the first value of those variables will be taken into account.

    The ``MODEL.BATCH_NORMALIZATION`` variable can be used to enable batch normalization on the ``unet``, ``resunet``, ``resunet++``, ``attention_unet``, ``seunet`` and ``unetr`` models. For the 3D versions of these networks (except for ``unetr``), the ``MODEL.Z_DOWN`` option can also be used to avoid downsampling in the z-axis, which is typically beneficial for anisotropic data.

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

- ``bmz``, which uses `Bioimage Model Zoo (bioimage.io) <https://bioimage.io/#/>`__ pretrained models. More a more models are added to the zoo so please check `Bioimage Model Zoo page <https://bioimage.io/#/>`__ to see available models. BiaPy can only consume models exported with `Torchscript <https://pytorch.org/docs/stable/jit.html>`__. 

Training phase
~~~~~~~~~~~~~~

To activate the training phase, set the ``TRAIN.ENABLE`` variable to ``True``. The ``TRAIN.OPTIMIZER`` variable can be set to either ``SGD``, ``ADAM`` or ``ADAMW``, and the learning rate can be set using the ``TRAIN.LR`` variable. If you do not have much expertise in choosing these settings, you can use ``ADAMW`` and ``1.E-4`` as a starting point. It is also possible to use a learning rate scheduler with ``TRAIN.LR_SCHEDULER`` variable.

Additionally, you need to specify how many images will be fed into the network at the same time using the ``TRAIN.BATCH_SIZE`` variable. For example, if you have ``100`` training samples and you select a batch size of ``6``, this means that ``17`` batches (``100/6 = 16.6``) are needed to input all the training data to the network, after which one epoch is completed.

To train the network, you need to specify the number of epochs using the ``TRAIN.EPOCHS`` variable. You can also set the patience using ``TRAIN.PATIENCE``, which will stop the training process if no improvement is made on the validation data for that number of epochs.

.. _config_test:

Test phase
~~~~~~~~~~

To initiate the testing phase, also referred to as inference or prediction, one must set the variable ``TEST.ENABLE`` to ``True`` within the BiaPy framework. BiaPy provides two distinct prediction options contingent upon the dimensions of the test images to be predicted. It is essential to consider that not only must the test image fit into memory, but also the model's prediction, characterized by a data type of ``float32`` (or ``float16`` if ``TEST.REDUCE_MEMORY`` is activated). Moreover, if the test image cannot be accommodated within the GPU memory, a cropping procedure becomes necessary. Typically, this entails cropping into patches with overlap and/or padding to circumvent border effects during the reconstruction of the original shape, albeit at the expense of increased memory usage. Given these considerations, two alternative procedures are available for predicting a test image:

- When each test image can be fit in memory the procedure is the following:
  
  If the test images are too large to be input directly into the GPU, for example, 3D images, you need to set ``TEST.STATS.PER_PATCH`` to ``True``. With this option, each test image will be divided into patches of size ``DATA.PATCH_SIZE`` and passed through the network individually, and then the original image will be reconstructed. This option will also automatically calculate performance metrics per patch if the ground truth is available (enabled by ``DATA.TEST.LOAD_GT``). You can also set ``TEST.STATS.MERGE_PATCHES`` to calculate the same metrics, but after the patches have been merged into the original image.

  If the entire images can be placed in the GPU, you can set only ``TEST.STATS.FULL_IMG`` without ``TEST.STATS.PER_PATCH`` and ``TEST.STATS.MERGE_PATCHES``, as explained above. This setting is only available for 2D images. Performance metrics will be calculated if the ground truth is available (enabled by ``DATA.TEST.LOAD_GT``).

  You can also use test-time augmentation by setting ``TEST.AUGMENTATION`` to ``True``, which will create multiple augmented copies of each test image, or patch if ``TEST.STATS.PER_PATCH`` has been selected, by all possible rotations (8 copies in 2D and 16 in 3D). This will slow down the inference process, but it will return more robust predictions.

  You can use also use ``DATA.REFLECT_TO_COMPLETE_SHAPE`` to ensure that the patches can be made as pointed out in :ref:`data_management`. 

  .. seealso::

      If the test images are large and you experience memory issues during the testing phase, you can set the ``TEST.REDUCE_MEMORY`` variable to ``True``. This will reduce memory usage as much as possible, but it may slow down the inference process.

- When each test image can not fit into memory (scalable solution):

  BiaPy offers to use `H5 <https://docs.h5py.org/en/stable/#:~:text=HDF5%20lets%20you%20store%20huge,they%20were%20real%20NumPy%20arrays.>`__ or `Zarr <https://zarr.readthedocs.io/en/stable/>`__ files to generate predictions by configuring ``TEST.BY_CHUNKS`` variable. In this setting, ``TEST.BY_CHUNKS.FORMAT`` decides which files are you working with and ``TEST.BY_CHUNKS.INPUT_IMG_AXES_ORDER`` sets the axis order (all the test images need to be order in the same way). This way, BiaPy enables multi-GPU processing per image by chunking large images into patches with overlap and padding to mitigate artifacts at the edges. Each GPU processes a chunk of the large image, storing the patch in its designated location using Zarr or H5 file formats. This is possible because these file formats facilitate reading and storing data chunks without requiring the entire file to be loaded into memory. Consequently, our approach allows the generation of predictions for large images, overcoming potential memory bottlenecks.
  
  .. warning::

      There is also an option to generate a TIFF file from the predictions with ``TEST.BY_CHUNKS.SAVE_OUT_TIF``. However, take into account that this option require to load the entire data into memory, which is sometimes not fleasible. 

  After the prediction is generated the variable ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.ENABLE`` controls whether the rest of the workflow process is going to be done or not (as may require large memory consumption depending on the workflow). If enabled, the prediction can be processed in two different ways (controlled by ``TEST.BY_CHUNKS.WORKFLOW_PROCESS.TYPE``):

    - ``chunk_by_chunk`` : prediction will be processed by chunks, where each chunk will be considered as an individual image. Select this operation if you have not enough memory to process the entire prediction image with ``entire_pred``.
    - ``entire_pred``: the predicted image will be loaded in memory at once and processed entirely (be aware of your memory budget).

  The option ``chunk_by_chunk`` is not trivial depending on the workflow, e.g. in instance segmentation different instances on each chunk need to be merged into one. Three workflows need to post-process the predictions to have a final result, semantic segmentation, instance segmentation and detection. Currently, ``chunk_by_chunk`` is only supported in detection workflow. 

Post-processing
~~~~~~~~~~~~~~~

BiaPy is equipped with several post-processing methods that are primarily applied in two distinct stages: 1) following the network's prediction and 2) after each primary process in the workflow is completed. The following is an explanation of these stages:

1.  After the network's prediction, the post-processing methods applied aim to improve the resulting probabilities returned by the network. This step is performed when the complete image is reconstructed by merging patches (``TEST.STATS.PER_PATCH`` and ``TEST.STATS.MERGE_PATCHES``) or when the full image is used (``TEST.STATS.FULL_IMG``).

    * A binary mask is applied to remove anything not contained within the mask. For this, the ``DATA.TEST.BINARY_MASKS`` path needs to be set.
    * Z-axis filtering is applied using the ``TEST.POST_PROCESSING.Z_FILTERING`` variable for 3D data when the ``TEST.STATS.PER_PATCH`` option is set. Additionally, YZ-axes filtering is implemented using the ``TEST.POST_PROCESSING.YZ_FILTERING`` variable.

2.  After each workflow main process is done there is another post-processing step on some of the workflows to achieve the final results. Find a full description of each method inside the workflow description:

    * Instance segmentation:

        * Big instance repair
        * Filter instances by morphological features

    * Detection:

        * Remove close points
        * Create instances from points