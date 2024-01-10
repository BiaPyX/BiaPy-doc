.. _general_guidelines_contrib:

General guidelines
------------------

In this section, the structure of BiaPy and the various steps involved in executing a workflow are outlined. The goal is to provide guidance to future contributors on where to modify the library in order to add new features. It is recommended to first read the :ref:`how_works_biapy` section, as it provides an overview of the overall functioning of BiaPy. The steps involved are as follows:

1. In the ``config/config.py`` file, all the variables that the user can define in the ``.yaml`` file are defined. This is where new workflow-specific variables should be added.

2. In the ``engine/check_configuration.py`` file, the checks for the workflow variables should be added.

3. The main process is then run by the ``engine/engine.py`` file. The following steps take place in this file, unless otherwise specified. If your workflow includes pre-processing, it should be declared under the following heading: ::

    ####################
    #  PRE-PROCESSING  #
    ####################


  `This link <https://github.com/BiaPyX/BiaPy/blob/ca6351bd73b9c952cba3b4d97b88116f58432af7/engine/engine.py#L38>`__ points to the specific line in the library's code where this part of the documentation was written. However, it should be noted that the library is under continuous development, so the link may not always point to the current version of the code. For more information on pre-processing, refer to the :ref:`pre_post_contrib` section of the documentation.

4. The next step in the workflow is to load the data. If the ``DATA.TRAIN.IN_MEMORY`` variable is enabled, the training, validation, and test data are loaded into memory. If this variable is disabled, the data generators (which is the next step) will load each image from the disk on-the-fly. Assuming that ``DATA.TRAIN.IN_MEMORY`` is enabled, there are two options for creating the training and validation data:

    * Use the ``load_and_prepare_2D_train_data`` (for 2D data) or ``load_and_prepare_3D_data`` (for 3D data) functions to load the images and their labels.
    * Use the ``load_data_classification`` (for 2D data) or ``load_3d_data_classification`` (for 3D data) functions to load the images and their classes (used for classification workflows).

  When loading data, you can choose to use any of the provided functions or use them as a reference for creating your own data loading process. It is important to note that the validation data can either be extracted from the training data (if the ``DATA.VAL.FROM_TRAIN`` variable is enabled) or loaded directly from a separate folder. The specific line in the code where this step is implemented can be found at this `link <https://github.com/BiaPyX/BiaPy/blob/ca6351bd73b9c952cba3b4d97b88116f58432af7/engine/engine.py#L69>`__. This step is under the following heading: ::
        
      #################
      #   LOAD DATA   #
      #################

  The test data is loaded in a similar manner to the training and validation data, using functions such as ``load_data_from_dir`` (for 2D data) and ``load_3d_images_from_dir`` (for 3D data), and for classification workflow, ``load_data_classification`` (for 2D data) or ``load_3d_data_classification`` (for 3D data).

5. After loading the data, the train and validation generators are created by calling the ``create_train_val_augmentors`` function in the ``data/generators/init.py`` file. There are two main classes of generators:

    * ``PairBaseDataGenerator`` which is extended by ``Pair2DImageDataGenerator`` (for 2D data) or ``Pair3DImageDataGenerator`` (for 3D data). These generators yield the image and its mask, and all data augmentation (DA) techniques are done considering the mask as well. These generators are used for all workflows except classification.
    * ``SingleBaseDataGenerator`` which is extended by ``Single2DImageDataGenerator`` (for 2D data) or ``Single3DImageDataGenerator`` (for 3D data). These generators yield the image and its class (integer). These generators are used for classification workflows. BiaPy uses the `imgaug <https://github.com/aleju/imgaug>`__ library, so when adding new DA methods, it's recommended to check if it has already been implemented by that library. Custom DA techniques can be placed in the ``data/generators/augmentors.py`` file.

  Finally, to enable parallelization, the last code lines of the ``create_train_val_augmentors`` function have been added. There, the generator output datatype must be specified. You can find the specific line `here <https://github.com/BiaPyX/BiaPy/blob/ca6351bd73b9c952cba3b4d97b88116f58432af7/data/generators/__init__.py#L196>`__. Adapt it to your specific case.

  This step `here <https://github.com/BiaPyX/BiaPy/blob/ca6351bd73b9c952cba3b4d97b88116f58432af7/engine/engine.py#L164>`__, under the following heading: ::
    
      ########################
      #  PREPARE GENERATORS  #
      ########################

6. The next step after the data and generator process is to define the model. A new model should be added to the ``models`` folder and constructed in the ``models/init.py`` file. Following this, the training process can begin.

7. The final step is to create a custom workflow class. For detailed instructions on how to create a workflow, refer to the :ref:`workflow_creation_contrib` section of the documentation.