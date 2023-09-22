.. _pre_post_contrib:

Add pre/post processing
-----------------------

To create new pre-processing and post-processing methods, the following steps should be taken:

* **Pre-processing**: This step is used to convert or create a new target from another source. For example, in instance segmentation, instance labels are converted into different channels for the network to learn. Another example is in the detection workflow, where CSV files are converted into binary label masks. Note that data normalization is performed within the generators (as described in the :ref:`data_norm` section of the documentation) rather than in these pre-processing functions.

  This step is applied in the ``engine/engine.py`` file, where the custom function should be inserted. It should be placed under the following heading: ::

      ####################
      #  PRE-PROCESSING  #
      ####################

  `Here <https://github.com/danifranco/BiaPy/blob/ca6351bd73b9c952cba3b4d97b88116f58432af7/engine/engine.py#L38>`__ is a link to the exact line when this part of the documentation was made. 

  The pre-processing function is typically located within the class that represents your workflow (if a new one is being created).

* **Post-processing**: There are different ways to perform post-processing with the BiaPy library, as described in the :ref:`config_test` section. It is recommended to read this section before continuing.

In general, post-processing can be divided into two moments: 1) after the network prediction and 2) after each main process in the workflow. The user must choose where to apply the post-processing function. All the functions are located in ``data/post_processing/post_processing.py``.

    1. After the network prediction, the post-processing methods are applied to improve the resulting probabilities. For example, in semantic segmentation, the output probabilities are binarized, and in instance segmentation, instances are created from the probabilities. The first post-processing step applied is the application of a binary mask (if selected) to the complete image that is reconstructed by merging patches (using ``TEST.STATS.PER_PATCH`` and ``TEST.STATS.MERGE_PATCHES``) or by doing the full image (using ``TEST.STATS.FULL_IMG``). This is done with the ``apply_binary_mask`` function. The rest of the post-processing methods that try to improve the resulting probabilities are done with the ``apply_post_processing`` function defined in ``data/post_processing/init.py``. These methods are tailored for 3D images, so they are done after ``TEST.STATS.PER_PATCH`` and ``TEST.STATS.MERGE_PATCHES`` in 3D, or after all images have been predicted and they are stacked as a 3D image by enabling ``TEST.ANALIZE_2D_IMGS_AS_3D_STACK``.

    2. In addition to the post-processing methods mentioned above, each workflow can have another post-processing step. For example, in the detection workflow, after the points have been extracted from the probabilities (the final output of the detection workflow), a filtering of close points is done with ``TEST.POST_PROCESSING.REMOVE_CLOSE_POINTS``. If you want to add this kind of process, you can see an example of how it is done in the instance segmentation workflow by looking at the code located `here <https://github.com/danifranco/BiaPy/blob/ca6351bd73b9c952cba3b4d97b88116f58432af7/engine/instance_seg.py#L97>`__ .