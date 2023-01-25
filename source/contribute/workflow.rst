.. _workflow_creation_contrib:

Add new workflow
----------------

Creating a new workflow requires considering the different areas where custom functions can be added. Additionally, to fully integrate the new workflow, it is necessary to follow the steps outlined in the :ref:`general_guidelines_contrib` section of the documentation.

All workflows are defined within the ``engine`` folder and constructed in the ``engine/engine.py`` file, specifically in the `test <https://github.com/danifranco/BiaPy/blob/ca6351bd73b9c952cba3b4d97b88116f58432af7/engine/engine.py#L208>`__ function. The initial lines in this function pertain to post-processing configuration, for more information on this see the :ref:`pre_post_contrib` section of the documentation. After that, each workflow is built, and the inference is performed using the ``process_sample`` function. The inference is done for each test image (currently one by one).

In addition to the ``process_sample`` function, which is the primary function, creating the ``print_stats`` and ``normalize_stats`` functions (which will be explained later), ``after_merge_patches``, ``after_full_image``, and ``after_all_images`` functions are also necessary. To understand how these functions work, it's helpful to know that the ``process_sample`` function has mainly three steps:

1. If the ``TEST.STATS.PER_PATCH`` variable is enabled, the test image is divided into smaller patches which are then passed through the model. After that, the original image shape is reconstructed.
2. If the ``TEST.STATS.MERGE_PATCHES`` variable is enabled, IoU metrics are calculated and a post-processing step is performed for 3D data.
When ``TEST.STATS.FULL_IMG`` is enabled and working in 2D, full images are passed through the model.
3. The ``after_merge_patches`` function is applied after the second step (regardless of whether ``TEST.STATS.MERGE_PATCHES`` is enabled), the ``after_full_image`` function is applied after the third step, and the ``after_all_images`` function is applied when all images have been passed through the ``process_sample`` function. With these three functions, you have the flexibility to apply any process you want after each main step.

Finally, you should implement the ``print_stats`` and ``normalize_stats`` functions within your own workflow so that you can print the values of the specific metrics you have calculated. For example, in instance segmentation, we reimplement ``print_stats`` to print matching metrics used to evaluate instances (see `here <https://github.com/danifranco/BiaPy/blob/64079785bd666d2fc7775b4437e2765e8162320d/engine/instance_seg.py#L153>`__). In the same way, we reimplement ``normalize_stats`` to normalize also the new measurements (see `here <https://github.com/danifranco/BiaPy/blob/64079785bd666d2fc7775b4437e2765e8162320d/engine/instance_seg.py#L144>`__).  