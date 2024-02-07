.. _general_guidelines_contrib:

How to contribute
-----------------

In this section, the structure of BiaPy and the various steps involved in executing a workflow are outlined. The goal is to provide guidance to future contributors on where to modify the library in order to add new features. It is recommended to first read the :ref:`how_works_biapy` section, as it provides an overview of the overall functioning of BiaPy. The steps involved are as follows:

1. In the ``biapy/config/config.py`` file, all the variables that the user can define in the ``.yaml`` file are defined. This is where new workflow-specific variables should be added.

2. In the ``biapy/engine/check_configuration.py`` file, the checks for the workflow variables should be added.

3. All the workflow extend the class Base_Workflow in ``biapy/engine/base_workflow.py``. 

4. During training, all the steps that a workflow must follow are listed in the first lines of `train() <https://github.com/BiaPyX/BiaPy/blob/d3abc3069ce490c688e102e96064be7463eae511/biapy/engine/base_workflow.py#L474>`__ function.

5. There are two main classes of generators:

    * ``PairBaseDataGenerator`` which is extended by ``Pair2DImageDataGenerator`` (for 2D data) or ``Pair3DImageDataGenerator`` (for ``3D`` data). These generators yield the image and its mask, and all data augmentation (DA) techniques are done considering the mask as well. These generators are used for all workflows except classification.
    * ``SingleBaseDataGenerator`` which is extended by ``Single2DImageDataGenerator`` (for 2D data) or ``Single3DImageDataGenerator`` (for ``3D`` data). These generators yield the image and its class (integer). These generators are used for classification workflows. BiaPy uses the `imgaug <https://github.com/aleju/imgaug>`__ library, so when adding new DA methods, it's recommended to check if it has already been implemented by that library. Custom DA techniques can be placed in the ``data/generators/augmentors.py`` file.

6. The next step after the data and generator process is to define the model. A new model should be added to the ``biapy.models`` folder and constructed in the ``biapy.models/init.py`` file. Following this, the training process can begin.

7. The model function call is done by `model_call_func <https://github.com/BiaPyX/BiaPy/blob/d3abc3069ce490c688e102e96064be7463eae511/biapy/engine/base_workflow.py#L374>`__, where depending on the backend used the model call is adapted. 

8. During test/inference, all the steps that a workflow must follow are listed in the first lines of `test() <https://github.com/BiaPyX/BiaPy/blob/d3abc3069ce490c688e102e96064be7463eae511/biapy/engine/base_workflow.py#L724>`__ function. There are two ways of do inference:

  * By using ``process_sample()``, where each test image is predicted by patches and reconstructed in an output TIF file. This method aims to deal with small and medium image sizes. 

  * By using ``process_sample_by_chunks()``, where the model prediction is reconstructed using Zarr/H5 files to avoid large memory footprint. This method is used for large files. 

