.. _cartocell:

CartoCell, a high-throughput pipeline for accurate 3D image analysis
--------------------------------------------------------------------

This tutorial describes how to train and infer using our custom ResU-Net 3D DNN in order to reproduce the results obtained in ``(Andrés-San Román, 2022)``. Given an initial training dataset of 21 segmented epithelial 3D cysts acquired after confocal microscopy, we follow the CartoCell pipeline (figure below) to high-throughput segment hundreds of cysts at low resolution automatically.

.. figure:: ../img/cartocell_pipeline.png
    :align: center

    CartoCell pipeline for high-throughput epithelial cysts segmentation.  


.. list-table:: 

  * - .. figure:: ../video/cyst_sample.gif
        :align: center
        :scale: 120%

        Cyst raw image   

    - .. figure:: ../video/cyst_instance_prediction.gif 
        :align: center
        :scale: 120%

        Cyst label image


Paper citation: :: 

    CartoCell, a high-throughput pipeline for accurate 3D image analysis, unveils cell 
    morphology patterns in epithelial cysts. Jesús Andrés-San Román, Carmen Gordillo-Vázquez,
    Daniel Franco-Barranco, Laura Morato, Antonio Tagua, Pablo Vicente-Munuera, 
    Ana M. Palacios, María P. Gavilán, Valentina Annese, Pedro Gómez-Gálvez, 
    Ignacio Arganda-Carreras, Luis M. Escudero. [under revision]


Data preparation
~~~~~~~~~~~~~~~~

The data needed is:

* `training_down-sampled_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-dd7044fc-dda2-43a2-9951-cbe6c1851030>`__, `training_down-sampled_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-3e5dded7-24c6-41e3-ab6d-9ca3587c0fbe>`__, `validation_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-83538c77-61d8-4770-85d1-1bac988c5e43>`__ and `validation_dataset_ground_truth <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-5195c7ac-eacd-491e-9d69-8115b36b6c43>`__ to feed the initial model (model M1, `Phase 2`). 

* `low-resolution_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-0506e31c-69f2-445d-80d8-d46b0547d320>`__ to run `Phase 3 – 5` of CartoCell pipeline.

* `test_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-ba6774bd-7858-4bfb-aca9-9ac307e72120>`__ or  `low-resolution_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-0506e31c-69f2-445d-80d8-d46b0547d320>`__  if you just want to run the inference using our pretrained model M2.

We also provide all the properly segmented cysts (ground truth) in `Mendeley <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft>`__.

How to train your model
~~~~~~~~~~~~~~~~~~~~~~~

* **Option 1**: to reproduce the exact results of our manuscript you need to use `cartocell_training.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/cartocell_training.yaml>`__ configuration file that allows to train the model using `command line section <https://biapy.readthedocs.io/en/latest/workflows/instance_segmentation.html#run>`__. 

  * In case you want to reproduce our **model M1, Phase 2**, you will need to modify the ``TRAIN.PATH`` and ``TRAIN.MASK_PATH`` with the paths of `training_down-sampled_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-dd7044fc-dda2-43a2-9951-cbe6c1851030>`__ and `training_down-sampled_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-3e5dded7-24c6-41e3-ab6d-9ca3587c0fbe>`__ respectively. In the same way you will need to modify ``VAL.PATH`` and ``VAL.MASK_PATH`` with `validation_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-83538c77-61d8-4770-85d1-1bac988c5e43>`__ and `validation_dataset_ground_truth <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-5195c7ac-eacd-491e-9d69-8115b36b6c43>`__. 
  
  * In case you want to reproduce our **model M2, Phase 5**, you need to merge `training_down-sampled_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-dd7044fc-dda2-43a2-9951-cbe6c1851030>`__ and `low-resolution_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-0506e31c-69f2-445d-80d8-d46b0547d320>`__ images in a folder and set its path in `TRAIN.PATH``. In the same way you need to merge `training_down-sampled_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-3e5dded7-24c6-41e3-ab6d-9ca3587c0fbe>`__ and `low-resolution_ground_truth <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-fa0564a8-1e55-4c97-b031-843de45b3771>`__ images in a folder and set its path in ``TRAIN.MASK_PATH``. 

  For the validation data, for both **model M1** and **model M2**, you will need to modify ``VAL.PATH`` and ``VAL.MASK_PATH`` with `validation_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-83538c77-61d8-4770-85d1-1bac988c5e43>`__ and `validation_dataset_ground_truth <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-5195c7ac-eacd-491e-9d69-8115b36b6c43>`__. 

* **Option 2**: another alternative is to use a Google Colab `notebook <https://colab.research.google.com/github/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/CartoCell%20-%20Training%20workflow%20(Phase%202).ipynb>`__. Noteworthy, Google Colab standard account do not allow you to run a long number of epochs due to time limitations. Because of this, we set ``100`` epochs to train and patience to ``20`` while the original configuration they are set to ``1300`` and ``100`` respectively. In this case you do not need to donwload any data, as the notebook will do it for you. 

How to run the inference
~~~~~~~~~~~~~~~~~~~~~~~~

* **Option 1**: to reproduce the exact results of our manuscript you need to use `cartocell_inference.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/cartocell_inference.yaml>`__ configuration file that allows to predict new images using `command line section <https://biapy.readthedocs.io/en/latest/workflows/instance_segmentation.html#run>`__. 

 In this case, you will need to set ``TEST.PATH`` and ``TEST.MASK_PATH`` with `test_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-ba6774bd-7858-4bfb-aca9-9ac307e72120>`__ and `test_dataset_ground_truth <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-efddb305-dec1-46e3-b235-00d7cd670e66>`__ data. 

 In case you want to reproduce our **model M2, Phase 5**, you can download our pretained model file `model_weights_cartocell.h5 <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/model_weights_cartocell.h5>`__ and set its path in ``PATHS.CHECKPOINT_FILE``. You can also use your own pretained model in case you followed the training explanation above. 

* **Option 2**: to perform an inference using a pretrained model, you can run a Google Colab `notebook <https://colab.research.google.com/github/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/CartoCell%20-%20Inference%20workflow%20(Phase%205).ipynb>`__. 

Results
~~~~~~~

The results follow same structure as explained in :ref:`instance_segmentation_results`.

                