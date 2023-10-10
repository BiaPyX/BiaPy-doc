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


CartoCell phases
~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Phase 1

        A small dataset of 21 cysts, stained with cell outlines markers, was acquired at high-resolution in a confocal microscope. Next, the individual cell instances were segmented. The high-resolution images from Phase 1 provides the accurate and realistic set of data necessary for the following steps.

   .. tab:: Phase 2

        Both high-resolution raw and label images were down-sampled to create our initial training dataset. Specifically, image volumes were reduced to match the resolution of the images acquired in Phase 3. Using that dataset, a first DNN was trained. We will refer to this first model as `model M1`.

   .. tab:: Phase 3

        A large number of low-resolution stacks of multiple epithelial cysts was acquired. This was a key step to allow the high-throughput analysis of samples since it greatly reduces acquisition time. Here, we extracted the single-layer and single-lumen cysts by cropping them from the complete stack. This way, we obtained a set of 293 low-resolution images, composed of 84 cysts at 4 days, 113 cysts at 7 days and 96 cysts at 10 days. Next, we applied our trained `model M1` to those images and post-processed their output to produce (i) a prediction of individual cell instances (obtained by marker-controlled watershed), and (ii) a prediction of the mask of the full cellular regions. At this stage, the output cell instances were generally not touching each other, which is a problem to study cell connectivity in epithelia. Therefore, we applied a 3D Voronoi algorithm to correctly mimic the epithelial packing. More specifically, each prediction of cell instances was used as a Voronoi seed, while the prediction of the mask of the cellular region defined the bounding territory that each cell could occupy. The result of this phase was a large dataset of low-resolution images and their corresponding accurate labels.

   .. tab:: Phase 4

        A new 3D ResU-Net model (`model M2`, from now on) was trained on the newly produced large dataset of low-resolution images and its paired label images. This was a crucial step, since the performance of deep learning models is highly dependent on the amount of training samples.

   .. tab:: Phase 5

        `model M2` was applied to new low-resolution cysts and their output was post-processed as in Phase 3, thus achieving high-throughput segmentation of the desired cysts. 


Data preparation
~~~~~~~~~~~~~~~~

The data needed is:

* `training_down-sampled_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-dd7044fc-dda2-43a2-9951-cbe6c1851030>`__, `training_down-sampled_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-3e5dded7-24c6-41e3-ab6d-9ca3587c0fbe>`__, `validation_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-83538c77-61d8-4770-85d1-1bac988c5e43>`__ and `validation_dataset_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-5195c7ac-eacd-491e-9d69-8115b36b6c43>`__ to feed the initial model (`model M1`, `Phase 2`). 

* `low-resolution_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-0506e31c-69f2-445d-80d8-d46b0547d320>`__ to run `Phase 3 – 5` of CartoCell pipeline.

* `test_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-ba6774bd-7858-4bfb-aca9-9ac307e72120>`__ or  `low-resolution_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-0506e31c-69f2-445d-80d8-d46b0547d320>`__  if you just want to run the inference using our pretrained `model M2`.

We also provide all the properly segmented cysts (ground truth) in `Mendeley <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft>`__.

How to train your model
~~~~~~~~~~~~~~~~~~~~~~~

You have two options to train your model: via **command line** or using **Google Colab**. 

.. tabs::

   .. tab:: Command line

        You can reproduce the exact results of our manuscript via **command line** using `cartocell_training.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/cartocell_training.yaml>`__ configuration file.

        * In case you want to reproduce our **model M1, Phase 2**, you will need to modify the ``TRAIN.PATH`` and ``TRAIN.GT_PATH`` with the paths of `training_down-sampled_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-dd7044fc-dda2-43a2-9951-cbe6c1851030>`__ and `training_down-sampled_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-3e5dded7-24c6-41e3-ab6d-9ca3587c0fbe>`__ respectively.

        * In case you want to reproduce our **model M2, Phase 4**, you need to merge `training_down-sampled_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-dd7044fc-dda2-43a2-9951-cbe6c1851030>`__ and Phase 3 (`model M1`) output in a folder and set its path in ``TRAIN.PATH``. In the same way you need to merge `training_down-sampled_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-3e5dded7-24c6-41e3-ab6d-9ca3587c0fbe>`__ and `low-resolution_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-fa0564a8-1e55-4c97-b031-843de45b3771>`__ images in a folder and set its path in ``TRAIN.GT_PATH``. 

        For the validation data, for both **model M1** and **model M2**, you will need to modify ``VAL.PATH`` and ``VAL.GT_PATH`` with `validation_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-83538c77-61d8-4770-85d1-1bac988c5e43>`__ and `validation_dataset_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-5195c7ac-eacd-491e-9d69-8115b36b6c43>`__. 

        The next step is to open a terminal (see :ref:`installation` section if you need help) and run the code as follows:

        .. code-block:: bash
            
            # Configuration file
            job_cfg_file=/home/user/cartocell_training.yaml       
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results  
            # Just a name for the job
            job_name=cartocell_training      
            # Number that should be increased when one need to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0                   

            # Move where BiaPy installation resides
            cd BiaPy

            # Load the environment
            conda activate BiaPy_env
            source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

            python -u main.py \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu $gpu_number  

   .. tab:: Google Colab

        Another alternative is to use a **Google Colab** |colablink_train|. Noteworthy, Google Colab standard account do not allow you to run a long number of epochs due to time limitations. Because of this, we set ``50`` epochs to train and patience to ``10`` while the original configuration they are set to ``1300`` and ``100`` respectively. In this case you do not need to donwload any data, as the notebook will do it for you. 

        .. |colablink_train| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/CartoCell%20-%20Training%20workflow%20(Phase%202).ipynb

How to run the inference
~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Command line

        You can reproduce the exact results of our **model M2, Phase 5**, of the manuscript via **command line** using `cartocell_inference.yaml <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/cartocell_inference.yaml>`__ configuration file.

        You will need to set ``TEST.PATH`` and ``TEST.GT_PATH`` with `test_dataset_raw_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-ba6774bd-7858-4bfb-aca9-9ac307e72120>`__ and `test_dataset_label_images <https://data.mendeley.com/v1/datasets/7gbkxgngpm/draft#folder-efddb305-dec1-46e3-b235-00d7cd670e66>`__ data. You will need to download `model_weights_cartocell.h5 <https://github.com/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/model_weights_cartocell.h5>`__ file, which is the pretained model, and set its path in ``PATHS.CHECKPOINT_FILE``. 


   .. tab:: Google Colab
    
        To perform an inference using a pretrained model, you can run a Google Colab |colablink_inference|. 

        .. |colablink_inference| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/danifranco/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/CartoCell%20-%20Inference%20workflow%20(Phase%205).ipynb

Results
~~~~~~~

Following the example, the results should be placed in ``/home/user/exp_results/cartocell/results``. You should find the following directory tree: ::

    cartocell/
    ├── config_files/
    |   ├── cartocell_training.yaml 
    │   └── cartocell_inference.yaml                                                                                                           
    ├── checkpoints
    │   └── model_weights_cartocell_1.h5
    └── results
        └── cartocell_1
            ├── aug
            │   └── .tif files
            ├── charts
            │   ├── cartocell_1_jaccard_index.png
            │   ├── cartocell_1_loss.png
            │   └── model_plot_cartocell_1.png
            ├── per_image
            │   └── .tif files
            ├── per_image_instances
            │   └── .tif files  
            ├── per_image_instances_voronoi
            │   └── .tif files                          
            └── watershed
                ├── seed_map.tif
                ├── foreground.tif                
                └── watershed.tif


* ``config_files``: directory where the .yaml filed used in the experiment is stored. 

    * ``cartocell_training.yaml``: YAML configuration file used for training. 

    * ``cartocell_inference.yaml``: YAML configuration file used for inference. 

* ``checkpoints``: directory where model's weights are stored.

    * ``model_weights_cartocell_1.h5``: model's weights file.

* ``results``: directory where all the generated checks and results will be stored. There, one folder per each run are going to be placed.

    * ``cartocell_1``: run 1 experiment folder. 

        * ``aug``: image augmentation samples.

        * ``charts``:  

             * ``cartocell_1_jaccard_index.png``: IoU (jaccard_index) over epochs plot (when training is done).

             * ``cartocell_1_loss.png``: loss over epochs plot (when training is done). 

             * ``model_plot_cartocell_1.png``: plot of the model.

        * ``per_image``:

            * ``.tif files``: reconstructed images from patches.   

        * ``per_image_instances``: 

            * ``.tif files``: same as ``per_image`` but with the instances.

        * ``per_image_post_processing``: 

            * ``.tif files``: same as ``per_image_instances`` but applied Voronoi, which has been the unique post-proccessing applied here. 

        * ``watershed``: 
                
                * ``seed_map.tif``: initial seeds created before growing. 
                
                * ``foreground.tif``: foreground mask area that delimits the grown of the seeds.
                
                * ``watershed.tif``: result of watershed.

