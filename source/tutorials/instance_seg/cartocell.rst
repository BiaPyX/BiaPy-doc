.. _cartocell:

CartoCell, a high-throughput pipeline for accurate 3D image analysis (Paper)
----------------------------------------------------------------------------

Task overview
~~~~~~~~~~~~~

This tutorial describes how to create a custom 3D instance segmentation workflow to reproduce the results published in `"CartoCell, a high-content pipeline for 3D image analysis, unveils cell morphology patterns in epithelia" (Cell Report Methods, 2023) <https://doi.org/10.1016/j.crmeth.2023.100597>`__ using **BiaPy**.

.. figure:: https://ars.els-cdn.com/content/image/1-s2.0-S2667237523002497-fx1_lrg.jpg
    :align: center                  
    :width: 350px

    Graphical abstract of `CartoCell (2023) <https://doi.org/10.1016/j.crmeth.2023.100597>`__.

The target of the workflow are 3D epithelial cysts acquired with confocal microscopy, whose segmented cells need to be in direct contact to study their packaging and organization.

.. list-table:: 
  :align: center
  :width: 680px

  * - .. figure:: ../../video/cyst_sample.gif
        :align: center
        :scale: 120%

        Example of cyst raw image (`CartoCell dataset <https://zenodo.org/records/10973241>`__).

    - .. figure:: ../../video/cyst_instance_prediction.gif 
        :align: center
        :scale: 120%

        Corresponding cyst label image (`CartoCell dataset <https://zenodo.org/records/10973241>`__).


Please note that **CartoCell** is based on a publication. If you use it successfully for your research please be so kind to cite our work:

.. code-block:: text
    
    Andres-San Roman, J.A., Gordillo-Vazquez, C., Franco-Barranco, D., Morato, L., 
    Fernandez-Espartero, C.H., Baonza, G., Tagua, A., Vicente-Munuera, P., Palacios, A.M.,
    Gavilán, M.P., Martín-Belmonte, F., Annese, V., Gómez-Gálvez, P., Arganda-Carreras, I.,
    Escudero, L.M. 2023. CartoCell, a high-content pipeline for 3D image analysis, unveils
    cell morphology patterns in epithelia. Cell Reports Methods, 3(10).
    https://doi.org/10.1016/j.crmeth.2023.100597.


CartoCell overview
~~~~~~~~~~~~~~~~~~

**CartoCell** follows a multi-phase pipeline to, given an initial training dataset of 21 3D labeled cysts, automatically segment hundreds of cysts at low resolution with enough quality to pergorm cell organization and packaging analysis. The five phases of **CartoCell** are briefly explained in the following tabs:

.. tabs::

   .. tab:: Phase 1

        A small dataset of 21 cysts, stained with cell outlines markers, was acquired at **high-resolution** in a confocal microscope. Next, the individual cell instances were **semi-automatically segmented and manually curated**. The high-resolution images from Phase 1 provide the accurate and realistic set of data necessary for the following steps.

      .. figure:: ../../img/tutorials/instance-segmentation/cartocell/cartocell-phase-1.png
        :align: center                  
        :width: 350px

   .. tab:: Phase 2

        Both high-resolution raw and label images were **down-sampled to create our initial training dataset**. Specifically, image volumes were reduced to match the resolution of the images acquired in Phase 3. Using that dataset, a first 3D residual U-Net model (*ResU-Net* for short) was trained. We will refer to this first model as **model M1**.

      .. figure:: ../../img/tutorials/instance-segmentation/cartocell/cartocell-phase-2.png
        :align: center                  
        :width: 600px

   .. tab:: Phase 3

        A large number of low-resolution stacks of multiple epithelial cysts was acquired. This was a key step to allow the high-throughput analysis of samples since it greatly reduces acquisition time. Here, we extracted the single-layer and single-lumen cysts by cropping them from the complete stack. This way, we obtained a set of **293 low-resolution images**, composed of 84 cysts at 4 days, 113 cysts at 7 days and 96 cysts at 10 days. Next, we applied our trained **model M1** to those images and post-processed their output to produce (i) a prediction of individual cell instances (obtained by marker-controlled watershed), and (ii) a prediction of the mask of the full cellular regions. At this stage, the output cell instances were generally not touching each other, which is a problem to study cell connectivity in epithelia. Therefore, we applied a 3D **Voronoi algorithm** to correctly mimic the epithelial packing. More specifically, each prediction of cell instances was used as a Voronoi seed, while the prediction of the mask of the cellular region defined the bounding territory that each cell could occupy. The result of this phase was a large dataset of low-resolution images and their corresponding accurate labels.

      .. figure:: ../../img/tutorials/instance-segmentation/cartocell/cartocell-phase-3.png
        :align: center                  
        :width: 680px

   .. tab:: Phase 4

        A new 3D ResU-Net model (**model M2**, from now on) was trained on the newly produced large dataset of low-resolution images and its paired label images. This was a crucial step, since the performance of deep learning models is highly dependent on the amount of training samples.

      .. figure:: ../../img/tutorials/instance-segmentation/cartocell/cartocell-phase-4.png
        :align: center                  
        :width: 350px

   .. tab:: Phase 5

        Finally, **model M2** was applied to new low-resolution cysts and their output was post-processed as in Phase 3, thus achieving high-throughput segmentation of the desired cysts.

      .. figure:: ../../img/tutorials/instance-segmentation/cartocell/cartocell-phase-5.png
        :align: center                  
        :width: 580px


Data preparation
~~~~~~~~~~~~~~~~

The data is accesible through Zenodo `here <https://zenodo.org/records/10973241>`__. The data you need on each phase is:

* `train_M1 <https://zenodo.org/records/10973241>`__ and `validation <https://zenodo.org/records/10973241>`__ to feed the initial model (`model M1`, `Phase 2`). 

* `train_M2 <https://zenodo.org/records/10973241>`__ to run `Phase 3 – 5` of CartoCell pipeline.

* `test <https://zenodo.org/records/10973241>`__ if you just want to run the inference using our pretrained `model M2`.

How to train your model
~~~~~~~~~~~~~~~~~~~~~~~

You have two options to train your model: via **command line** or using **Google Colab**. 

.. tabs::

   .. tab:: Command line

        You can reproduce the exact results of our manuscript via **command line** using `cartocell_training.yaml <https://github.com/BiaPyX/BiaPy/blob/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/cartocell_training.yaml>`__ configuration file.

        * In case you want to reproduce our **model M1, Phase 2**, you will need to modify the ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` with the raw image and their corresponding labels, that is to say, with the paths of `train_M1/x <https://zenodo.org/records/10973241>`__ and `train_M1/y <https://zenodo.org/records/10973241>`__ respectively.

        * In case you want to reproduce our **model M2, Phase 4**, you will need to modify the ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` as above but now using the paths of `train_M2/x <https://zenodo.org/records/10973241>`__ and `train_M2/y <https://zenodo.org/records/10973241>`__.

        For the validation data, for both **model M1** and **model M2**, you will need to modify ``DATA.VAL.PATH`` and ``DATA.VAL.GT_PATH`` with the raw image and their corresponding labels, that is to say, with the paths of `validation/x <https://zenodo.org/records/10973241>`__ and `validation/y <https://zenodo.org/records/10973241>`__ respectively.

        The next step is to `open a terminal <../../get_started/faq.html#opening-a-terminal>`__ and run the code as follows:

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
            git clone git@github.com:BiaPyX/BiaPy.git
            cd BiaPy
            git checkout 2bfa7508c36694e0977fdf2c828e3b424011e4b1

            # Load the environment
            conda activate BiaPy_env

            python -u main.py \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu "$gpu_number"  

   .. tab:: Google Colab

        Another alternative is to use a **Google Colab** |colablink_train|. Noteworthy, Google Colab standard account do not allow you to run a long number of epochs due to time limitations. Because of this, we set ``50`` epochs to train and patience to ``10`` while the original configuration they are set to ``1300`` and ``100`` respectively. In this case you do not need to donwload any data, as the notebook will do it for you. 

        .. |colablink_train| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/CartoCell%20-%20Training%20workflow%20(Phase%202).ipynb

How to run the inference
~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Command line

        You can reproduce the exact results of our **model M2, Phase 5**, of the manuscript via **command line** using `cartocell_inference.yaml <https://github.com/BiaPyX/BiaPy/blob/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/cartocell_inference.yaml>`__ configuration file.

        You will need to set ``DATA.TEST.PATH`` and ``DATA.TEST.GT_PATH`` with `test/x <https://zenodo.org/records/10973241>`__ and `test/y <https://zenodo.org/records/10973241>`__ data. You will need to download `model_weights_cartocell.h5 <https://github.com/BiaPyX/BiaPy/raw/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/model_weights_cartocell.h5>`__ file, which is the pretained model, and set its path in ``PATHS.CHECKPOINT_FILE``. 

        The next step is to `open a terminal <../../get_started/faq.html#opening-a-terminal>`__ and run the code as follows:

        .. code-block:: bash
            
            # Configuration file
            job_cfg_file=/home/user/cartocell_inference.yaml       
            # Where the experiment output directory should be created
            result_dir=/home/user/exp_results  
            # Just a name for the job
            job_name=cartocell_inference      
            # Number that should be increased when one need to run the same job multiple times (reproducibility)
            job_counter=1
            # Number of the GPU to run the job in (according to 'nvidia-smi' command)
            gpu_number=0                   

            # Move where BiaPy installation resides (if you didn't in the previous steps)
            git clone git@github.com:BiaPyX/BiaPy.git
            cd BiaPy
            git checkout 2bfa7508c36694e0977fdf2c828e3b424011e4b1

            # Load the environment
            conda activate BiaPy_env

            python -u main.py \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu "$gpu_number"  

   .. tab:: Google Colab
    
        To perform an inference using a pretrained model, you can run a Google Colab |colablink_inference|. 

        .. |colablink_inference| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/CartoCell%20-%20Inference%20workflow%20(Phase%205).ipynb

Results
~~~~~~~

Following the example, the results should be placed in ``/home/user/exp_results/cartocell/results``. You should find the following directory tree: ::

    cartocell/
    ├── config_files/
    |   ├── cartocell_training.yaml 
    │   └── cartocell_inference.yaml                                                                                                           
    ├── checkpoints
    │   └── model_weights_cartocell_1.h5
    └── results
        └── cartocell_1
            ├── aug
            │   └── .tif files
            ├── charts
            │   ├── cartocell_1_jaccard_index.png
            │   ├── cartocell_1_loss.png
            │   └── model_plot_cartocell_1.png
            ├── per_image
            │   └── .tif files
            ├── per_image_instances
            │   └── .tif files  
            ├── per_image_instances_voronoi
            │   └── .tif files                          
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

