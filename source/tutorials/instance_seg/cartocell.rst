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

The target of the workflow are **3D epithelial cysts** acquired with confocal microscopy, whose segmented cells need to be in direct contact to study their packaging and organization.

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

All data needed in this tutorial is accesible through Zenodo `here <https://zenodo.org/records/10973241>`__. Download and unzip the `CartoCell.zip <https://zenodo.org/records/10973241/files/CartoCell.zip?download=1>`__ file (185.7 MB). Once unzipped, you should find the following directory tree: ::

    CartoCell/
    ├── train_M1
    │   ├── x
    │   │   ├── Cyst 4d filt 2po Pha,Bcat,DAPI 02.08.19 40x POC 3 Z6.tif
    │   │   ├── Cyst 4d filt 2po Pha,Bcat,DAPI 02.08.19 40x Z4.5 4a.tif
    │   │   ├── . . .
    │   │   └── cyst 7d filt 3po pha bcat dapi 15.07.19 40x z4.5 4a.tif
    │   └── y
    │       ├── Cyst 4d filt 2po Pha,Bcat,DAPI 02.08.19 40x POC 3 Z6.tif
    │       ├── Cyst 4d filt 2po Pha,Bcat,DAPI 02.08.19 40x Z4.5 4a.tif
    │       ├── . . .
    │       └── cyst 7d filt 3po pha bcat dapi 15.07.19 40x z4.5 4a.tif
    ├── validation
    │   ├── x
    │   │   ├── CYST 7d Filt 3well Pha,Bcat,DAPI 40x Z4 15.7.19 3a.tif
    │   │   └── cyst 4d fil 3well Pha,bcat,dapi 02.08.19 40x Z5 12a.tif
    │   └── y
    │       ├── CYST 7d Filt 3well Pha,Bcat,DAPI 40x Z4 15.7.19 3a.tif
    │       └── cyst 4d fil 3well Pha,bcat,dapi 02.08.19 40x Z5 12a.tif
    ├── train_M2
    │   ├── x
    │   │   ├── 10d.1B.26.2.tif
    │   │   ├── 10d.1B.29.1.tif
    │   │   ├── . . .
    │   │   └── control_7d.3HX3.1HX1.C.9.3.tif
    │   └── y
    │       ├── 10d.1B.26.2.tif
    │       ├── 10d.1B.29.1.tif
    │       ├── . . .
    │       └── control_7d.3HX3.1HX1.C.9.3.tif
    └── test
        ├── x
        │   ├── 10d.1B.10.1.tif
        │   ├── 10d.1B.10.2.tif
        │   ├── . . .
        │   └── 7d.4C.8_2.tif
        └── y
            ├── 10d.1B.10.1.tif
            ├── 10d.1B.10.2.tif
            ├── . . .
            └── 7d.4C.8_2.tif


More specifically, the data you need on each phase is as follows:

* **Phase 2**: folders `train_M1 <https://zenodo.org/records/10973241>`__ (19 volumes) and `validation <https://zenodo.org/records/10973241>`__ (2 volumes) to train the initial model (**model M1**). 

* **Phases 3 and 4**: folder `train_M2 <https://zenodo.org/records/10973241>`__ (293 volumes) to be segmented with **model M1** (phase 3) and then train **model M2** (phase 4).

* **Phase 5**: `test <https://zenodo.org/records/10973241>`__ (122 volumes) to run the inference using our pretrained **model M2** on unseen data.

Reproducing published results (legacy version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**BiaPy**, the library behind **CartoCell**, has undergone many changes since the `CartoCell paper <https://doi.org/10.1016/j.crmeth.2023.100597>`__ was published. Here you have the instructions to reproduce exactly the **CartoCell** pipeline using the same version of **BiaPy** available at the time of publication.

.. note::

  **CartoCell** can also be executed using the latest version of **BiaPy** (see instructions below). These steps are only needed to use the exact same code and configuration used at the time of publication.

Configure environment for old BiaPy version
*******************************************
To reproduce the exact pipeline published with our `manuscript <https://doi.org/10.1016/j.crmeth.2023.100597>`__, you need to configure BiaPy to use the code version associated with the publication. To do so, the easiest way is to configure a **Conda environment** from the command line as follows:

.. code-block:: bash
  
  # Create enviroment called "CartoCell_env" using Python v3.10.11
  conda create -n CartoCell_env python=3.10.11

  # Activate environment
  conda activate CartoCell_env

  # Install dependences
  conda install scikit-image==0.20.0 scikit-learn==1.2.2 tqdm==4.65.0 pandas==1.5.3
  conda install imgaug==0.4.0 yacs==0.1.6 pydot

  pip install fill-voids

  conda install -c conda-forge tensorflow-gpu==2.11.1 edt==2.3.1

Model training
**************

The training of **model M1** and **model M2** is essentially the same, only the input dataset changes. To train either model, you have two options: via **command line** or using **Google Colab**. 

.. tabs::

   .. tab:: Command line

        You can reproduce the exact results of our manuscript via the **command line** using the `cartocell_training.yaml <https://github.com/BiaPyX/BiaPy/blob/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/cartocell_training.yaml>`__ configuration file.

        * In case you want to reproduce the training of our **model M1** (from phase 2), you will need to modify the ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` with the paths to the folders containing the raw images and their corresponding labels, that is to say, with the paths of `train_M1/x <https://zenodo.org/records/10973241>`__ and `train_M1/y <https://zenodo.org/records/10973241>`__ respectively.

        * In case you want to reproduce the training of our **model M2** (from phase 4), you will need to modify the ``DATA.TRAIN.PATH`` and ``DATA.TRAIN.GT_PATH`` as above but now using the paths of `train_M2/x <https://zenodo.org/records/10973241>`__ and `train_M2/y <https://zenodo.org/records/10973241>`__.

        For the validation data, for both **model M1** and **model M2**, you will need to modify ``DATA.VAL.PATH`` and ``DATA.VAL.GT_PATH`` with the paths of `validation/x <https://zenodo.org/records/10973241>`__ and `validation/y <https://zenodo.org/records/10973241>`__, respectively.

        The next step is to `open a terminal <../../get_started/faq.html#opening-a-terminal>`__ and run the code as follows:

        .. code-block:: bash
            
            # Set the full path to CartoCell's training configuration file
            # (replace '/home/user/' with an actual path)
            job_cfg_file=/home/user/cartocell_training.yaml       
            # Set the folder path where results will be saved
            result_dir=/home/user/exp_results
            # Assign a job name to identify this experiment
            job_name=cartocell_training      
            # Set an execution count for tracking repetitions (start with 1)
            job_counter=1
            # Specify the GPU's id to run the job in (according to 'nvidia-smi' command)
            gpu_number=0                   

            # Clone BiaPy's repository (only needed once)
            git clone git@github.com:BiaPyX/BiaPy.git
            # Move to BiaPy's folder
            cd BiaPy
            # Checkout BiaPy's version at the time of publication
            git checkout 2bfa7508c36694e0977fdf2c828e3b424011e4b1

            # Load the environment (created in the previous section)
            conda activate CartoCell_env

            # Run training workflow
            python -u main.py \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu "$gpu_number"  

   .. tab:: Google Colab

        An alternative is to use our **Google Colab** |colablink_train| notebook. Noteworthy, Google Colab standard account do not allow you to run a long number of epochs due to time limitations. Because of this, we set ``50`` epochs to train and patience to ``10`` while the original configuration they are set to ``1300`` and ``100`` respectively. In this case you do not need to donwload any data, as the notebook will do it for you. 

        .. |colablink_train| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://github.com/BiaPyX/BiaPy/blob/master/templates/instance_segmentation/CartoCell_paper/CartoCell%20-%20Training%20workflow%20(Phases%202%20and%204).ipynb

        .. warning::
          This option is **deprecated**, since we don't have control over the versions of the packages installed in Google Colab and there is no way to install the required version of BiaPy through pip (that option was created after the publication of CartoCell).

Model testing
*************
Once trained, the models can be applied to the test image volumes as follows:


.. tabs::

   .. tab:: Command line

        You can reproduce the exact results of our **model M2** (from phase 5), of the manuscript via the **command line** using the `cartocell_inference.yaml <https://github.com/BiaPyX/BiaPy/blob/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/cartocell_inference.yaml>`__ configuration file.

        You will need to set ``DATA.TEST.PATH`` and ``DATA.TEST.GT_PATH`` with the paths to the `test/x <https://zenodo.org/records/10973241>`__ and `test/y <https://zenodo.org/records/10973241>`__ folders. To reproduce our results, you can download the `model_weights_cartocell.h5 <https://github.com/BiaPyX/BiaPy/raw/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/model_weights_cartocell.h5>`__ file, which contains out pretained **model M2**, and set its path in ``PATHS.CHECKPOINT_FILE``. 

        The next step is to `open a terminal <../../get_started/faq.html#opening-a-terminal>`__ and run the code as follows:

        .. code-block:: bash
            
            # Set the full path to CartoCell's inference configuration file
            # (replace '/home/user/' with an actual path)
            job_cfg_file=/home/user/cartocell_inference.yaml       
            # Set the folder path where results will be saved
            result_dir=/home/user/exp_results
            # Assign a job name to identify this experiment
            job_name=cartocell_inference      
            # Set an execution count for tracking repetitions (start with 1)
            job_counter=1
            # Specify the GPU's id to run the job in (according to 'nvidia-smi' command)
            gpu_number=0                    

            # Clone BiaPy's repository (only needed once)
            git clone git@github.com:BiaPyX/BiaPy.git
            # Move to BiaPy's folder
            cd BiaPy
            # Checkout BiaPy's version at the time of publication
            git checkout 2bfa7508c36694e0977fdf2c828e3b424011e4b1

            # Load the environment (created in the previous section)
            conda activate CartoCell_env

            # Run inference workflow
            python -u main.py \
                --config $job_cfg_file \
                --result_dir $result_dir  \ 
                --name $job_name    \
                --run_id $job_counter  \
                --gpu "$gpu_number"  

   .. tab:: Google Colab
    
        As an alternative to perform inference (testing) using a pretrained model, you can run our Google Colab |colablink_inference| notebook. 

        .. |colablink_inference| image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/BiaPyX/BiaPy/blob/ad2f1aca67f2ac7420e25aab5047c596738c12dc/templates/instance_segmentation/CartoCell_paper/CartoCell%20-%20Inference%20workflow%20(Phase%205).ipynb

        .. warning::
          This option is **deprecated**, since we don't have control over the versions of the packages installed in Google Colab and there is no way to install the required version of BiaPy through pip (that option was created after the publication of CartoCell).

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

Citation
~~~~~~~~
Please note that **CartoCell** is based on a publication. If you use it successfully for your research please be so kind to cite our work:

.. code-block:: text
    
    Andres-San Roman, J.A., Gordillo-Vazquez, C., Franco-Barranco, D., Morato, L., 
    Fernandez-Espartero, C.H., Baonza, G., Tagua, A., Vicente-Munuera, P., Palacios, A.M.,
    Gavilán, M.P., Martín-Belmonte, F., Annese, V., Gómez-Gálvez, P., Arganda-Carreras, I.,
    Escudero, L.M. 2023. CartoCell, a high-content pipeline for 3D image analysis, unveils
    cell morphology patterns in epithelia. Cell Reports Methods, 3(10).
    https://doi.org/10.1016/j.crmeth.2023.100597.
