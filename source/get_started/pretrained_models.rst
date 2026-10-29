.. _pretrained_models:

Running pretrained models
-------------------------

BiaPy lets you run inference (prediction/test) or fine-tune a model without training it from scratch. There are two sources of pretrained models:

* **BioImage Model Zoo (BMZ)** models, shared by the community through the `BioImage Model Zoo <https://bioimage.io/#/>`__. See the `BioImage Model Zoo page <bmz.html>`__ for more details.
* **BiaPy models**, i.e. models you have previously trained with BiaPy. They are stored as a ``.pth`` checkpoint file in the ``checkpoints`` folder of the job (for example ``<result_dir>/<name>/checkpoints/<name>_1-checkpoint-best.pth``).

.. note::
   The model must be compatible with the workflow you are running (same task, dimensionality, number of classes and channels). BiaPy can only import models exported as a `PyTorch state dictionary <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html>`__. The GUI and the notebooks only offer compatible models.

The variables to set depend on the source of the model:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model source
     - Configuration variables
   * - BiaPy model (``.pth`` file)
     - ``MODEL.SOURCE: biapy``, ``MODEL.LOAD_CHECKPOINT: True`` and ``PATHS.CHECKPOINT_FILE`` set to the path of the model file.
   * - BioImage Model Zoo model
     - ``MODEL.SOURCE: bmz`` and ``MODEL.BMZ.SOURCE_MODEL_ID`` set to the model's nickname (e.g. ``affable-shark``) or DOI.

Select the tab that matches the way you are running BiaPy:

.. tabs::

   .. tab:: GUI

        In the GUI, the Wizard asks whether you want to use a pretrained model. Start the Wizard as described in the `BioImage Model Zoo page <bmz.html#graphical-user-interface>`__ (choose the configuration file, the image dimensions and the workflow), then answer the question **"Do you want to use a pre-trained model?"** with one of these options:

        * **"Yes, I have a model previously trained in BiaPy"**: a new question, *"Please select the pretrained model trained with BiaPy before"*, appears. Select the ``.pth`` file of your model.

        * **"Yes, I want to check if there is a pretrained model I can use"**: click on **"Check models"** to list the BMZ models compatible with the selected workflow and dimensions. Double-click on a model to select it.

        Once the model is selected, the GUI sets the required variables for you. Then, in the **Data > Phases** section, disable the training phase if you only want to predict, and click on **Run Workflow**.

        .. carousel::
            :show_controls:
            :show_captions_below:
            :data-bs-interval: false
            :show_indicators:
            :show_dark:

            .. figure:: ../img/bmz/bmz_gui_step1.png

                Step 0: Specify the path and name of your configuration file and click on "Start".

            .. figure:: ../img/bmz/bmz_gui_step2.png

                Step 1: Answer whether your images are in 3D.

            .. figure:: ../img/bmz/bmz_gui_step3.png

                Step 2: Select your task by answering the wizard.

            .. figure:: ../img/bmz/bmz_gui_step4.png

                Step 3: Specify if you want to look for pretrained models or use a previously trained BiaPy model.

            .. figure:: ../img/bmz/bmz_gui_step5.png

                Step 4: Click on "Check models".

            .. figure:: ../img/bmz/bmz_gui_step6.png

                Step 5: Select a model from the list by double-clicking on it.


        .. note::
           If you are using a BMZ model, the GUI imposes some variables (e.g. number of classes or channels) according to the model. These cannot be changed.

   .. tab:: Google Colab

        The `BiaPy Inference notebook <https://github.com/BiaPyX/BiaPy/blob/master/notebooks/BiaPy_Inference.ipynb>`__ runs a pretrained model on your test images without any coding. Open it in Google Colab and follow these steps:

        #. **Install BiaPy**: run the cell "Play to install BiaPy and its dependences".

        #. **Upload your data**: upload the test images (and, optionally, their ground truth) and your YAML configuration file, then set the paths in the "Path to test images", "Path to test ground truth", "Path to the YAML configuration file" and "Path to store the resulting images" cells. For example, after uploading with option 1, use ``/content/input/test/raw`` as the test images path and ``/content/out`` as the output path.

        #. **Select the model**: in the "Model selection" section, choose one of the following options:

           * **Option 1: Upload custom weights**: upload your ``.pth`` (or ``.safetensors``) file, then set its path in the "Set the path to checkpoint file" cell. For example, ``/content/my_2d_semantic_segmentation_1-checkpoint-best.pth``.
           * **Option 2: Load a pretrained model from the BioImage Model Zoo**: run the "Check BioImage Model Zoo models compatible with BiaPy" cell to list the compatible models, then copy the nickname of the model you want (for example, ``sensible-cat``) and paste it in the "Set the ID/nickname of the BioImage Model Zoo model to load" cell.

        #. **Run the model**: run the "Play to run the model" cell. It updates your YAML file to load the selected model and disables training so that only testing is run.

        #. **Get the results**: the predictions are saved in the output path. Download them with the "Play to download a zip file with all the results" cell.

        .. note:: The notebook sets the model variables in the YAML for you, so you do not need to edit them by hand. The notebook also has an "Option 0" that keeps the model defined in your YAML file, for example a BMZ model already set in ``MODEL.BMZ.SOURCE_MODEL_ID``.

   .. tab:: Docker

        Follow the same steps as in the "How to run" section of your workflow (see `Select your workflow <select_workflow.html>`__), with these changes:

        #. In your configuration (.YAML) file, set the model variables. For a BiaPy model:

           .. code-block:: yaml

               MODEL:
                   SOURCE: biapy
                   LOAD_CHECKPOINT: True
               PATHS:
                   CHECKPOINT_FILE: /home/user/models/my_2d_semantic_segmentation_1-checkpoint-best.pth
               TRAIN:
                   ENABLE: False
               TEST:
                   ENABLE: True

           For a BMZ model, use instead:

           .. code-block:: yaml

               MODEL:
                   SOURCE: bmz
                   BMZ:
                       SOURCE_MODEL_ID: "affable-shark"

           Set ``TRAIN.ENABLE`` to ``True`` instead if you want to fine-tune the loaded model. The Docker command does not accept variables on its command line, so these values must be set in the YAML file.

        #. Mount the model file in the container, so it is accessible at the path given in ``PATHS.CHECKPOINT_FILE``. For example, using the semantic segmentation workflow:

           .. code-block:: bash

               # Configuration file
               job_cfg_file=/home/user/2d_semantic_segmentation.yaml
               # Model file (BiaPy checkpoint). Not needed for BMZ models
               model_file=/home/user/models/my_2d_semantic_segmentation_1-checkpoint-best.pth
               # Path to the data directory
               data_dir=/home/user/data
               # Where the experiment output directory should be created
               result_dir=/home/user/exp_results
               # Just a name for the job
               job_name=my_2d_semantic_segmentation_inference
               # Number that should be increased when one need to run the same job multiple times (reproducibility)
               job_counter=1
               # Number of the GPU to run the job in (according to 'nvidia-smi' command)
               gpu_number=0

               docker run --rm \
                   --gpus "device=$gpu_number" \
                   --mount type=bind,source=$job_cfg_file,target=$job_cfg_file \
                   --mount type=bind,source=$model_file,target=$model_file \
                   --mount type=bind,source=$result_dir,target=$result_dir \
                   --mount type=bind,source=$data_dir,target=$data_dir \
                   biapyx/biapy:latest-11.8 \
                       biapy \
                       --config $job_cfg_file \
                       --result_dir $result_dir \
                       --name $job_name \
                       --run_id $job_counter \
                       --gpu "$gpu_number"

        .. note::
           Remember to set ``DATA.TEST.PATH`` in the configuration file to the folder with the images you want to predict.

   .. tab:: Command line

        Follow the same steps as in the "How to run" section of your workflow (see `Select your workflow <select_workflow.html>`__), with these changes:

        #. In your configuration (.YAML) file, set the model variables as shown in the Docker tab. For a BiaPy model, set ``MODEL.SOURCE: biapy``, ``MODEL.LOAD_CHECKPOINT: True`` and ``PATHS.CHECKPOINT_FILE`` to the path of the model file. For a BMZ model, set ``MODEL.SOURCE: bmz`` and ``MODEL.BMZ.SOURCE_MODEL_ID``. The ``biapy`` command does not accept variables on its command line, so these values must be set in the YAML file.

        #. Run the workflow:

           .. code-block:: bash

               # Configuration file
               job_cfg_file=/home/user/2d_semantic_segmentation.yaml
               # Where the experiment output directory should be created
               result_dir=/home/user/exp_results
               # Just a name for the job
               job_name=my_2d_semantic_segmentation_inference
               # Number that should be increased when one need to run the same job multiple times (reproducibility)
               job_counter=1
               # Number of the GPU to run the job in (according to 'nvidia-smi' command)
               gpu_number=0

               # Load the environment
               conda activate BiaPy_env

               biapy \
                   --config $job_cfg_file \
                   --result_dir $result_dir \
                   --name $job_name \
                   --run_id $job_counter \
                   --gpu "$gpu_number"

   .. tab:: API

        Use the ``BiaPy`` class from Python as described in `Using BiaPy from Python <../for_developers/python_api.html>`__. The model is loaded with ``BiaPy.load_workflow_from_model()``, which builds a workflow ready to test or predict.

        **BiaPy model** (``.pth`` file). The configuration is embedded in the checkpoint, so no YAML file is needed:

        .. code-block:: python

            from biapy import BiaPy

            biapy = BiaPy.load_workflow_from_model(
                "/home/user/exp_results/my_2d_semantic_segmentation/checkpoints/my_2d_semantic_segmentation_1-checkpoint-best.pth",
                result_dir="/home/user/exp_results",
                name="my_2d_semantic_segmentation_inference",
                gpu="0",
            )

            # Predict all the images in a folder and write the results to result_dir
            biapy.update_config({"TEST.ENABLE": True, "DATA.TEST.PATH": "/home/user/data/test/x"})
            biapy.test()

        **Predict a single image in memory**. Omit ``result_dir`` to write nothing to disk. The result is returned as a NumPy array:

        .. code-block:: python

            from skimage.io import imread
            from biapy import BiaPy

            biapy = BiaPy.load_workflow_from_model("affable-shark")  # BMZ model nickname or DOI
            image = imread("/home/user/data/test/x/image_001.tif")
            mask = biapy.predict(image)

        To fine-tune a loaded model instead, enable the training phase and set the training data with ``update_config()``, then call ``biapy.train()``. See `Fine-tune a trained model <../for_developers/python_api.html#fine-tune-a-trained-model>`__ for details.

        .. note::
           ``.safetensors`` checkpoints do not include the configuration, so they cannot be loaded this way. Use a ``.pth`` file instead.

Further information
*******************

* `BioImage Model Zoo <bmz.html>`__: how to import and export BMZ models in each execution method.
* `FAQ & Troubleshooting <faq.html>`__: answers to the most common questions about importing pretrained models.
* `Configuration <../for_developers/configuration.html>`__: all the configuration variables of BiaPy.
