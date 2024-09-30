BioImage Model Zoo
------------------

BiaPy is a Community Partner of the `BioImage Model Zoo (BMZ) <https://bioimage.io/#/>`__ initiative, meaning that it is prepared to consume, finetune and export models into BMZ format. Depending on how are you using BiaPy you can access these models in the following ways. 

Graphical user interface
~~~~~~~~~~~~~~~~~~~~~~~~

Through the graphical user interface there are two places where you can check which models are compatible and **import** them.  

* **Through the Wizard**, after answering the first two questions where the user selects the workflow to execute and the image dimensions to work with, they will reach the **"Model source"** question. Here, they can choose the third option, **"Yes, I want to check if there is a pretrained model I can use"**, which will activate a subsequent question to check for pretrained models, including those from the BioImage Model Zoo. At this point, the user should click the **"Check models"** button to begin searching for models compatible with BiaPy. This process requires an internet connection and may take a few minutes. Once completed, a **new window** will display the models available for use in BiaPy, based on the selected workflow and image dimensions. See below the steps you need to follow:

.. carousel::
    :show_controls:
    :data-bs-interval: false
    :show_indicators:
    :show_dark:

    .. image:: ../img/bmz/bmz_gui_step1.png

    .. image:: ../img/bmz/bmz_gui_step2.png

    .. image:: ../img/bmz/bmz_gui_step3.png

    .. image:: ../img/bmz/bmz_gui_step4.png

    .. image:: ../img/bmz/bmz_gui_step5.png

    .. image:: ../img/bmz/bmz_gui_step6.png

* **Outside of the Wizard**, you can choose a pretrained BMZ model in the **"Generic options"** screen. To do this, select **"Yes"** for the **"Load pretrained model"** question, then choose **"I want to check other online sources"** under **"Source of the model"**. This will activate the **"Model ID"** option, where you will see a button labeled **"Check models"**. Clicking this button, similar to the Wizard, will begin the search for models compatible with BiaPy. This process requires an internet connection and may take a few minutes. Once complete, a **new window** will show the models available for use in BiaPy based on the selected workflow and image dimensions. Follow the steps below to proceed:

.. carousel::
    :show_controls:
    :data-bs-interval: false
    :show_indicators:
    :show_dark:

    .. image:: ../img/bmz/bmz_gui_no_wizard_step1.png

    .. image:: ../img/bmz/bmz_gui_no_wizard_step2.png

Since the Wizard is designed for users without a background in computer science or deep learning, the option to export models in BMZ format is intentionally not included. However, BMZ model exportation is supported through the GUI. To access it, navigate to the **"Generic Options"** screen, you can enable this by selecting **"Yes"** for the question **"Export model to BioImage Model Zoo (BMZ) format?"**. Once selected, you have two options:

#. Provide the necessary information to export the model manually.

#. If you are using an existing BMZ model, its data can be utilized to create a new model. To enable this, set **"Loading pretrained model"** to **"Yes"** and **"Source of the model"** to **"I want to check other online sources"**. Then, select a BMZ model. After this, the option "**Reuse BMZ model configuration"** will appear, allowing you to choose this feature.

The following options are available in the GUI:

.. carousel::
    :show_controls:
    :show_captions_below:
    :data-bs-interval: false
    :show_indicators:
    :show_dark:

    .. figure:: ../img/bmz/bmz_gui_no_wizard_export_opt1.png

        Option 1) Provide the necessary information manually

    .. figure:: ../img/bmz/bmz_gui_no_wizard_export_opt2.png

        Option 2) Reuse BMZ model data

After running the workflow and completing the training and/or testing phases, a ZIP file containing the model in BMZ format will be generated. This file will be saved in the results folder, within a directory named **"BMZ_files"**. The file path will also be displayed in the running window.

Jupyter notebooks 
~~~~~~~~~~~~~~~~~

In all notebooks there are two cells prepared to import and export models from/to BioImage Model Zoo.

.. carousel::
    :show_controls:
    :show_captions_below:
    :data-bs-interval: false
    :show_indicators:
    :show_dark:

    .. figure:: ../img/bmz/bmz_notebook_cell.png
        
        Import model from BioImage Model Zoo

    .. figure:: ../img/bmz/bmz_notebook_cell_export.png
        
        Export model to BioImage Model Zoo format


Command line
~~~~~~~~~~~~

To use a BMZ model via command line you would need to configure ``MODEL.SOURCE`` as ``"bmz"`` and use ``MODEL.BMZ.SOURCE_MODEL_ID`` to select the model. This last field could be the DOI of the model or its nickname, e.g. `"affable-shark" <https://bioimage.io/#/?id=10.5281%2Fzenodo.5764892>`__. BiaPy can consume models exported in `Pytorch <https://pytorch.org/>`__ with a `pytorch state dict <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html#:~:text=A%20state_dict%20is%20an%20integral,to%20PyTorch%20models%20and%20optimizers.>`__. 

For exporting a model to BMZ format you would need to call `export_model_to_bmz() <https://github.com/BiaPyX/BiaPy/blob/284ec3838766392c9a333ac9d27b55816a267bb9/biapy/_biapy.py#L219>`__ function. In `export_bmz_test.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/utils/scripts/export_bmz_test.py>`__ script you will find all the instructions to export a model. For example, can be called as follows:

.. code-block:: python

    # Call BiaPy 
    biapy = BiaPy(...)
    biapy.run_job() # Or you could call also .prepare_model(), .train() or .test() depending on your case

    # Create a dict with all BMZ requirements
    bmz_cfg = {}
    bmz_cfg["description"] = "Mitochondria segmentation for electron microscopy"
    bmz_cfg["authors"] = [{"name": "Daniel Franco", "github_user": "danifranco"}]
    bmz_cfg["license"] = "CC-BY-4.0"
    bmz_cfg["tags"] = ["electron-microscopy", "mitochondria"]
    bmz_cfg["cite"] = [
        {"text": "training library", "doi": "10.1101/2024.02.03.576026"},
        {"text": "architecture", "doi": "10.1109/LGRS.2018.2802944"},
        {"text": "data", "doi": "10.48550/arXiv.1812.06024"},
        ]
    bmz_cfg["doc"] = args["doc_file"]
    bmz_cfg["model_name"] = args["model_name"]

    biapy.export_model_to_bmz(output_path, bmz_cfg=bmz_cfg)

If the model you used was previously imported from BMZ you would have the option to reuse its fields during the model exportation:

.. code-block:: python

    # Call BiaPy 
    biapy = BiaPy(...)
    biapy.run_job() # Or you could call also .prepare_model(), .train() or .test() depending you case

    # Create a dict with all BMZ requirements
    biapy.export_model_to_bmz(output_path, reuse_original_bmz_config=True)

Apart from the options above, you can also configure all required variables through the YAML file. You can have a look to these variables `here <https://github.com/BiaPyX/BiaPy-GUI/blob/49fd4c0116bd8d0414e6a579bb6d98a7acf90d8b/biapy/biapy_config.py#L726>`__. 