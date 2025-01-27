BioImage Model Zoo
------------------

BiaPy is a Community Partner of the `BioImage Model Zoo (BMZ) <https://bioimage.io/#/>`__ initiative, which means it is capable of consuming, fine-tuning, and exporting models in BMZ format. Depending on how you're using BiaPy, you can access these models in the following ways:

Graphical user interface
~~~~~~~~~~~~~~~~~~~~~~~~

Import
******

In the graphical user interface (GUI), there are two locations where you can explore and **import** compatible models.

* **From the Wizard**:

  #. Specify the image dimensions.
  #. Select the workflow you want to create.
  #. You'll encounter the **"Model source"** prompt. Select the third option: **"Yes, I want to check if there is a pretrained model I can use"**.
  #. This will trigger a follow-up question, prompting you to search for pretrained models, including those from the BioImage Model Zoo. Click on the **"Check models"** button to initiate the search for models compatible with BiaPy. The search process requires an internet connection and may take a few minutes.
  #. Once the search is complete, a **new window** will display a list of models available for use in BiaPy, filtered by the selected workflow and image dimensions.

  Find a depiction of these steps below:

  .. carousel::
    :show_controls:
    :show_captions_below:
    :data-bs-interval: false
    :show_indicators:
    :show_dark:

    .. figure:: ../img/bmz/bmz_gui_step1.png

        Step 0: Specify the path and name of your configuration file

    .. figure:: ../img/bmz/bmz_gui_step2.png

        Step 1: Select your image dimensions (2D or 3D)

    .. figure:: ../img/bmz/bmz_gui_step3.png

        Step 2: Indicate which workflow you want to create

    .. figure:: ../img/bmz/bmz_gui_step4.png

        Step 3: Specify you want to look for pretrained models

    .. figure:: ../img/bmz/bmz_gui_step5.png

        Step 4: Click on "Check models"

    .. figure:: ../img/bmz/bmz_gui_step6.png

        Step 5: Select a model from the list by double-clicking on it
        

* **Outside of the Wizard**:

  #. Go to the **"Generic options"** screen.
  #. Select **"Yes"** for the **"Load pretrained model"** question.
  #. Then choose **"I want to check other online sources"** under **"Source of the model"**. This will activate the **"Model ID"** option, where you will see a button labeled **"Check models"**.
  #. Clicking this button, similar to the Wizard, will begin the search for models compatible with BiaPy. This process requires an internet connection and may take a few minutes. Once complete, a **new window** will show the models available for use in BiaPy based on the selected workflow and image dimensions.
  
  Here you have a depiction of the steps you need to follow:

  .. carousel::
    :show_controls:
    :show_captions_below:
    :data-bs-interval: false
    :show_indicators:
    :show_dark:

    .. figure:: ../img/bmz/bmz_gui_no_wizard_step1.png

        In *General options* > *Checkpoint configuration* > *Source of the model*, select "I want to check other online sources"

    .. figure:: ../img/bmz/bmz_gui_no_wizard_step2.png

        Click on "Check models" and select a model from the pop-up list by double-clicking on it

  You can also check our video tutorial on how to import a BMZ model in the GUI using as example the 2D instance segmentation workflow:

  .. raw:: html

      <iframe width="560" height="315" src="https://www.youtube.com/embed/Zq50Ew1s8ag?si=ejnhoKM8cb83NlQx" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
  

Export
******

Since the Wizard is designed for users without a background in computer science or deep learning, the option to export models in BMZ format is intentionally not included. However, BMZ model exportation is supported through the GUI. To access it, navigate to the **"Generic Options"** screen, you can enable this by selecting **"Yes"** for the question **"Export model to BioImage Model Zoo (BMZ) format?"**. Once selected, you have two options:

.. carousel::
    :show_controls:
    :show_captions_below:
    :data-bs-interval: false
    :show_indicators:
    :show_dark:

    .. figure:: ../img/bmz/bmz_gui_no_wizard_export_opt1.png

        Option 1) Provide the necessary information manually. 

    .. figure:: ../img/bmz/bmz_gui_no_wizard_export_opt2.png

        Option 2) Reuse BMZ model data

* **Option 1: Provide the necessary information to export the model manually.** More specifically, you'll need to input the following metadata of the model (in accordance with the `BioImage.IO Model Resource Description File Specifications <https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/user_docs/model_descr_latest.md>`__):

    * **Model name**: A human-readable name of the model. It should be no longer than 64 characters and may only contain letter, number, underscore, minus, parentheses and spaces. It is recommended to chose a name that refers to the model's task and image modality.

      Examples: 2D-U-Net-Fluorescence-Cell-Segmentation, 3D_UNETR_Mitochondria_Detection.

    * **Description**: A string containing a brief description.
      
      Example: A UNETR-Base model trained to detect the 3D center of mitochondria on electron microscopy images. 

    * **Authors**: The list of authors, i.e., the creators of the model and the primary points of contact. They should be listed as a *sequence* (list of dictionaries in Python) between squared brackets and contain the ``name`` and ``github_user`` keywords.
      
      Example: ``[{"name": "Marie Curie", "github_user": "mcurie"}, {"name": "Pierre Curie", "github_user": "pcurie"}]``.

    * **License**: A `SPDX license identifier <https://spdx.org/licenses/>`__. BMZ does not support custom license beyond the SPDX license list, if you need that please `open a GitHub issue <https://github.com/bioimage-io/spec-bioimage-io/issues/new/choose>`__ to discuss your intentions with the community.
      
      Examples: CC0-1.0, MIT, BSD-2-Clause.

    * **Tags**: Associated tags.  They should be listed as a *sequence* between squared brackets.
      
      Example: ``["unet2d", "pytorch", ""nucleus", "segmentation", "dsb2018"]``.
      
      Notice the quotation marks for each tag.

    * **Citations**: The list of references for the BMZ model. They should be listed as a *sequence* (list of dictionaries in Python) between squared brackets and contain the ``text`` (a free text description) and ``doi`` (a digital object identifier, eee https://www.doi.org/ for details) keywords.
      
      Example: ``[{"text": "training library", "doi": "10.1101/2024.02.03.576026"}, {"text": "architecture", "doi": "10.1109/LGRS.2018.2802944"}, {"text": "data", "doi": "10.48550/arXiv.1812.06024"}]``.

    * **Documentation**: Path to a ``.md`` extension file with the documentation of the model. If it is not set, the model documentation will point to `BiaPy README.md file <https://github.com/BiaPyX/BiaPy/blob/master/README.md>`__. Take other models in https://bioimage.io/#/ as reference.

* **Option 2: Reuse an existing BMZ model**. The model's metadata can be used to export your model. To enable this, set **"Loading pretrained model"** to **"Yes"** and **"Source of the model"** to **"I want to check other online sources"**. Then, select a BMZ model. After this, the option "**Reuse BMZ model configuration"** will appear, allowing you to choose this feature.


After running the workflow and completing the training and/or testing phases, a ZIP file containing the model in BMZ format will be generated. This file will be saved in the results folder, within a directory named **"BMZ_files"**. The file path will also be displayed in the running window.

Jupyter notebooks 
~~~~~~~~~~~~~~~~~

In all notebooks there are two cells prepared to **import and export** models from/to the BioImage Model Zoo:

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

As with the GUI, you can reuse the metadata of a previous BMZ model or input the corresponding metadata manually (with the same fields as described for the GUI). We have also prepared a video tutorial explaining the whole BMZ import/export process using as example the 2D instance segmentation workflow available as a Colab notebook:

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/R0Li3tZ7Ryc?si=HDglCfWxDFONgDlF" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

\

Command line
~~~~~~~~~~~~

Import
******

To use a BMZ model via the command line, you need to set the ``MODEL.SOURCE`` parameter to ``"bmz"`` and specify the model with ``MODEL.BMZ.SOURCE_MODEL_ID``. This field can either be the DOI of the model or its nickname, such as `"affable-shark" <https://bioimage.io/#/?id=10.5281%2Fzenodo.5764892>`__. BiaPy supports consuming models exported in `PyTorch <https://pytorch.org/>`__ using a `PyTorch state dict <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html#:~:text=A%20state_dict%20is%20an%20integral,to%20PyTorch%20models%20and%20optimizers.>`__.

Export
******

To export a model to BMZ format, call the `export_model_to_bmz() <https://github.com/BiaPyX/BiaPy/blob/284ec3838766392c9a333ac9d27b55816a267bb9/biapy/_biapy.py#L219>`__ function. You can find all the instructions for exporting a model in the `export_bmz_test.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/utils/scripts/export_bmz_test.py>`__ script. For example, it can be invoked as follows:

.. code-block:: python

    # Initialize BiaPy 
    biapy = BiaPy(...)
    biapy.run_job() # You can also call .prepare_model(), .train(), or .test() depending on your use case.

    # Create a dictionary with all BMZ requirements
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

    # Export model
    biapy.export_model_to_bmz(output_path, bmz_cfg=bmz_cfg)


If the model was previously imported from BMZ, you have the option to reuse its fields during the export process:

.. code-block:: python

    # Initialize BiaPy 
    biapy = BiaPy(...)
    biapy.run_job() # Or you could call also .prepare_model(), .train() or .test() depending you case

    # Export model, reusing the original BMZ configuration
    biapy.export_model_to_bmz(output_path, reuse_original_bmz_config=True)

Alternatively, you can configure all the required variables through a YAML file. You can review the relevant variables `here <https://github.com/BiaPyX/BiaPy-GUI/blob/49fd4c0116bd8d0414e6a579bb6d98a7acf90d8b/biapy/biapy_config.py#L726>`__.
