FAQ & Troubleshooting
---------------------

The `Image.sc Forum <https://forum.image.sc/>`__ is the main discussion channel for BiaPy, hence we recommend to use it for any question or curisity related to it. Use a tag such as ``biapy`` so we can go through your questions. Try to find out if the issue you are having has already been discussed or solved by other people. If not, feel free to create a new topic (please provide a clear and concise description to understand and ideally reproduce the issue you're having). 

Installation
~~~~~~~~~~~~

* If double-clicking the BiaPy binary doesn't initiate the program, attempt starting it via the `terminal <faq.html#opening-a-terminal>`__ to display any potential errors. For Linux users encountering a glibc error (something like ``version `GLIBC_2.34' not found``), particularly on Ubuntu ``20.04``, you can try the following: ::

    sudo apt update
    sudo apt install libc6 

  If updating the glibc library to the necessary version (``2.33``) for starting the GUI is unsuccessful, you should consider upgrading to Ubuntu ``22.04``. This upgrade requires a limited number of commands, and there are numerous tutorials available on how to accomplish it. We recommend `this tutorial <https://www.cyberciti.biz/faq/upgrade-ubuntu-20-04-lts-to-22-04-lts/>`__. 

Opening a terminal
~~~~~~~~~~~~~~~~~~

To open a terminal on your operating system, you can follow these steps:

* In **Windows**: Click Start, type PowerShell, and then click Windows PowerShell. Alternatively, if you followed the instructions to install git in BiaPy installation, you should have a terminal called ``Git Bash`` installed on your system. To open it, go to the Start menu and search for ``Git Bash``. Click on the application to open it.
* In **macOS**: You already have the Bash terminal installed on your system, so you can simply open it. If you have never used it before, you can find more information `here <https://support.apple.com/en-ie/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac>`__.
* In **Linux**: You already have the Bash terminal installed on your system, so you can simply open it. If you have never used it before, you can find more information `here <https://www.geeksforgeeks.org/how-to-open-terminal-in-linux/>`__.

General usage
~~~~~~~~~~~~~

* When using Docker or GUI on Windows, issues can arise with containers accessing network-mounted paths. If you encounter problems where certain paths are not detected, despite being accessible on your machine, consider using local paths instead.

Train questions
~~~~~~~~~~~~~~~

* My training is too slow. What should I do?  

    There are a few things you can do: 1) ensure ``TRAIN.EPOCHS`` and ``TRAIN.PATIENCE`` are set as you want ; 2) increase ``TRAIN.BATCH_SIZE`` ; 3) If you are not loading all the training data in memory, i.e. ``DATA.TRAIN.IN_MEMORY`` is ``False``, try to setting it to speed up the training process. 

    Furthermore, if you have more than one GPU you could do the training using a multi-GPU setting. For instance, to use GPU ``0`` and ``1`` you could call BiaPy like this:  ::

        python -u -m torch.distributed.run \
                --nproc_per_node=2 \
                main.py \
                --config XXX \
                --result_dir XXX  \ 
                --name XXX    \
                --run_id XXX  \
                --gpu "0,1"

    ``nproc_per_node`` need to be equal to the number of GPUs you are using, 2 in this example.

* I have no enough memory in my computer to set ``DATA.TRAIN.IN_MEMORY``, so I've been using ``DATA.EXTRACT_RANDOM_PATCH``. However, the training process is slow. Also, I need to ensure the entire training image is visited every epoch, not just a random patch extracted from it. What should I do?

    You can previously crop the data into patches of ``DATA.PATCH_SIZE`` you want to work with and disable ``DATA.EXTRACT_RANDOM_PATCH`` because all the images will have same shape. You can use `crop_2D_dataset.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/utils/scripts/crop_2D_dataset.py>`__ or `crop_3D_dataset.py <https://github.com/BiaPyX/BiaPy/blob/master/biapy/utils/scripts/crop_3D_dataset.py>`__ to crop the data.

Test/Inference questions
~~~~~~~~~~~~~~~~~~~~~~~~

* Test image output is totally black or very bad. No sign of learning seems to be performed. What can I do?

    In order to determine if the model's poor output is a result of incorrect training, it is crucial to first evaluate the training process. One way to do this is to examine the output of the training, specifically the loss and metric values. These values should be decreasing over time, which suggests that the model is learning and improving. Additionally, it is helpful to use the trained model to make predictions on the training data and compare the results to the actual output. This can provide further confirmation that the model has learned from the training data.

    Assuming that the training process appears to be correct, the next step is to investigate the test input image and compare it to the images used during training. The test image should be similar in terms of values and range to the images used during training. If there is a significant discrepancy between the test image and the training images in terms of values or range, it could be a contributing factor to the poor output of the model.

* In the output a kind of grid or squares are appreciated. What can I do to improve the result? 

    Sometimes the model's prediction is worse in the borders of each patch than in the middle. To solve this you can use ``DATA.TEST.OVERLAP`` and ``DATA.TEST.PADDING`` variables. This last especially is designed to remove that `border effect`. E.g. if ``DATA.PATCH_SIZE`` selected is ``(256, 256, 1)``, try setting ``DATA.TEST.PADDING`` to ``(32, 32)`` to remove the jagged prediction effect when reconstructing the final test image. 

* I trained the model and predicted some test data. Now I want to predict more new images, what can I do? 

    You can disable ``TRAIN.ENABLE`` and enable ``MODEL.LOAD_CHECKPOINT``. Those variables will disable training phase and find and load the training checkpoint respectively. Ensure you use the same job name, i.e. ``--name`` option when calling BiaPy, so the library can find the checkpoint that was stored in the job's folder.

* The test images, and their labels if exist, are large and I have no enough memory to make the inference. What can I do?

    You can try setting ``TEST.REDUCE_MEMORY`` which will save as much memory as the library can at the price of slow down the inference process. 

    Furthermore, we have an option to use ``TEST.BY_CHUNKS`` option, which will reconstruct each test image using Zarr/H5 files in order to avoid using a large amount of memory. Also, enablign this option Zarr/H5 files can be used as input, to reduce even more the amount of data loaded in memory, as only the patches being processed are loaded into memory one by one and not the entire image. If you have more that one GPU consider using multi-GPU setting to speed up the process. 

    .. warning ::
        Be aware of enabling ``TEST.BY_CHUNKS.SAVE_OUT_TIF`` option as it will require to load the prediction entirely in order to save it.


Graphical User interface (GUI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In case you have troubles with GUI you can find instructions on how to use it in the following video (at 41min51s in the video):

.. image:: https://raw.githubusercontent.com/BiaPyX/BiaPy/master/img/BiaPy_presentation_and_demo_at_RTmfm.jpg
    :alt: BiaPy history and GUI demo
    :target: https://www.youtube.com/watch?v=Gnm-VsZQ6Cc&t=41m51s

Windows 
=======

Once you donwload the Windows binary an error may arise when running it: ``Windows protected your PC``. This message occurs if an application is unrecognized by Microsoft. In this situation you can click in ``More info`` button and ``Run anyway``.

Linux
=====

Once you donwload the Linux binary you need to grant execution permission to it by typing the following command in a `terminal <faq.html#opening-a-terminal>`__: ::

    chmod +x BiaPy

macOS
=====

macOS users might experience the following error when open the app for the first time:

.. image:: https://raw.githubusercontent.com/BiaPyX/BiaPy-GUI/main/images/macOS_binary_error.png
   :align: center 

To sort it, remove the quarantine attribute through `terminal <faq.html#opening-a-terminal>`__: ::

    xattr -d com.apple.quarantine BiaPy.app  


Limitations
===========

Through the graphical user interface the multi-GPU is not supported. 

