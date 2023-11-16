Frequently Asked Questions (FAQ)
--------------------------------

The `Image.sc Forum <https://forum.image.sc/>`_ is the main discussion channel for BiaPy, hence we recommend to use it for any question or curisity related to it. Use a tag such as ``biapy`` so we can go through your questions. Try to find out if the issue you are having has already been discussed or solved by other people. If not, feel free to create a new topic (please provide a clear and concise description to understand and ideally reproduce the issue you're having). 


Installation
~~~~~~~~~~~~

* If double-clicking the BiaPy binary doesn't initiate the program, attempt starting it via the terminal to display any potential errors. For Linux users encountering a glibc error (something like ``version `GLIBC_2.34' not found``), particularly on Ubuntu ``20.04``, you can try the following: ::

    sudo apt update
    sudo apt install libc6 

  If updating the glibc library to the necessary version (``2.33``) for starting the GUI is unsuccessful, you should consider upgrading to Ubuntu ``22.04``. This upgrade requires a limited number of commands, and there are numerous tutorials available on how to accomplish it. We recommend `this tutorial <https://www.cyberciti.biz/faq/upgrade-ubuntu-20-04-lts-to-22-04-lts/>`_. 

General usage
~~~~~~~~~~~~~

* When using Docker or GUI on Windows, issues can arise with containers accessing network-mounted paths. If you encounter problems where certain paths are not detected, despite being accessible on your machine, consider using local paths instead.

Train questions
~~~~~~~~~~~~~~~

* My training is too slow. What should I do?  

    There are a few things you can do: 1) ensure ``TRAIN.EPOCHS`` and ``TRAIN.PATIENCE`` are set as you want ; 2) increase ``TRAIN.BATCH_SIZE`` ; 3) If you are not loading all the training data in memory, i.e. ``DATA.TRAIN.IN_MEMORY`` is ``False``, try to setting it to speed up the training process.

* I have no enough memory in my computer to set ``DATA.TRAIN.IN_MEMORY``, so I've been using ``DATA.EXTRACT_RANDOM_PATCH``. However, the training process is slow. Also, I need to ensure the entire training image is visited every epoch, not just a random patch extracted from it. What should I do?

    You can previously crop the data into patches of ``DATA.PATCH_SIZE`` you want to work with and disable ``DATA.EXTRACT_RANDOM_PATCH`` because all the images will have same shape. You can use `crop_2D_dataset.py <https://github.com/danifranco/BiaPy/blob/master/utils/scripts/crop_2D_dataset.py>`_ or `crop_3D_dataset.py <https://github.com/danifranco/BiaPy/blob/master/utils/scripts/crop_3D_dataset.py>`_ to crop the data.

Test/Inference questions
~~~~~~~~~~~~~~~~~~~~~~~~

* Test image output is totally black or very bad. No sign of learning seems to be performed. What can I do?

    In order to determine if the model's poor output is a result of incorrect training, it is crucial to first evaluate the training process. One way to do this is to examine the output of the training, specifically the loss and metric values. These values should be decreasing over time, which suggests that the model is learning and improving. Additionally, it is helpful to use the trained model to make predictions on the training data and compare the results to the actual output. This can provide further confirmation that the model has learned from the training data.

    Assuming that the training process appears to be correct, the next step is to investigate the test input image and compare it to the images used during training. The test image should be similar in terms of values and range to the images used during training. If there is a significant discrepancy between the test image and the training images in terms of values or range, it could be a contributing factor to the poor output of the model.

* In the output a kind of grid or squares are appreciated. What can I do to improve the result? 

    Sometimes the model's prediction is worse in the borders of each patch than in the middle. To solve this you can use ``DATA.TEST.OVERLAP`` and ``DATA.TEST.PADDING`` variables. This last especially is designed to remove that `border effect`. E.g. if ``DATA.PATCH_SIZE`` selected is ``(256, 256, 1)``, try setting ``DATA.TEST.PADDING`` to ``(32, 32)`` to remove the jagged prediction effect when reconstructing the final test image. 

* I trained the model and predicted some test data. Now I want to predict more new images, what can I do? 

    You can disable ``TRAIN.ENABLE`` and enable ``MODEL.LOAD_CHECKPOINT``. Those variables will disable training phase and find and load the training checkpoint respectively. Ensure you use the same job name, i.e. ``--name`` option when calling BiaPy, so the library can find the checkpoint that was stored in the job's folder.

* The test images and their labels (if exist) are large and I have no enough memory to make the inference. What can I do?

    You can try setting ``TEST.REDUCE_MEMORY`` which will save as much memory as the library can at the price of slow down the inference process.