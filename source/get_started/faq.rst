FAQ & Troubleshooting
---------------------

The `Image.sc Forum <https://forum.image.sc/>`__ is the main discussion channel for BiaPy, hence we recommend to use it for any question or curiosity related to it. Use a tag such as ``biapy`` so we can go through your questions. Try to find out if the issue you are having has already been discussed or solved by other people. If not, feel free to create a new topic (please provide a clear and concise description to understand and ideally reproduce the issue you're having). 

Frequently asked questions
**************************

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

* When using Docker or BiaPy GUI on Windows, issues can arise with containers accessing network-mounted paths. If you encounter problems where certain paths are not detected, despite being accessible on your machine, consider using local paths instead.

Train questions
~~~~~~~~~~~~~~~
* Can I reuse a previously trained model?

    Yes, you can reuse that model if you have both its weights (``.pth``) and configuration (``.yaml``) files. Here you have a video explaining how to do it in BiaPy's GUI:

    .. raw:: html

        <iframe width="560" height="315" src="https://www.youtube.com/embed/wxahMOKpLKM?si=aU1eNutnVN3NVOQq" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


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
                --gpu "cuda:0,1"

    ``nproc_per_node`` need to be equal to the number of GPUs you are using, 2 in this example.

* My training got stuck in the first epoch without no error. What should I do?  

    Probably the problem is the GPU memory. We experienced, in Windows, that even if the GPU memory gets saturated the operating system doesn't report an out of memory error. Try to decrease the ``TRAIN.BATCH_SIZE`` to ``1`` (you can increase the value later progresively) and reduce the network parameters, e.g. by reducing ``MODEL.FEATURE_MAPS`` if you are using an U-Net like model. You can also reduce the number of levels, e.g. from ``[16, 32, 64, 128, 256]`` to ``[32, 64, 128]``.

* There can be problems with parallel loads in Windows that throw an error as below. To solve that you can set ``SYSTEM.NUM_WORKERS`` to ``0``. In the GUI, you can set it in "general options" window, under "advance options" in the field "Number of workers". 

    .. collapse:: Expand error trace

        .. code-block:: bash

            [12:46:39.363853] #####################
            [12:46:39.363884] #  TRAIN THE MODEL  #
            [12:46:39.363893] #####################
            [12:46:39.363905] Start training in epoch 1 - Total: 100
            [12:46:39.363935] ~~~ Epoch 1/100 ~~~

            Traceback (most recent call last):
            File "/installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1133, in _try_get_data
                data = self._data_queue.get(timeout=timeout)
            File "/installations/miniconda3/envs/BiaPy_env/lib/python3.10/queue.py", line 180, in get
                self.not_empty.wait(remaining)
            File "/installations/miniconda3/envs/BiaPy_env/lib/python3.10/threading.py", line 324, in wait
                gotit = waiter.acquire(True, timeout)
            File "/installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
                _error_if_any_worker_fails()
            RuntimeError: DataLoader worker (pid 1285) is killed by signal: Killed. 

            The above exception was the direct cause of the following exception:

            Traceback (most recent call last):
            File "/installations/BiaPy/main.py", line 51, in <module>
                _biapy.run_job()
            File "/installations/BiaPy/biapy/_biapy.py", line 400, in run_job
                self.train()
            File "/installations/BiaPy/biapy/_biapy.py", line 151, in train
                self.workflow.train()
            File "/installations/BiaPy/biapy/engine/base_workflow.py", line 508, in train
                train_stats = train_one_epoch(self.cfg, model=self.model, model_call_func=self.model_call_func, loss_function=self.loss, 
            File "/installations/BiaPy/biapy/engine/train_engine.py", line 21, in train_one_epoch
                for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            File "/installations/BiaPy/biapy/utils/misc.py", line 413, in log_every
                for obj in iterable:
            File "/installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 631, in __next__
                data = self._next_data()
            File "/installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1329, in _next_data
                idx, data = self._get_data()
            File "/installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1285, in _get_data
                success, data = self._try_get_data()
            File "/installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1146, in _try_get_data
                raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly') from e
            RuntimeError: DataLoader worker (pid(s) 1285) exited unexpectedly
            ERROR conda.cli.main_run:execute(124): `conda run python3 -u /installations/BiaPy/main.py --config /BiaPy_files/input.yaml --result_dir /C/Users/Pille/Desktop/training/BiaPy/U-Net_new --name u-net_test2_df --run_id 1 --dist_backend gloo --gpu "cuda:0"` failed. (See above for error)

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

Troubleshooting
***************

General errors
~~~~~~~~~~~~~~

- In Linux an error like the following may arise: ::

    OSError: [Errno 24] Too many open files

To sort it out increase the number of open files with the command ``ulimit -Sn 10000``. You can check the limits typing ``ulimit -a``. Add it to your ``~/.bashrc`` file to ensure the change it's permanent.


Graphical User interface (GUI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In case you have troubles with BiaPy's GUI, you can find instructions on how to use it in our walkthrough video:

.. raw:: html

        <iframe width="560" height="315" src="https://www.youtube.com/embed/vY7aBh5FUNk?si=yvVolBnu5APNeHwB" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

\

* Running the GUI for the first time:

    * **Windows**: once you donwload the Windows binary an error may arise when running it: ``Windows protected your PC``. This message occurs if an application is unrecognized by Microsoft. In this situation you can click in ``More info`` button and ``Run anyway``.
    
    * **Linux**: once you donwload the Linux binary you need to grant execution permission to it by typing the following command in a `terminal <faq.html#opening-a-terminal>`__: ::

        chmod +x BiaPy

    * **macOS**: you might experience the following error when open the app for the first time:

        .. image:: https://raw.githubusercontent.com/BiaPyX/BiaPy-GUI/main/images/macOS_binary_error.png
            :align: center 

     This is a common situation when opening third-party applications. Apple offers different ways of `turning BiaPy an authorized application <https://support.apple.com/en-us/102445>`__.
     
     In short, you can remove the quarantine attribute through `terminal <faq.html#opening-a-terminal>`__: ::

         xattr -d com.apple.quarantine BiaPy.app  

* When running BiaPy, as it is starting and after downloading you may get the following error: 

    .. code-block:: bash
        
        GPU error docker.errors.APIError: 500 Server Error for http+docker://localhost/v1.46/containers/9ff69069d7627753045d46f9bb4246f56024a937b48746e0708d3499c9f852a5/start: 
        Internal Server Error ("could not select device driver "" with capabilities: [[gpu]]")

  This suggest that the NVIDIA GPU compatibility was not correctly set up (probably the **nvidia container toolkit**). Find the following useful links describing a few steps you can follow: https://github.com/NVIDIA/nvidia-docker/issues/1034 and https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.13.5/install-guide.html 

Limitations
===========

Through the graphical user interface the multi-GPU is not supported. 

