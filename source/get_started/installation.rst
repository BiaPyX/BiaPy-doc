.. _installation:

Installation
------------

BiaPy can be installed and run locally on any Linux, Windows, or Mac OS platform using `Docker <https://www.docker.com/>`__ or via the command line with Anaconda/Miniconda and Git.  Alternatively, BiaPy can also be used on `Google Colab <https://colab.research.google.com/>`__.


.. _installation_command_line:

Command line installation
~~~~~~~~~~~~~~~~~~~~~~~~~

To use BiaPy via the command line, you will need to set up a ``conda`` environment. To do this, you will first need to install `Anaconda/Miniconda <https://www.anaconda.com/>`__. For detailed installation instructions based on your operating system, please see the following links: `Windows <https://docs.anaconda.com/anaconda/install/windows/>`__, `macOS <https://docs.anaconda.com/anaconda/install/mac-os/>`__ and `Linux <https://docs.anaconda.com/anaconda/install/linux/>`__. 

In addition, you will also need to install  `git <https://git-scm.com/>`__, a free and open source distributed version control system. Git will allow you to easily download the code with a single command. You can download and install it `here <https://git-scm.com/downloads>`__. For detailed installation instructions based on your operating system, please see the following links: `Windows <https://git-scm.com/download/win>`__, `macOS <https://git-scm.com/download/mac>`__ and `Linux <https://git-scm.com/download/linux>`__. 

Once you have installed Anaconda and git, you will need to use a terminal to complete the following steps. To open a terminal on your operating system, you can follow these steps:

* In **Windows**: If you followed the instructions above to install git, you should have a terminal called ``Git Bash`` installed on your system. To open it, go to the Start menu and search for ``Git Bash``. Click on the application to open it.
* In **macOS**: You already have the Bash terminal installed on your system, so you can simply open it. If you have never used it before, you can find more information `here <https://support.apple.com/en-ie/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac>`__.
* In **Linux**: You already have the Bash terminal installed on your system, so you can simply open it. If you have never used it before, you can find more information `here <https://www.geeksforgeeks.org/how-to-open-terminal-in-linux/>`__.

Then, you are prepared to download `BiaPy <https://github.com/danifranco/BiaPy>`__ repository by running this command in the terminal ::

    git clone https://github.com/danifranco/BiaPy.git

This will create a folder called ``BiaPy`` that contains all the files of the `library's official repository <https://github.com/danifranco/BiaPy>`__. Then you need to create a ``conda`` environment and install the dependencies ::
    
    # Create and activate the environment
    conda create -n BiaPy_env python=3.8
    conda activate BiaPy_env
        
    # Install Pytorch and GPU dependencies    
    pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
    pip install --editable . 
    

Verify installation: ::

    python -c 'import torch; print(torch.__version__)'
    >>> 1.12.1+cu102
    python -c 'import torch; print(torch.cuda.is_available())'
    >>> True
    
From now on, to run BiaPy you will need to just activate the environment: ::

    conda activate BiaPy_env

.. note:: 
    In this installation CUDA 10.2 is installed but if your machine does not support this version, check how you can see it with ``nvidia-smi`` command in the next section, you can find older versions `here <https://pytorch.org/get-started/previous-versions/>`__. 

The next step consist in `select the specific workflow <select_workflow.html>`_ that aligns with your intended use.

Docker installation
~~~~~~~~~~~~~~~~~~~

Currently the docker image support only CUDA version drivers above ``10.2.0``. To check your actual driver version you can type the following command in the terminal (note ``CUDA Version: 11.8`` in this example in the top right corner): ::

    $ nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 520.61.05    Driver Version: 520.61.05    CUDA Version: 11.8     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  On   | 00000000:1C:00.0 Off |                  N/A |
    | 30%   39C    P8    24W / 350W |      5MiB / 24576MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      2104      G   /usr/lib/xorg/Xorg                  4MiB |
    +-----------------------------------------------------------------------------+

To install `Docker <https://docs.docker.com/>`__ in your operating system, you can follow these steps:

* In **Windows**: You can install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__. Whenever you wan to run BiaPy though Docker you need to start Docker Desktop. 

* In **Linux**: You will need to follow the steps described `here <https://docs.docker.com/desktop/install/linux-install/>`__. 

If you follow the steps and still have problems maybe you need to add your user to docker group: ::
    
    sudo usermod -aG docker $USER
    newgrp docker

* In **macOS**: You can install `Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__. Whenever you wan to run BiaPy though Docker you need to start Docker Desktop. 

The next step consist in `select the specific workflow <select_workflow.html>`_ that aligns with your intended use.

Google Colab
~~~~~~~~~~~~

Nothing special is needed except a browser on your PC. You can run any of the avaialable workflows in BiaPy through Jupyter notebook using Google Colab by clicking in the "Open in colab" button in each workflow page's "Run" section. You can find all workflows in the left menu. 

The next step consist in `select the specific workflow <select_workflow.html>`_ that aligns with your intended use.
