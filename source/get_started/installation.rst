.. _installation:

Installation
------------

BiaPy can be installed and run locally on any Linux, Windows, or macOS system using `Docker <https://www.docker.com/>`__, or via the command line using a package manager like `Conda <https://docs.conda.io/projects/conda/en/stable/>`__, `Mamba <https://mamba.readthedocs.io/en/latest/>`__, and `Git <https://git-scm.com/>`__ or `pip <https://pypi.org/project/pip/>`__. Additionally, BiaPy can be seamlessly installed and used on `Google Colab <https://colab.research.google.com/>`__ through our **code-free notebooks**. Each of these approaches is designed to cater to different user experience levels, so choose the installation method that best fits your expertise.

.. image:: ../img/how_to_run.svg
   :width: 80%
   :align: center

|

Prerequisites 
~~~~~~~~~~~~~

- If your system has one or more GPUs, ensure your `NVIDIA drivers <https://www.nvidia.com/download/index.aspx>`__ are up to date.
- For Docker and graphical user interface (GUI) installations: Virtualization Technology must be enabled in your BIOS. Follow this `guide <https://support.bluestacks.com/hc/en-us/articles/4409279876621-How-to-enable-Virtualization-VT-on-Windows-11-for-BlueStacks-5#%E2%80%9CA%E2%80%9D>`__ for instructions on how to enable it.


Choose your installation method 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::
   .. tab:: GUI

        Download the corresponding BiaPy GUI for you OS:

        - `Windows 64-bit <https://drive.google.com/uc?export=download&id=1iV0wzdFhpCpBCBgsameGyT3iFyQ6av5o>`__ 
        - `Linux 64-bit <https://drive.google.com/uc?export=download&id=13jllkLTR6S3yVZLRdMwhWUu7lq3HyJsD>`__ 
        - `macOS 64-bit <https://drive.google.com/uc?export=download&id=1fIpj9A8SWIN1fABEUAS--DNhOHzqSL7f>`__

        Then, to use the GUI you will need to install `Docker <https://docs.docker.com/>`__ in your operating system. You can follow these steps:

        .. tabs::

           .. tab:: Windows 

                In Windows you will need to install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__ with Windows Subsystem for Linux (WSL) activated. There is a good video on how you can do it `here <https://www.youtube.com/watch?v=PB7zM3JrgkI>`__. Manually, the steps are these:

                * Install Ubuntu inside WSL. For that `open PowerShell <faq.html#opening-a-terminal>`__ or Windows Command Prompt in administrator mode by right-clicking and selecting `Run as administrator` and type the following: :: 
                    
                        wsl --install

                  This command will enable the features necessary to run WSL and install the Ubuntu distribution of Linux. Then restart your machine and you can do it again so you can check that it is already installed. 

                  Once the installation ends it will ask for a username and a password. This is not necessary, exit the installation by using **Ctrl+C** or by closing the window.

                  Then you need to make Ubuntu the default Linux distribution. List installed Linux distributions typing: ::

                        wsl --list --verbose

                  The one with * is the default configuration. So, if it is not Ubuntu, it can be changed by using the command: ::

                        wsl --set-default Ubuntu

                * Install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__.

                  After installation, verify that Docker Desktop is properly configured:
                    
                    - Open the Docker Desktop application.

                    - Navigate to `Configuration` (gear icon in the top-right corner).

                    - Under the `General` tab, ensure the option for `WSL 2` is enabled.
              
                  \

                  .. tip:: If you're using a GPU, check the official documentation on `GPU support in Docker Desktop <https://docs.docker.com/desktop/gpu/>`__ for additional setup instructions.

           .. tab:: Linux  

                You will need to follow the steps described `here <https://docs.docker.com/desktop/install/linux-install/>`__. 
           
                If you follow the steps and still have problems maybe you need to add your user to docker group: ::
                    
                    sudo usermod -aG docker $USER
                    newgrp docker

                To grant execution permission to the binary, enter the following command in a `terminal <faq.html#opening-a-terminal>`__: ::

                    chmod +x BiaPy

           .. tab:: macOS 

                You need to install `Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__. 

        Then, the only thing you need to do is double-click on the BiaPy binary (application) file you downloaded.

        .. note::  
               Whenever you want to run BiaPy's GUI you need to `start Docker Desktop <https://docs.docker.com/desktop/install/windows-install/#start-docker-desktop>`__ first. 

   .. tab:: Google Colab

        No special setup is required other than a browser on your PC. To run any of the BiaPy workflows, simply click the "Open in Colab" button in the "How to run" section of the corresponding workflow configuration page. All available workflows are listed in the menu on the left.

   .. tab:: Docker

        We provide two Docker containers for running BiaPy, one compatible with current NVIDIA driver versions and another for older drivers:

            * ``biapyx/biapy:latest-11.8``: Based on Ubuntu ``22.04`` with `Pytorch <https://pytorch.org/get-started/locally/>`__ ``2.4.0`` and CUDA ``11.8`` support. `Link to container <https://hub.docker.com/layers/biapyx/biapy/latest-11.8/images/sha256-86cf198ab05a953ba950bb96fb74b18045d2ed7318afb8fa9b212c97c41be904?context=repo>`__.
            * ``biapyx/biapy:latest-10.2``: Based on Ubuntu ``20.04`` with `Pytorch <https://pytorch.org/get-started/locally/>`__ ``1.12.1`` and CUDA ``10.2`` support. `Link to container <https://hub.docker.com/layers/biapyx/biapy/latest-10.2/images/sha256-c437972cfe30909879085ffd1769666d11875f0ff239df3100fa04ea056d09ab?context=repo>`__.

        To determine the appropriate container for your system, check which CUDA version your NVIDIA driver supports. You can do this by running the command ``nvidia-smi`` in Linux/macOS, or by using the ``NVIDIA Control Panel`` in Windows. The driver information will indicate the maximum CUDA version supported. Choose the container accordingly. For example, if your driver supports CUDA ``12.0``, use the ``biapyx/biapy:latest-11.8`` container. 
        
        To install `Docker <https://docs.docker.com/>`__ in your operating system, you can follow these steps:

        .. tabs::

           .. tab:: Windows 

               To run BiaPy on Windows, you'll need to install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__ with Windows Subsystem for Linux (WSL) enabled. You can follow this helpful video tutorial `here <https://www.youtube.com/watch?v=PB7zM3JrgkI>`__. Below are the steps to get started: 

               #. Install Ubuntu inside WSL:

                  * `Open PowerShell <faq.html#opening-a-terminal>`__ or the Windows Command Prompt in administrator mode by right-clicking and selecting `Run as administrator`.
                  
                  * Run the following command:

                       .. code-block:: bash
                            
                            wsl --install

                       This command will enable the necessary features to run WSL and install the Ubuntu Linux distribution. After running the command, restart your machine. You can then run the command again to confirm that Ubuntu has been installed.
                       
                       During the installation, you may be prompted to create a username and password. This step is not necessary for our purposes; you can exit the installation by pressing **Ctrl+C** or simply closing the window.

               #. Set Ubuntu as the default Linux distribution:

                  * To check which Linux distributions are installed, type:
                
                       .. code-block:: bash

                            wsl --list --verbose

                  * The default distribution is marked with an asterisk (*). If Ubuntu is not set as the default, you can change it by running:
                
                       .. code-block:: bash

                            wsl --set-default Ubuntu

               #. Install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__.

                  After installation, verify that Docker Desktop is properly configured:
                    
                    - Open the Docker Desktop application.

                    - Navigate to `Configuration` (gear icon in the top-right corner).

                    - Under the `General` tab, ensure the option for `WSL 2` is enabled.
              
                  \

                  .. tip:: If you're using a GPU, check the official documentation on `GPU support in Docker Desktop <https://docs.docker.com/desktop/gpu/>`__ for additional setup instructions.

           .. tab:: Linux  

                You will need to follow the steps described `here <https://docs.docker.com/desktop/install/linux-install/>`__. 
           
                If you follow the steps and still have problems maybe you need to add your user to docker group: ::
                    
                    sudo usermod -aG docker $USER
                    newgrp docker

           .. tab:: macOS 

                You need to install `Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__. 

        .. note::  
               Whenever you want to run BiaPy through Docker you need to `start Docker Desktop <https://docs.docker.com/desktop/install/windows-install/#start-docker-desktop>`__ first. 

   .. tab:: Command line

       .. tabs::

          .. tab:: Conda + pip

               .. _installation_command_line_condapip:

               To use BiaPy via the command line, you will need to set up a ``conda`` environment. To do this, you will first need to install `Conda <https://docs.conda.io/projects/conda/en/stable/>`__. Then you need to create a ``conda`` environment through a `terminal <faq.html#opening-a-terminal>`__: ::

                    # Create and activate the environment
                    conda create -n BiaPy_env python=3.10
                    conda activate BiaPy_env

               Then you will need to install `BiaPy package <https://pypi.org/project/biapy/>`__: ::

                    pip install biapy

               Afterwards you need to install `Pytorch <https://pytorch.org/get-started/locally/>`__:
               
               .. tabs::

                    .. tab:: GPU support

                         :: 

                              # Then install Pytorch 2.4.0 + CUDA 11.8
                              pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118 
                         
                    .. tab:: CPU only support

                         :: 

                              # Then install Pytorch 2.4.0 + CUDA 11.8
                              pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu 
                              
               Ultimately, it is necessary to install additional dependencies that rely on the `Pytorch <https://pytorch.org/get-started/locally/>`__ installation; therefore, they must be installed last: ::

                    pip install timm pytorch-msssim torchmetrics[image]

               .. note:: 

                    The PyPI package does not install `Pytorch <https://pytorch.org/get-started/locally/>`__ because there is no option to build that package specifying exactly the CUDA version you want to use. There are a few solutions to set up ``pyproject.toml`` with poetry and specify the CUDA version, as discussed `here <https://github.com/python-poetry/poetry/issues/6409>`__, but then PyPI package can not be built (as stated `here <https://peps.python.org/pep-0440/#direct-references>`__).


          .. tab:: Mamba + pip

               .. _installation_command_line_mamba:

               * Before you begin, ensure you have `Mamba <https://github.com/mamba-org/mamba>`__ installed. `Mamba <https://github.com/mamba-org/mamba>`__ is a faster alternative to `Conda <https://docs.conda.io/projects/conda/en/stable/>`__ and can be used to manage your ``conda`` environments. Install ``mamba`` in the base ``conda`` environment, allowing you to use it across all your environments.
               
               .. tabs::

                    .. tab:: Option 1

                         Download `the miniforge installer <https://github.com/conda-forge/miniforge#mambaforge>`__ specific to your OS and run it. 

                    .. tab:: Option 2

                         If you have ``conda`` already installed: ::

                              conda install mamba -n base -c conda-forge

               * Create a new `Conda <https://docs.conda.io/projects/conda/en/stable/>`__ environment with Python 3.10: ::

                    mamba create -n BiaPy_env python=3.10
                    mamba activate BiaPy_env

               * Now you need to install `Pytorch <https://pytorch.org/get-started/locally/>`__ and related packages. Double check `Pytorch's official page <https://pytorch.org/get-started/locally/>`__ for its specific installation. For example, to install the last version of `Pytorch <https://pytorch.org/get-started/locally/>`__ with ``conda`` installation in Windows OS under cuda 12.1: ::

                    mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

                 Alternatively, for macOS it would be like this: ::

                    mamba install pytorch::pytorch torchvision torchaudio -c pytorch

               * Then, add extra pytorch related packages: ::

                    mamba install timm torchmetrics

               * Install BiaPy Dependencies: ::
                    
                    mamba install pytz asciitree tzdata typer tqdm torchinfo tifffile threadpoolctl
                    mamba install six Shapely scipy ruamel.yaml.clib pyparsing protobuf numcodecs lazy_loader kiwisolver
                    mamba install joblib h5py fonttools fastremap fasteners cycler contourpy zarr=2.16.1 scikit-learn=1.4.0
                    mamba install scikit-image ruamel.yaml python-dateutil pydot=1.4.2 pandas matplotlib xarray imgaug
                    mamba install bioimageio.spec bioimageio.core=0.6.7

               * Install packages not available on conda-forge, so install it via pip: ::
                    
                    pip install fill-voids pytorch_msssim opencv-python==4.8.0.76 opencv-python-headless imagecodecs==2024.1.1 numpy==1.25.2 pooch==1.8.1 tensorboardX==2.6.2.2 yacs==0.1.8 edt==2.3.2

               * Install BiaPy: ::

                    pip install --no-deps biapy

          .. tab:: Developer

               .. _installation_command_line_dev:

               Set up a ``conda`` environment first by installing `Conda <https://docs.conda.io/projects/conda/en/stable/>`__. Then create the environment : ::

                    # Create and activate the environment
                    conda create -n BiaPy_env python=3.10
                    conda activate BiaPy_env
               
               To clone the repository you will need to install `Git <https://git-scm.com/>`__, a free and open source distributed version control system. `Git <https://git-scm.com/>`__ will allow you to easily download the code with a single command. You can download and install it `here <https://git-scm.com/downloads>`__. For detailed installation instructions based on your operating system, please see the following links: `Windows <https://git-scm.com/download/win>`__, `macOS <https://git-scm.com/download/mac>`__ and `Linux <https://git-scm.com/download/linux>`__. 

               Once you have installed Anaconda and `Git <https://git-scm.com/>`__, you will need to t, you will need to `open a terminal <open-terminal.html>`__ to complete the following steps. Then, you are prepared to download `BiaPy <https://github.com/BiaPyX/BiaPy>`__ repository by running this command in the `terminal <faq.html#opening-a-terminal>`__ : :: 

                    git clone https://github.com/BiaPyX/BiaPy.git

               This will create a folder called ``BiaPy`` that contains all the files of the `library's official repository <https://github.com/BiaPyX/BiaPy>`__. Then you need to create a ``conda`` environment and install the dependencies.

               You need to check the CUDA version that you NVIDIA driver can handle. You can do that with ``nvidia-smi`` command in Linux/macOS or by running ``NVIDIA Control Panel`` in Windows. The driver information will tell you the maximum CUDA version it can handle. We here provide two stable installations, one based in CUDA ``11.8`` and another one with an older version of `Pytorch <https://pytorch.org/get-started/locally/>`__ and with CUDA ``10.2`` (BiaPy will work anyway). Once you have checked it, proceed with the installation depending on the CUDA version: 

               .. tabs::

                    .. tab:: CUDA 11.8

                         ::

                              cd BiaPy
                              pip install --editable .

                              # Install Pytorch 2.4.0 + CUDA 11.8
                              pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118 
                              pip install timm pytorch-msssim torchmetrics[image]

                    .. tab:: CUDA 10.2

                         ::
                              
                              cd BiaPy
                              pip install --editable .

                              # Install Pytorch 1.12.1 + CUDA 10.2  
                              conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
                              pip install timm pytorch-msssim torchmetrics[image]


     \ 

     Verify installation: ::

          python -c 'import torch; print(torch.__version__)'
          >>> 2.4.0
          python -c 'import torch; print(torch.cuda.is_available())'
          >>> True
          

The next step consists in `selecting the specific workflow <select_workflow.html>`_ that aligns with your intended use.
