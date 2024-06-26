.. _installation:

Installation
------------

BiaPy can be installed and run locally on any Linux, Windows, or macOS platform using `Docker <https://www.docker.com/>`__ or via the command line with `Conda <https://docs.conda.io/projects/conda/en/stable/>`__ and `Git <https://git-scm.com/>`__. Alternatively, BiaPy can also be used on `Google Colab <https://colab.research.google.com/>`__. Each of these approaches is designed for different types of experiences and users (select the installation based on your level of expertise).

.. image:: ../img/how_to_run.svg
   :width: 80%
   :align: center 

|

Prerequisites 
~~~~~~~~~~~~~

* Update your `NVIDIA drivers <https://www.nvidia.com/download/index.aspx>`__ for you GPUs in your system. 
* For Docker and graphical user interface (GUI) installations only: you will need to enable Virtualization Technology on your BIOS. Find here a useful `link <https://support.bluestacks.com/hc/en-us/articles/4409279876621-How-to-enable-Virtualization-VT-on-Windows-11-for-BlueStacks-5#%E2%80%9CA%E2%80%9D>`__ to do it. 

Choose your installation method 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::
   .. tab:: GUI

        Download BiaPy GUI for you OS:

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

                        wsl --list -verbose

                  The one with * is the default configuration. So, if it is not Ubuntu, it can be changed by using the command: ::

                        wsl --set-default Ubuntu

                * Install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__. 

                  Check that everything is correct by opening `Docker Desktop` application, going to `Configuration` (wheel icon in the right top corner), in `General` tab the option `WSL 2` should be checked. 
                  
                .. note::  
                  Whenever you want to run BiaPy through Docker you need to `start Docker Desktop <https://docs.docker.com/desktop/install/windows-install/#start-docker-desktop>`__ first. 

           .. tab:: Linux  

                You will need to follow the steps described `here <https://docs.docker.com/desktop/install/linux-install/>`__. 
           
                If you follow the steps and still have problems maybe you need to add your user to docker group: ::
                    
                    sudo usermod -aG docker $USER
                    newgrp docker

                To grant execution permission to the binary, enter the following command in a `terminal <faq.html#opening-a-terminal>`__: ::

                    chmod +x BiaPy

           .. tab:: macOS 

                You need to install `Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__. 

                .. note::  
                  Whenever you want to run BiaPy through Docker you need to `start Docker Desktop <https://docs.docker.com/desktop/install/windows-install/#start-docker-desktop>`__ first. 

        Then, the only thing you need to do is double-click in BiaPy binary downloaded. 

   .. tab:: Google Colab

        Nothing special is needed except a browser on your PC. You can run any of the avaialable workflows in BiaPy through Jupyter notebook using Google Colab by clicking in the "Open in colab" button in each workflow page's "Run" section. You can find all workflows in the left menu. 

   .. tab:: Docker

        We have two containers prepared to run BiaPy, one for the actual NVIDIA driver versions and another container for old drivers: 

            * ``biapyx/biapy:latest-11.8``: Ubuntu ``22.04`` SO with `Pytorch <https://pytorch.org/get-started/locally/>`__ ``2.3.1`` installed supporting CUDA ``11.8`` (`container link <https://hub.docker.com/layers/biapyx/biapy/latest-11.8/images/sha256-86cf198ab05a953ba950bb96fb74b18045d2ed7318afb8fa9b212c97c41be904?context=repo>`__).
            * ``biapyx/biapy:latest-10.2``: Ubuntu ``20.04`` SO with `Pytorch <https://pytorch.org/get-started/locally/>`__ ``1.12.1`` installed supporting CUDA ``10.2`` (`container link <https://hub.docker.com/layers/biapyx/biapy/latest-10.2/images/sha256-c437972cfe30909879085ffd1769666d11875f0ff239df3100fa04ea056d09ab?context=repo>`__).

        You need to check the CUDA version that you NVIDIA driver can handle. You can do that with ``nvidia-smi`` command in Linux/macOS or by running ``NVIDIA Control Panel`` in Windows. The driver information will tell you the maximum CUDA version it can handle. Select one of the above containers depending on your GPU driver. For instance, if the CUDA version it can handle is ``12.0`` you can use ``biapyx/biapy:latest-11.8`` container. 
        
        To install `Docker <https://docs.docker.com/>`__ in your operating system, you can follow these steps:

        .. tabs::

           .. tab:: Windows 

                In Windows you will need to install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__ with Windows Subsystem for Linux (WSL) activated. There is a good video `here <https://www.youtube.com/watch?v=PB7zM3JrgkI>`__. Let's start the installation:

                * Install Ubuntu inside WSL. For that `open PowerShell <faq.html#opening-a-terminal>`__ or Windows Command Prompt in administrator mode by right-clicking and selecting `Run as administrator` and type the following: :: 
                    
                        wsl --install

                  This command will enable the features necessary to run WSL and install the Ubuntu distribution of Linux. Then restart your machine and you can do it again so you can check that it is already installed. 

                  Once the installation ends it will ask for a username and a password. This is not necessary, exit the installation by using **Ctrl+C** or by closing the window.

                  Then you need to make Ubuntu the default Linux distribution. List installed Linux distributions typing: ::

                        wsl --list -verbose

                  The one with * is the default configuration. So, if it is not Ubuntu, it can be changed by using the command: ::

                        wsl --set-default Ubuntu

                * Install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__. 

                  Check that everything is correct by opening `Docker Desktop` application, going to `Configuration` (wheel icon in the right top corner), in `General` tab the option `WSL 2` should be checked. 
                  
                .. note::  
                  Whenever you want to run BiaPy through Docker you need to `start Docker Desktop <https://docs.docker.com/desktop/install/windows-install/#start-docker-desktop>`__ first. 

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

                              # Then install Pytorch 2.3.1 + CUDA 11.8
                              pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 
                         
                    .. tab:: CPU only support

                         :: 

                              # Then install Pytorch 2.3.1 + CUDA 11.8
                              pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu 
                              
               Ultimately, it is necessary to install additional dependencies that rely on the `Pytorch <https://pytorch.org/get-started/locally/>`__ installation; therefore, they must be installed last: ::

                    pip install timm torchmetrics pytorch-msssim

               .. note:: 

                    The PyPI package does not install `Pytorch <https://pytorch.org/get-started/locally/>`__ because there is no option to build that package specifying exactly the CUDA version you want to use. There are a few solutions to set up ``pyproject.toml`` with poetry and specify the CUDA version, as discussed `here <https://github.com/python-poetry/poetry/issues/6409>`__, but then PyPI package can not be built (as stated `here <https://peps.python.org/pep-0440/#direct-references>`__).


          .. tab:: Mamba

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

                    mamba timm torchmetrics

               * Install BiaPy Dependencies: ::
                    
                    mamba install pytz asciitree tzdata typer tqdm torchinfo tifffile threadpoolctl six Shapely scipy ruamel.yaml.clib pyparsing protobuf numcodecs marshmallow lazy_loader kiwisolver joblib imageio h5py fonttools fastremap fasteners cycler contourpy zarr=2.16.1 scikit-learn=1.4.0 scikit-image=0.21.0 ruamel.yaml python-dateutil pydot=1.4.2 marshmallow-union marshmallow-jsonschema pandas matplotlib bioimageio.spec=0.4.9 xarray imgaug bioimageio.core=0.5.9

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

                              # Install Pytorch 2.3.1 + CUDA 11.8
                              pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 
                              pip install timm torchmetrics pytorch-msssim

                    .. tab:: CUDA 10.2

                         ::
                              
                              cd BiaPy
                              pip install --editable .

                              # Install Pytorch 1.12.1 + CUDA 10.2  
                              conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
                              pip install timm torchmetrics pytorch-msssim


     \ 

     Verify installation: ::

          python -c 'import torch; print(torch.__version__)'
          >>> 2.3.1
          python -c 'import torch; print(torch.cuda.is_available())'
          >>> True
          

The next step consists in `select the specific workflow <select_workflow.html>`_ that aligns with your intended use.