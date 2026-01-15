.. _installation:

.. role:: raw-html(raw)
    :format: html

Installation
------------

BiaPy can be installed and run locally on any Linux, Windows, or macOS system using `Docker <https://www.docker.com/>`__, or via the command line using a package manager like `Conda <https://docs.conda.io/projects/conda/en/stable/>`__, `Mamba <https://mamba.readthedocs.io/en/latest/>`__, and `Git <https://git-scm.com/>`__ or `pip <https://pypi.org/project/pip/>`__. Additionally, BiaPy can be seamlessly installed and used on `Google Colab <https://colab.research.google.com/>`__ through our **code-free notebooks**. Each of these approaches is designed to cater to different user experience levels, so choose the installation method that best fits your expertise.

.. image:: ../img/how_to_run.svg
   :width: 100%
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

                Then, the only thing you need to do is **double-click on the BiaPy binary** (application) file you downloaded.
  
                You may run into a message telling you that "BiaPy" is an unrecognized app when running it. This message occurs if an application is unrecognized by Microsoft. In this situation you can click in ``More info`` button and ``Run anyway``. These steps are depicted in the figure below:

                .. figure:: ../img/gui/windows-unrecognized-app.png
                          :align: center                  

                          How to bypass the security error message when executing BiaPy app in Windows.


           .. tab:: Linux  

                You need to install either `Docker Desktop <https://docs.docker.com/desktop/install/linux-install/>`__ (friendlier but not open source) or `Docker Engine <https://docs.docker.com/engine/install/>`__ (open source but command line only).
           
                If you follow the steps and still have problems, you may need to add your user to docker group: ::
                    
                    sudo usermod -aG docker $USER
                    newgrp docker

                To grant execution permission to the binary, enter the following command in a `terminal <faq.html#opening-a-terminal>`__: ::

                    chmod +x BiaPy
                
                Then, simply run it: ::
                    
                    ./BiaPy

           .. tab:: macOS 

                You need to install `Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__. 

                Then, the only thing you need to do is **double-click on the BiaPy binary** (application) file you downloaded.

                In the latest versions of macOS, you may run into a message telling you that *"BiaPy-macOS" can't be opened because Apple cannot check it for malicious software*. In that case, follow these  :ref:`instructions <macos_malicious_error>` to be able to run **BiaPy** in your Mac.

                .. figure:: ../img/gui/macOS-security-error-malicious-software.png
                          :align: center                  
                          :width: 350px

                          **Security error message when executing BiaPy app in macOS**. :raw-html:`<br />` To bypass it, follow these :ref:`instructions <macos_malicious_error>`.

        \ 

        .. note::  
              Whenever you want to run BiaPy's GUI you need to `start Docker Desktop <https://docs.docker.com/desktop/install/windows-install/#start-docker-desktop>`__ first. 

   .. tab:: Google Colab

        No special setup is required other than a browser on your PC. To run any of the BiaPy workflows, simply click the "Open in Colab" button in the "How to run" section of the corresponding workflow configuration page. All available workflows are listed in the menu on the left.

   .. tab:: Galaxy

          BiaPy is available as a tool in the `Galaxy <https://galaxyproject.org/>`__ platform, enabling users to run biomedical image analysis workflows through an intuitive, web-based interface without requiring any local installation. The tool can be accessed directly from the Galaxy ToolShed at this `link <https://imaging.usegalaxy.eu/?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Fbiapy%2Fbiapy%2F3.6.5%2Bgalaxy0&version=latest>`__, and it can also be found by searching for 'biapy' in the Galaxy ToolShed interface here `here <https://usegalaxy.eu/>`__.

   .. tab:: Docker

        We provide a Docker container for running BiaPy called ``biapyx/biapy:latest-11.8`` (`link to the container <https://hub.docker.com/layers/biapyx/biapy/latest-11.8/images/sha256-86cf198ab05a953ba950bb96fb74b18045d2ed7318afb8fa9b212c97c41be904?context=repo>`__.). It is based on Ubuntu ``22.04`` with `Pytorch <https://pytorch.org/get-started/locally/>`__ ``2.4.0`` and CUDA ``11.8`` support.

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

                You need to install either `Docker Desktop <https://docs.docker.com/desktop/install/linux-install/>`__ (friendlier but not open source) or `Docker Engine <https://docs.docker.com/engine/install/>`__ (open source but command line only).
           
                If you follow the steps and still have problems, you may need to add your user to docker group: ::
                    
                    sudo usermod -aG docker $USER
                    newgrp docker

           .. tab:: macOS 

                You need to install `Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__. 

        .. note::  
               Whenever you want to run BiaPy through Docker you need to `start Docker Desktop <https://docs.docker.com/desktop/install/windows-install/#start-docker-desktop>`__ first. 

   .. tab:: CLI

       .. tabs::

          .. tab:: Conda

               .. _installation_command_line_conda:

               To use BiaPy via the command line, you will need to set up a ``conda`` environment. To do this, you will first need to install `Conda <https://docs.conda.io/projects/conda/en/stable/>`__. Then choose one of the following options based on your machine capabilities:
               
               **A. GPU-capable machine (NVIDIA GPU)** ::

                    conda config --set channel_priority strict
                    conda create -n BiaPy_env -c conda-forge python=3.11 biapy pytorch-gpu
                    conda activate BiaPy_env

               Verify GPU at runtime: ::
               
                    python -c 'import torch; print(torch.__version__)'
                    >>> 2.9.1
                    python -c 'import torch; print(torch.cuda.is_available())'
                    >>> True

               **B. CPU-only machine** ::

                    conda config --set channel_priority strict
                    conda create -n BiaPy_env -c conda-forge python=3.11 biapy
                    conda activate BiaPy_env

          .. tab:: Mamba

               .. _installation_command_line_mamba:

               Before you begin, ensure you have `Mamba <https://github.com/mamba-org/mamba>`__ installed. `Mamba <https://github.com/mamba-org/mamba>`__ is a faster alternative to `Conda <https://docs.conda.io/projects/conda/en/stable/>`__ and can be used to manage your ``conda`` environments. Once you have mamba installed you will to choose one of the following options based on your machine capabilities:
               
               **A. GPU-capable machine (NVIDIA GPU)** ::

                    mamba create -n BiaPy_env -c conda-forge python=3.11 biapy pytorch-gpu
                    mamba activate BiaPy_env

               Verify GPU at runtime: ::
               
                    python -c 'import torch; print(torch.__version__)'
                    >>> 2.9.1
                    python -c 'import torch; print(torch.cuda.is_available())'
                    >>> True

               **B. CPU-only machine** ::

                    mamba create -n BiaPy_env -c conda-forge python=3.11 biapy
                    mamba activate BiaPy_env

          .. tab:: Developer

               .. _installation_command_line_dev:

               Set up a conda/mamba environment: ::

                    mamba create -n BiaPy_env -c conda-forge python=3.11
                    mamba activate BiaPy_env
               
               Clone BiaPy repository: :: 

                    git clone https://github.com/BiaPyX/BiaPy.git
               
               Install PyTorch first, choosing GPU if available. Use the official `PyTorch selector <https://pytorch.org/get-started/locally/>`__ for your platform (CUDA / ROCm / CPU). Example (CUDA, just as an example-use the selector’s exact command): ::
               
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
               
               Install BiaPy in editable mode: ::

                    cd BiaPy
                    pip install -e .
                     
   .. tab:: API

        If you want to use BiaPy as a library in your own Python scripts, you can install it via `pip <https://pypi.org/project/pip/>`__: ::
          
            pip install biapy

        or via conda/mamba: ::
          
            conda install -c conda-forge biapy
     
        Once installed you will need to install PyTorch, choosing GPU if available. Use the official `PyTorch selector <https://pytorch.org/get-started/locally/>`__ for your platform (CUDA / ROCm / CPU). Example (CUDA, just as an example—use the selector’s exact command): ::
               
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

        After that you can import BiaPy in your Python scripts: ::

            import biapy

        You can find more information in the following sections:
       
          * `Library examples <https://biapy.readthedocs.io/en/latest/for_developers/library_examples.html>`__ that show how to use BiaPy as a library in your own Python scripts.

          * `API documentation <https://biapy.readthedocs.io/en/latest/for_developers/api.html>`__ for more information on how to use BiaPy as a library in your own Python scripts.  

The next step consists in `selecting the specific workflow <select_workflow.html>`_ that aligns with your intended use.
