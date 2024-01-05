.. _installation:

Installation
------------

BiaPy can be installed and run locally on any Linux, Windows, or macOS platform using `Docker <https://www.docker.com/>`__ or via the command line with Anaconda/Miniconda and Git. Alternatively, BiaPy can also be used on `Google Colab <https://colab.research.google.com/>`__. Each of these approaches is designed for different types of experiences and users (select the installation based on your level of expertise).

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

   .. tab:: Command line

        .. _installation_command_line:

        To use BiaPy via the command line, you will need to set up a ``conda`` environment. To do this, you will first need to install `Anaconda/Miniconda <https://www.anaconda.com/>`__. For detailed installation instructions based on your operating system, please see the following links: `Windows <https://docs.anaconda.com/anaconda/install/windows/>`__, `macOS <https://docs.anaconda.com/anaconda/install/mac-os/>`__ and `Linux <https://docs.anaconda.com/anaconda/install/linux/>`__. 

        In addition, you will also need to install  `git <https://git-scm.com/>`__, a free and open source distributed version control system. Git will allow you to easily download the code with a single command. You can download and install it `here <https://git-scm.com/downloads>`__. For detailed installation instructions based on your operating system, please see the following links: `Windows <https://git-scm.com/download/win>`__, `macOS <https://git-scm.com/download/mac>`__ and `Linux <https://git-scm.com/download/linux>`__. 

        Once you have installed Anaconda and git, you will need to use a terminal to complete the following steps. To open a terminal on your operating system, you can follow these steps:

        * In **Windows**: If you followed the instructions above to install git, you should have a terminal called ``Git Bash`` installed on your system. To open it, go to the Start menu and search for ``Git Bash``. Click on the application to open it.
        * In **macOS**: You already have the Bash terminal installed on your system, so you can simply open it. If you have never used it before, you can find more information `here <https://support.apple.com/en-ie/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac>`__.
        * In **Linux**: You already have the Bash terminal installed on your system, so you can simply open it. If you have never used it before, you can find more information `here <https://www.geeksforgeeks.org/how-to-open-terminal-in-linux/>`__.

        Then, you are prepared to download `BiaPy <https://github.com/danifranco/BiaPy>`__ repository by running this command in the terminal ::

            git clone https://github.com/danifranco/BiaPy.git

        This will create a folder called ``BiaPy`` that contains all the files of the `library's official repository <https://github.com/danifranco/BiaPy>`__. Then you need to create a ``conda`` environment and install the dependencies.

        You need to check the CUDA version that you NVIDIA driver can handle. You can do that with ``nvidia-smi`` command in Linux/macOS or by running ``NVIDIA Control Panel`` in Windows. The driver information will tell you the maximum CUDA version it can handle. We here provide two stable installations, one based in CUDA ``11.8`` and another one with an older version of Pytorch and with CUDA ``10.2`` (BiaPy will work anyway). Once you have checked it, proceed with the installation depending on the CUDA version: 

        .. tabs::

           .. tab:: CUDA 11.8

                ::

                    # Create and activate the environment
                    conda create -n BiaPy_env python=3.8
                    conda activate BiaPy_env

                    # Option 1: Install the dependecies directly via pip
                    pip install biapy

                    # Option 2: Install Pytorch and GPU dependencies    
                    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
                    cd BiaPy
                    pip install --editable .

           .. tab:: CUDA 10.2

                ::

                    # Create and activate the environment
                    conda create -n BiaPy_env python=3.8
                    conda activate BiaPy_env
                        
                    # Install Pytorch and GPU dependencies    
                    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

                    # Move to BiaPy folder and install the rest of dependecies
                    cd BiaPy
                    pip install --editable .

        Verify installation: ::

            python -c 'import torch; print(torch.__version__)'
            >>> 2.1.0
            python -c 'import torch; print(torch.cuda.is_available())'
            >>> True
            
        From now on, to run BiaPy you will need to just activate the environment: ::

            conda activate BiaPy_env

        The next step consists in `select the specific workflow <select_workflow.html>`_ that aligns with your intended use.


   .. tab:: Docker

        We have two container prepared to run BiaPy, one for the actual NVIDIA driver versions and another container for old drivers: 

            * ``danifranco/biapy:latest-11.8``: Ubuntu ``22.04`` SO with Pytorch ``2.1`` installed supporting CUDA ``11.8``.
            * ``danifranco/biapy:latest-10.2``: Ubuntu ``20.04`` SO with Pytorch ``1.12.1`` installed supporting CUDA ``10.2``.

        You need to check the CUDA version that you NVIDIA driver can handle. You can do that with ``nvidia-smi`` command in Linux/macOS or by running ``NVIDIA Control Panel`` in Windows. The driver information will tell you the maximum CUDA version it can handle. Select one of the above containers depending on your GPU driver. For instance, if the CUDA version it can handle is ``12.0`` you can use ``danifranco/biapy:latest-11.8`` container. 
        
        To install `Docker <https://docs.docker.com/>`__ in your operating system, you can follow these steps:

        .. tabs::

           .. tab:: Windows 

                In Windows you will need to install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__ with Windows Subsystem for Linux (WSL) activated. There is a good video `here <https://www.youtube.com/watch?v=PB7zM3JrgkI>`__. Let's start the installation:

                * Install Ubuntu inside WSL. For that open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting `Run as administrator` and type the following: :: 
                    
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

        The next step consists in `select the specific workflow <select_workflow.html>`_ that aligns with your intended use.


   .. tab:: Google Colab

        Nothing special is needed except a browser on your PC. You can run any of the avaialable workflows in BiaPy through Jupyter notebook using Google Colab by clicking in the "Open in colab" button in each workflow page's "Run" section. You can find all workflows in the left menu. 

        The next step consists in `select the specific workflow <select_workflow.html>`_ that aligns with your intended use.

   .. tab:: GUI

        Download BiaPy GUI for you OS:

        - `Windows 64-bit <https://github.com/danifranco/BiaPy-GUI/raw/main/dist-win/BiaPy.exe>`__ 
        - `Linux 64-bit <https://github.com/danifranco/BiaPy-GUI/raw/main/dist-linux/BiaPy>`__ 
        - `macOS 64-bit <https://github.com/danifranco/BiaPy-GUI/raw/main/dist-macOS/BiaPy-macOS.zip>`__
        
        Then, to use the GUI you will need to install `Docker <https://docs.docker.com/>`__ in your operating system. You can follow these steps:

        .. tabs::

           .. tab:: Windows 

                In Windows you will need to install `Docker Desktop <https://docs.docker.com/desktop/install/windows-install/>`__ with Windows Subsystem for Linux (WSL) activated. There is a good video on how you can do it `here <https://www.youtube.com/watch?v=PB7zM3JrgkI>`__. Manually, the steps are these:

                * Install Ubuntu inside WSL. For that open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting `Run as administrator` and type the following: :: 
                    
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

                To grant execution permission to the binary, enter the following command in a terminal: ::

                    chmod +x BiaPy

           .. tab:: macOS 

                You need to install `Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__. 

                .. note::  
                  Whenever you want to run BiaPy through Docker you need to `start Docker Desktop <https://docs.docker.com/desktop/install/windows-install/#start-docker-desktop>`__ first. 

        Then the only thing you need to do is double-click in BiaPy binary you downloaded. 