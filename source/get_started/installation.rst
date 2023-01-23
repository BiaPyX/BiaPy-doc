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

This will create a folder called ``BiaPy`` that contains all the files of the `library's official repository <https://github.com/danifranco/BiaPy>`__. Then you need to create a ``conda`` environment using the file located in `BiaPy/utils/env/environment.yml <https://github.com/danifranco/BiaPy/blob/master/utils/env/environment.yml>`__ ::
    
    conda env create -f BiaPy/utils/env/environment.yml


Docker installation
~~~~~~~~~~~~~~~~~~~

To run BiaPy using Docker, you need to install it first `here <https://docs.docker.com/get-docker/>`__.

Google Colab
~~~~~~~~~~~~

Nothing special is needed except a browser on your PC.

