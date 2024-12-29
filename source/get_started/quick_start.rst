.. _quick_start:

Quick start
-----------

Install & Run
*************

To quickly get started with BiaPy, follow these steps:

* **Step 1**: Follow the `installation instructions <installation.html>`__ to set up BiaPy.

* **Step 2**: `Select the specific workflow <select_workflow.html>`__ you wish to use.

* **Step 3**: Refer to the workflow-specific execution instructions found in the "How to run" section of each **WORKFLOW CONFIGURATION** page (located in the left-hand menu of this documentation), depending on the `execution method <quick_start.html#execution-methods>`__ you choose.

Find a visual guide of these steps below:

.. carousel::
    :show_controls:
    :show_captions_below:
    :data-bs-interval: false
    :show_indicators:    

    .. figure:: ../img/installation-methods.svg

        Step 1: Under GET STARTED > Installation > Choose your installation method, look for the option that best matches your expertise and operating system.

    .. figure:: ../img/workflow-selection.svg

        Step 2: Under GET STARTED > Select workflow, find the description of a workflow that best matches your task.

    .. figure:: ../img/how-to-run.svg

        Step 3: Under WORKFLOW CONFIGURATION, click on the workflow you selected in the previous step and follow the instructions under "How to run".


Execution methods
*****************

BiaPy offers several execution methods, designed to accommodate a range of expertise levels, from beginner to advanced:

* **Graphical User Interface (GUI)**: With a user-friendly wizard, ideal for beginners. Have a look at our quick GUI walkthrough video:

    .. raw:: html

        <iframe width="560" height="315" src="https://www.youtube.com/embed/vY7aBh5FUNk?si=yvVolBnu5APNeHwB" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

* **Jupyter Code-free Notebooks**: Can be run locally or through Google Colab. More details are available in our quick notebooks walkthrough video:

    .. raw:: html
        
        <iframe width="560" height="315" src="https://www.youtube.com/embed/KEqfio-EnYw?si=eu8nfOjjV1ioY32q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

* **Docker Containers**: For portable and reproducible environments.
* **Command-Line Interface (CLI)**: For advanced users comfortable with terminal commands.


Limitations
***********

While BiaPy covers a wide range of tasks such as segmentation, object detection, and super-resolution, it does not include functionalities for common tasks like image visualization, registration, tracking, or manual annotation. For these tasks, we recommend using other popular `tools in the bioimage analysis community <https://forum.image.sc/>`__.

For example, visualization is not natively supported in BiaPy. To address this, we have collaborated with the `Brainglobe project <https://brainglobe.info/>`__, an open-source initiative focused on computational neuroanatomy. In this partnership, BiaPy handles large-scale image processing, such as brain-wide cell detection, while Brainglobe's framework is used for visualizing and further analyzing the results.

A tutorial on how to integrate BiaPy with Brainglobe can be accessed `here <../tutorials/detection/brain_cell_detection.html>`__.

Moving forward, BiaPy will remain focused on deep learning tasks, relying on specialized platforms for complementary functions like visualization and registration.

Further information
*******************

For a more in-depth understanding of BiaPy functioning, visit the `"How it works" <how_it_works.html>`__ and `"Configuration" <configuration.html>`__ sections. Similarly, if you encounter any challenges while running BiaPy, please consult the `FAQ & Troubleshooting <faq.html>`__ section for assistance.

The **TUTORIALS** section is populated with detailed tutorials designed to replicate the tasks accomplished in all the projects where different BiaPy workflows have been employed.

To contribute or develop your own code based on BiaPy, please follow the guidelines on the `"How to contribute" <contribute.html>`__ section.