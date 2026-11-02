biapy.wizard
------------

.. automodule:: biapy.wizard
    :members:
    :undoc-members:
    :show-inheritance:

Submodules
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :mod:`biapy.wizard.common`
     - Parts shared by the workflow strategies: when a strategy can be applied, the voxel spacing to plan with and the U-Net plan written into the configuration.
   * - :mod:`biapy.wizard.config_builder`
     - Creates the configuration from the wizard's answers: variables set, default values for a first run and the instance representation chosen.
   * - :mod:`biapy.wizard.data_check`
     - Checks the data folders selected in the wizard, reading the files with BiaPy's readers, and gathers channels, classes and the data fingerprint.
   * - :mod:`biapy.wizard.detection`
     - Strategy for the detection workflow, from the fingerprints of the images and of the CSV files with the points.
   * - :mod:`biapy.wizard.fingerprint`
     - Dataset fingerprint: properties of the data of a folder that the strategies use to configure BiaPy, in the spirit of nnU-Net.
   * - :mod:`biapy.wizard.inference`
     - Inference that fits in the computer's memory (RAM), estimated from the shape of the largest test image.
   * - :mod:`biapy.wizard.instance_seg`
     - Strategy for the instance segmentation workflow.
   * - :mod:`biapy.wizard.planning`
     - Configuration adapted to the data, in the spirit of nnU-Net, from the fingerprints of the data checked in the wizard.
   * - :mod:`biapy.wizard.preview`
     - Preview of the data selected in the wizard: a few raw images next to their targets.
   * - :mod:`biapy.wizard.resources`
     - Estimation of the computer resources (RAM, GPU memory and disk) a wizard configuration needs.
   * - :mod:`biapy.wizard.semantic_seg`
     - Strategy for the semantic segmentation workflow, from the fingerprints of the training images and masks.
   * - :mod:`biapy.wizard.unet`
     - nnU-Net-like planning of BiaPy's U-Net from the dataset fingerprint.

.. toctree::
   :maxdepth: 1
   :hidden:

   common
   config_builder
   data_check
   detection
   fingerprint
   inference
   instance_seg
   planning
   preview
   resources
   semantic_seg
   unet
