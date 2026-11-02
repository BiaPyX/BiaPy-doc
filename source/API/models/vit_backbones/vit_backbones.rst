biapy.models.vit_backbones
--------------------------

.. automodule:: biapy.models.vit_backbones
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
   * - :mod:`biapy.models.vit_backbones.celldino`
     - Cell-DINO ViT-L/16 backbone for 1-channel images.
   * - :mod:`biapy.models.vit_backbones.cellpose_sam`
     - Cellpose-SAM encoder, without depending on the cellpose/segment_anything packages.
   * - :mod:`biapy.models.vit_backbones.common`
     - Patch embedding, predefined ViT backbones and their blocks, checks and pretrained weight loading.
   * - :mod:`biapy.models.vit_backbones.dinov3`
     - DINOv3 ViT-L/16 and ViT-B/16 image encoders.
   * - :mod:`biapy.models.vit_backbones.encoder`
     - ViT encoder mixin shared by the UNETR, DPT and ViT-Readout architectures.
   * - :mod:`biapy.models.vit_backbones.sam3`
     - SAM 3 (Segment Anything Model 3) ViT image encoder.
   * - :mod:`biapy.models.vit_backbones.weights`
     - Helpers shared by the pretrained ViT backbone weight loaders.

.. toctree::
   :maxdepth: 1
   :hidden:

   celldino
   cellpose_sam
   common
   dinov3
   encoder
   sam3
   weights
