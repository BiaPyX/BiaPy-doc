biapy.models.vit_decoders
-------------------------

.. automodule:: biapy.models.vit_decoders
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
   * - :mod:`biapy.models.vit_decoders.dpt_decoder`
     - DPT's reassemble/fusion decoder.
   * - :mod:`biapy.models.vit_decoders.output_heads`
     - Output heads of the ViT-based dense-prediction architectures (DPT and ViT-Readout).
   * - :mod:`biapy.models.vit_decoders.readout_decoder`
     - Cellpose 4's per-token readout, pixel-shuffled into image-sized predictions.
   * - :mod:`biapy.models.vit_decoders.unetr_decoder`
     - UNETR's convolutional decoder, with skip connections from intermediate blocks.

.. toctree::
   :maxdepth: 1
   :hidden:

   dpt_decoder
   output_heads
   readout_decoder
   unetr_decoder
