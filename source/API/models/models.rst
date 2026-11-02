.. _models-reference:

biapy.models
============

.. automodule:: biapy.models
    :members:
    :undoc-members:
    :show-inheritance:

Submodules
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :mod:`biapy.models.attention_unet`
     - U-Net variant with attention gates for improved feature propagation.
   * - :mod:`biapy.models.bmz_search`
     - Search of the BioImage Model Zoo for the models BiaPy can use.
   * - :mod:`biapy.models.dfcan`
     - Deep Fourier Channel Attention Network for image restoration.
   * - :mod:`biapy.models.dpt`
     - Dense Prediction Transformer (DPT) for dense prediction tasks.
   * - :mod:`biapy.models.edsr`
     - Enhanced Deep Super-Resolution (EDSR) model.
   * - :mod:`biapy.models.efficientnet`
     - EfficientNet backbone for image classification and feature extraction.
   * - :mod:`biapy.models.gan_wrapper`
     - Generic GAN wrapper that attaches a PatchGAN discriminator to any generator module.
   * - :mod:`biapy.models.heads`
     - Head architectures, including Atrous Spatial Pyramid Pooling (ASPP) and projection heads.
   * - :mod:`biapy.models.hrnet`
     - High-Resolution Network (HRNet) for semantic segmentation.
   * - :mod:`biapy.models.mae`
     - Masked Autoencoder (MAE) for self-supervised pretraining.
   * - :mod:`biapy.models.multiresunet`
     - MultiResUNet for multi-scale feature fusion in segmentation.
   * - :mod:`biapy.models.nafnet`
     - NAFNet model (2D/3D) and its PatchGAN discriminator wiring.
   * - :mod:`biapy.models.patchgan`
     - PatchGAN discriminator, which scores realism per patch instead of globally.
   * - :mod:`biapy.models.rcan`
     - Residual Channel Attention Network for image super-resolution.
   * - :mod:`biapy.models.rdbm`
     - Residual Diffusion Bridge Model (RDBM) for paired image-to-image translation.
   * - :mod:`biapy.models.resunet`
     - Residual U-Net for segmentation tasks.
   * - :mod:`biapy.models.resunet_plus_plus`
     - ResUNet++: residual U-Net variant with attention and dense blocks (importable wrapper of ``resunet++``).
   * - :mod:`biapy.models.resunet_se`
     - Residual U-Net with Squeeze-and-Excitation blocks.
   * - :mod:`biapy.models.seunet`
     - U-Net variant incorporating Squeeze-and-Excitation blocks.
   * - :mod:`biapy.models.simple_cnn`
     - Simple convolutional neural network for baseline tasks.
   * - :mod:`biapy.models.stunet`
     - STU-Net, a scalable and transferable U-Net for medical image segmentation.
   * - :mod:`biapy.models.unet`
     - Classic U-Net architecture for image segmentation.
   * - :mod:`biapy.models.unetr`
     - U-Net Transformer (UNETR) architecture.
   * - :mod:`biapy.models.unext_v1`
     - UNeXt version 1: lightweight U-Net with MLP-like blocks.
   * - :mod:`biapy.models.unext_v2`
     - UNeXt version 2: refined lightweight U-Net variant.
   * - :mod:`biapy.models.vit`
     - Vision Transformer (ViT) backbone for image analysis.
   * - :mod:`biapy.models.vit_readout`
     - ViT-Readout: a plain Vision Transformer whose last-layer tokens are read out directly into pixels.
   * - :mod:`biapy.models.wavelettention`
     - Wavelettention: hybrid attention transformer for super-resolution, trained with a wavelet-subband loss.
   * - :mod:`biapy.models.wdsr`
     - Wide Activation Super-Resolution (WDSR) model.
   * - :mod:`biapy.models.blocks`
     - Building blocks and layers used across multiple models.
   * - :mod:`biapy.models.bmz_utils`
     - Utility functions for BMZ-based operations.
   * - :mod:`biapy.models.memory_bank`
     - Memory bank module for contrastive learning.
   * - :mod:`biapy.models.vit_backbones`
     - ViT backbones (SAM 3, DINOv3, Cell-DINO, Cellpose-SAM...) shared by the transformer-based architectures.
   * - :mod:`biapy.models.vit_decoders`
     - Decoders of the ViT encoder-decoder architectures (UNETR, DPT and ViT-Readout).

.. toctree::
   :maxdepth: 1
   :hidden:

   attention_unet
   bmz_search
   dfcan
   dpt
   edsr
   efficientnet
   gan_wrapper
   heads
   hrnet
   mae
   multiresunet
   nafnet
   patchgan
   rcan
   rdbm
   resunet
   resunet_plus_plus
   resunet_se
   seunet
   simple_cnn
   stunet
   unet
   unetr
   unext_v1
   unext_v2
   vit
   vit_readout
   wavelettention
   wdsr
   blocks
   bmz_utils
   memory_bank
   vit_backbones/vit_backbones
   vit_decoders/vit_decoders

