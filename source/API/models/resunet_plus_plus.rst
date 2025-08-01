=====================
ResUNet++ Model
=====================

.. py:module:: resunet_plus_plus
   :noindex:

This module implements the ResUNet++ architecture, a deep learning model tailored for semantic segmentation tasks in biomedical image segmentation. It extends the traditional U-Net architecture with residual connections, squeeze-and-excitation (SE) blocks, attention mechanisms, and atrous spatial pyramid pooling (ASPP), offering enhanced feature representation and robustness across 2D and 3D image data.

The implementation is flexible to support tasks like:

* Semantic segmentation
* Instance segmentation (with multi-head output)
* Point detection
* Super-resolution
* Contrastive learning

Reference:
ResUNet++: An Advanced Architecture for Medical Image Segmentation
`https://arxiv.org/pdf/1911.07067.pdf`

.. autoclass:: resunet_plus_plus.ResUNetPlusPlus
   :members:
   :show-inheritance: