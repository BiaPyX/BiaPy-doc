CPU vs GPU
----------

In order to measure the differences between using a GPU or just CPU for training we conducted several tests with different workflows and image dimensions.

2D problem differences
~~~~~~~~~~~~~~~~~~~~~~

The following values were measured with BiaPy `v3.5.0 <https://github.com/BiaPyX/BiaPy/tree/v3.5.0>`__, `wound_segmentation template <https://github.com/BiaPyX/BiaPy/blob/v3.5.0/templates/semantic_segmentation/wound_segmentation.yaml>`__ and using `the embryo wound segmentation data <https://drive.google.com/file/d/1qehkWYVJRXfMwvbpayKhb4nmPyYvclAj/view?usp=drive_link>`__ (:cite:p:`backova2023modeling`). 

+-------------------------------------+-------------------+-------------------+
| Semantic segmentation               | CPU               | GPU               |
+=====================================+===================+===================+
| Train time                          | 3:15:16 ± 0:05:30 | 0:22:12 ± 0:00:07 |
|                                     |                   |                   |
| Test Foreground IoU (per patch)     | 0,336 ± 0.000     | 0,279 ± 0.004     |
|                                     |                   |                   |
| Test Foreground IoU (merge patches) | 0,410 ± 0.000     | 0,369 ± 0.012     |
+-------------------------------------+-------------------+-------------------+

The following values were measured with BiaPy `v3.5.0 <https://github.com/BiaPyX/BiaPy/tree/v3.5.0>`__, `instance_segmentation template <https://github.com/BiaPyX/BiaPy/blob/v3.5.0/templates/instance_segmentation/2d_instance_segmentation.yaml>`__ and using `the ZeroCostDL4Mic - Stardist example training and test dataset <https://drive.google.com/file/d/1b7_WDDGEEaEoIpO_1EefVr0w0VQaetmg/view>`__ for cell instance segmentation.

+-------------------------------------+-------------------+-------------------+
| Instance segmentation               | CPU               | GPU               |
+=====================================+===================+===================+
| Train time                          | 3:31:36 ± 0:01:54 | 0:12:54 ± 0:00:11 |
|                                     |                   |                   |
| Test Foreground IoU (per patch)     | 0,077 ± 0.000     | 0,074 ± 0.002     |
|                                     |                   |                   |
| Test Foreground IoU (merge patches) | 0,741 ± 0.000     | 0,732 ± 0.008     |
|                                     |                   |                   |
| Precision (threshold 0.3)           | 0,973 ± 0.000     | 0,932 ± 0.003     |
|                                     |                   |                   |
| Precision (threshold 0.5)           | 0,968 ± 0.000     | 0,922 ± 0.003     |
|                                     |                   |                   |
| Precision (threshold 0.75)          | 0,802 ± 0.000     | 0,859 ± 0.008     |
|                                     |                   |                   |
| Recall (threshold 0.3)              | 0,982 ± 0.000     | 0,984 ± 0.003     |
|                                     |                   |                   |
| Recall (threshold 0.5)              | 0,977 ± 0.000     | 0,972 ± 0.003     |
|                                     |                   |                   |
| Recall (threshold 0.75)             | 0,809 ± 0.000     | 0,907 ± 0.008     |
|                                     |                   |                   |
| Accuracy (threshold 0.3)            | 0,955 ± 0.000     | 0,918 ± 0.006     |
|                                     |                   |                   |
| Accuracy (threshold 0.5)            | 0,947 ± 0.000     | 0,898 ± 0.006     |
|                                     |                   |                   |
| Accuracy (threshold 0.75)           | 0,674 ± 0.000     | 0,790 ± 0.013     |
|                                     |                   |                   |
| F1 (threshold 0.3)                  | 0,977 ± 0.000     | 0,957 ± 0.003     |
|                                     |                   |                   |
| F1 (threshold 0.5)                  | 0,973 ± 0.000     | 0,946 ± 0.003     |
|                                     |                   |                   |
| F1 (threshold 0.75)                 | 0,805 ± 0.000     | 0,883 ± 0.008     |
|                                     |                   |                   |
| Panoptic quality (threshold 0.3)    | 0,820 ± 0.000     | 0,830 ± 0.001     |
|                                     |                   |                   |
| Panoptic quality (threshold 0.5)    | 0,818 ± 0.000     | 0,825 ± 0.001     |
|                                     |                   |                   |
| Panoptic quality (threshold 0.75)   | 0,704 ± 0.000     | 0,783 ± 0.007     |
+-------------------------------------+-------------------+-------------------+


3D problem differences
~~~~~~~~~~~~~~~~~~~~~~

The following values were measured with BiaPy `v3.5.0 <https://github.com/BiaPyX/BiaPy/tree/v3.5.0>`__, `semantic_segmentation template <https://github.com/BiaPyX/BiaPy/blob/v3.5.0/templates/semantic_segmentation/3d_semantic_segmentation.yaml>`__ and using `the electron microscopy (EPFL - CVLAB) dataset <https://drive.google.com/file/d/10Cf11PtERq4pDHCJroekxu_hf10EZzwG/view>`__ for mitochondria segmentation. 

+-------------------------------------+---------------------+---------------------+
| Semantic segmentation               | CPU                 | GPU                 |
+=====================================+=====================+=====================+
| Train time                          | 13:56:47 ± 00:02:02 | 02:11:03 ± 00:12:19 |
|                                     |                     |                     |
| Test Foreground IoU (per patch)     | 0,121 ± 0.000       | 0,115 ± 0.010       |
|                                     |                     |                     |
| Test Foreground IoU (merge patches) | 0,858 ± 0.000       | 0,829 ± 0.046       |
+-------------------------------------+---------------------+---------------------+

The following values were measured with BiaPy `v3.5.0 <https://github.com/BiaPyX/BiaPy/tree/v3.5.0>`__, `instance_segmentation template <https://github.com/BiaPyX/BiaPy/blob/v3.5.0/templates/instance_segmentation/3d_instance_segmentation.yaml>`__ and using `the 3D demo dataset from the StarDist 0.3.0 <https://drive.google.com/file/d/1fdL35ZTNw5hhiKau1gadaGu-rc5ZU_C7/view?usp=drive_link>`__ for nuclei instance segmentation.

+-------------------------------------+-------------------+-------------------+
| Instance segmentation               | CPU               | GPU               |
+=====================================+===================+===================+
| Train time                          | 4:24:58 ± 0:06:49 | 0:42:29 ± 0:00:55 |
|                                     |                   |                   |
| Test Foreground IoU (per patch)     | 0,074 ± 0.000     | 0,073 ± 0.001     |
|                                     |                   |                   |
| Test Foreground IoU (merge patches) | 0,797 ± 0.000     | 0,804 ± 0.001     |
|                                     |                   |                   |
| Precision (threshold 0.3)           | 0,996 ± 0.000     | 0,996 ± 0.000     |
|                                     |                   |                   |
| Precision (threshold 0.5)           | 0,912 ± 0.000     | 0,924 ± 0.012     |
|                                     |                   |                   |
| Precision (threshold 0.75)          | 0,106 ± 0.000     | 0,096 ± 0.010     |
|                                     |                   |                   |
| Recall (threshold 0.3)              | 0,957 ± 0.000     | 0,953 ± 0.006     |
|                                     |                   |                   |
| Recall (threshold 0.5)              | 0,877 ± 0.000     | 0,885 ± 0.006     |
|                                     |                   |                   |
| Recall (threshold 0.75)             | 0,102 ± 0.000     | 0,091 ± 0.009     |
|                                     |                   |                   |
| Accuracy (threshold 0.3)            | 0,953 ± 0.000     | 0,949 ± 0.006     |
|                                     |                   |                   |
| Accuracy (threshold 0.5)            | 0,808 ± 0.000     | 0,825 ± 0.015     |
|                                     |                   |                   |
| Accuracy (threshold 0.75)           | 0,055 ± 0.000     | 0,049 ± 0.005     |
|                                     |                   |                   |
| F1 (threshold 0.3)                  | 0,976 ± 0.000     | 0,974 ± 0.003     |
|                                     |                   |                   |
| F1 (threshold 0.5)                  | 0,894 ± 0.000     | 0,904 ± 0.009     |
|                                     |                   |                   |
| F1 (threshold 0.75)                 | 0,104 ± 0.000     | 0,093 ± 0.010     |
|                                     |                   |                   |
| Panoptic quality (threshold 0.3)    | 0,629 ± 0.000     | 0,631 ± 0.002     |
|                                     |                   |                   |
| Panoptic quality (threshold 0.5)    | 0,591 ± 0.000     | 0,598 ± 0.003     |
|                                     |                   |                   |
| Panoptic quality (threshold 0.75)   | 0,081 ± 0.000     | 0,073 ± 0.007     |
+-------------------------------------+-------------------+-------------------+
