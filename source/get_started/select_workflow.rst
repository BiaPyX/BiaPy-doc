Select workflow
---------------

In bioimage analysis, the **input** and **output** data vary depending on the specific workflow being used. The following are the workflows implemented in BiaPy and the corresponding input and output data they require. Once you've identified the one you wish to use, follow the running instructions found on each workflow's page (under "How to run").

* `Semantic segmentation <../workflows/semantic_segmentation.html>`_, the input is an image of the area or object of interest, while the output is another image of the same shape as the input, with a **semantic label** (a numerical value defining its category) **assigned to each pixel**. During the training phase, the expected label image of the input (i.e. the ground truth) needs to be also provided for the model to learn:

  .. image:: ../img/semantic_seg/workflow-scheme.svg
   :width: 70%
   :align: center 

\

* `Instance segmentation <../workflows/instance_segmentation.html>`_, the input and output are similar to semantic segmentation, but the output also includes **a unique identifier for each individual object of interest**. During the training phase, the expected instance label image of the input (i.e. the ground truth) needs to be also provided for the model to learn:

  .. image:: ../img/instance-seg/workflow-scheme.svg
   :width: 70%
   :align: center 

\


* `Object detection <../workflows/detection.html>`_, the goal is to recognize objects in images without needing a pixel-level accuracy output. The input is an image, while the output is a CSV file containing the **coordinates of the center point of each object**. During the training phase, the list of coordinates from the input objects (i.e. the ground truth) needs to be also provided for the model to learn:

  .. image:: ../img/detection/workflow-scheme.svg
   :width: 70%
   :align: center 

  \
  Additionally, Biapy may output an image with the probability map of each object's center.


* `Image denoising <../workflows/denoising.html>`_, the goal is to remove noise from a given input image. The input is a noisy image, and the **output is the denoised image**. No ground truth is required as the model uses an unsupervised learning technique to remove noise (`Noise2Void <https://arxiv.org/abs/1811.10980>`__).

  .. image:: ../img/denoising/workflow-scheme.svg
   :width: 70%
   :align: center 

\  

* `Single image super-resolution <../workflows/super_resolution.html>`_, the goal is to reconstruct high-resolution images from low-resolution ones. The input is a low-resolution image, and the **output is a high-resolution** (usually ``×2`` or ``×4`` larger) **version of the same image**.

  .. image:: ../img/super-resolution/workflow-scheme.svg
   :width: 70%
   :align: center 

\

* `Self-supervised pre-training <../workflows/self_supervision.html>`_, the model is trained without the use of labeled data. Instead, the model is presented with a so-called pretext task, such as predicting the rotation of an image, which allows it to learn useful features from the data. Once this initial training is complete, the model can be fine-tuned using labeled data for a specific task, such as image classification. The input in this workflow is simply a set of images, and the **output is the pre-trained model**.

  .. image:: ../img/self-supervised/workflow-scheme.svg
   :width: 70%
   :align: center 

\

* `Image classification <../workflows/classification.html>`_, the goal is to match a given input image to its corresponding class. The **input is an image, and the output is the label of the corresponding class**.

  .. image:: ../img/classification/workflow-scheme.svg
   :width: 70%
   :align: center 

\

* `Image to image translation <../workflows/image_to_image.html>`_, the purpose of this workflow is to **translate or map input images to corresponding target images**. Often referred to as "image-to-image," this process is versatile and can be applied to various goals, including **image inpainting, colorization, and even super-resolution** (with a scale factor of ``x1``).

  .. image:: ../img/i2i/workflow-scheme.svg
   :width: 70%
   :align: center 

