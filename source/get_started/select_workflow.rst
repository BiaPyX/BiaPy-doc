Select workflow
---------------

In bioimage analysis, the input and ground truth data vary depending on the specific workflow being used. The following are the workflows implemented in BiaPy and the corresponding input and ground truth data they require. Once you've selected the one you wish to use, follow the running instructions found on each workflow's page.

* `Semantic segmentation <../workflows/semantic_segmentation.html>`_, the input is an image of the area or object of interest, while the ground truth is another image of the same shape as the input, with a label assigned to each pixel.  
* `Instance segmentation <../workflows/instance_segmentation.html>`_, the input and ground truth are similar to semantic segmentation, but the ground truth also includes a unique identifier for each object.
* `Detection <../workflows/detection.html>`_, the goal is to recognize objects in images without needing a pixel-level accuracy output. The input is an image, while the ground truth is a CSV file containing the coordinates of the center point of each object.
* `Denoising <../workflows/denoising.html>`_, the goal is to remove noise from a given input image. The input is a noisy image, and no ground truth is required as the model uses an unsupervised learning technique to remove noise (`Noise2Void <https://arxiv.org/abs/1811.10980>`__).
* `Super-resolution <../workflows/super_resolution.html>`_, the goal is to reconstruct high-resolution images from low-resolution ones. The input is a low-resolution image, and the ground truth is a high-resolution (``×2`` or ``×4`` larger) version of the same image. 
* `Self-supervision <../workflows/self_supervision.html>`_, the model is trained without the use of labeled data. Instead, the model is presented with a so-called pretext task, such as predicting the rotation of an image, which allows it to learn useful features from the data. Once this initial training is complete, the model can be fine-tuned using labeled data for a specific task, such as image classification. The input in this workflow is simply an image, as no ground truth is needed for the initial training phase (unsupervised learning). 
* `Classification <../workflows/classification.html>`_, the goal is to match a given input image to its corresponding class. The input is an image, and the ground truth is the label of the corresponding class.

.. image:: ../img/BiaPy-workflow-examples.svg
   :width: 70%
   :align: center 