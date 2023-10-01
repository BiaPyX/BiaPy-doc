BiaPy: Bioimage analysis pipelines in Python
============================================

.. image:: img/biapy_logo.svg
   :width: 50%
   :align: center 

`BiaPy <https://github.com/danifranco/BiaPy>`_ is an open source Python library for building bioimage analysis pipelines, also called workflows. This repository is actively under development by the Biomedical Computer Vision group at the `University of the Basque Country <https://www.ehu.eus/en/en-home>`_ and the `Donostia International Physics Center <http://dipc.ehu.es/>`_. 

The library provides an easy way to create image processing pipelines that are commonly used in the analysis of biology microscopy images in 2D and 3D. Specifically, BiaPy contains ready-to-use solutions for tasks such as `semantic segmentation <workflows/semantic_segmentation.html>`_, `instance segmentation <workflows/instance_segmentation.html>`_, `object detection <workflows/detection.html>`_, `image denoising <workflows/denoising.html>`_, `single image super-resolution <workflows/super_resolution.html>`_, `self-supervised learning <workflows/self_supervision.html>`_ and `image classification <workflows/classification.html>`_. The source code is based on Pytorch as the backend. As BiaPy's core is based on deep learning, it is recommended to use a machine with a graphics processing unit (GPU) for faster training and execution.                                                                        
   
.. toctree::
   :maxdepth: 1
   :caption: Get started
   :glob:
   
   get_started/quick_start.rst
   get_started/installation.rst
   get_started/how_it_works.rst
   get_started/configuration.rst
   get_started/select_workflow.rst
   get_started/faq.rst

.. toctree::
   :maxdepth: 1
   :caption: Workflows
   :glob:

   workflows/semantic_segmentation.rst
   workflows/instance_segmentation.rst
   workflows/detection.rst
   workflows/denoising.rst
   workflows/super_resolution.rst
   workflows/self_supervision.rst
   workflows/classification.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :glob:

   tutorials/stable.rst
   tutorials/cartocell.rst
   tutorials/mitoem.rst
   tutorials/nucleus.rst

.. toctree::                                                                    
   :maxdepth: 1
   :caption: How to contribute
   :glob:

   contribute/general
   contribute/workflow
   contribute/pre_post_proc

.. toctree::                                                                    
   :maxdepth: 1
   :caption: API
   :glob:

   API/config/config
   API/data/data
   API/engine/engine
   API/models/models
   API/utils/utils


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Bibliography

   bibliography
   

Citation
========

This repository is the base of the following works: ::

   @inproceedings{franco2023biapy,
      title={BiaPy: a ready-to-use library for Bioimage Analysis Pipelines},
      author={Franco-Barranco, Daniel and Andr{\'e}s-San Rom{\'a}n, Jes{\'u}s A and G{\'o}mez-G{\'a}lvez, Pedro and Escudero, Luis M and Mu{\~n}oz-Barrutia, Arrate and Arganda-Carreras, Ignacio},
      booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
      pages={1--5},
      year={2023},
      organization={IEEE}
   }
   
   @article {Andr{\'e}s-San Rom{\'a}n2023.01.05.522724,
      author = {Jes{\'u}s A. Andr{\'e}s-San Rom{\'a}n and Carmen Gordillo-V{\'a}zquez and Daniel Franco-Barranco and Laura Morato and Cecilia H. Fern{\'a}ndez-Espartero and Gabriel Baonza and Antonio Tagua and Pablo Vicente-Munuera and Ana M. Palacios and Mar{\'\i}a P. Gavil{\'a}n and Fernando Mart{\'\i}n-Belmonte and Valentina Annese and Pedro G{\'o}mez-G{\'a}lvez and Ignacio Arganda-Carreras and Luis M. Escudero},
      title = {CartoCell, a high-content pipeline for 3D image analysis, unveils cell morphology patterns in epithelia},
      elocation-id = {2023.01.05.522724},
      year = {2023},
      doi = {10.1101/2023.01.05.522724},
      publisher = {Cold Spring Harbor Laboratory},
      URL = {https://www.biorxiv.org/content/early/2023/08/31/2023.01.05.522724},
      eprint = {https://www.biorxiv.org/content/early/2023/08/31/2023.01.05.522724.full.pdf},
      journal = {bioRxiv}
   }

   @article{franco2022domain,
      title = {Deep learning based domain adaptation for mitochondria segmentation on EM volumes},
      journal = {Computer Methods and Programs in Biomedicine},
      volume = {222},
      pages = {106949},
      year = {2022},
      publisher={Elsevier}
      issn = {0169-2607},
      doi = {https://doi.org/10.1016/j.cmpb.2022.106949},
      url = {https://www.sciencedirect.com/science/article/pii/S0169260722003315},
      author={Franco-Barranco, Daniel and Pastor-Tronch, Julio and Gonz{\'a}lez-Marfil, Aitor and Mu{\~n}oz-Barrutia, Arrate and Arganda-Carreras, Ignacio},
   }

   @Article{Franco-Barranco2021,                                                                                           
      author={Franco-Barranco, Daniel and Muñoz-Barrutia, Arrate and Arganda-Carreras, Ignacio},                        
      title={Stable Deep Neural Network Architectures for Mitochondria Segmentation on Electron Microscopy Volumes},          
      journal={Neuroinformatics},                                                                                             
      year={2021},                                                                                                            
      month={Dec},                                                                                                            
      day={02},                                                                                                               
      issn={1559-0089},                                                                                                       
      doi={10.1007/s12021-021-09556-1},                                                                                       
      url={https://doi.org/10.1007/s12021-021-09556-1}                                                                        
   }        

  @inproceedings{wei2020mitoem,
      title={MitoEM dataset: large-scale 3D mitochondria instance segmentation from EM images},
      author={Wei, Donglai and Lin, Zudi and Franco-Barranco, Daniel and Wendt, Nils and Liu, Xingyu and Yin, Wenjie and Huang, Xin and Gupta, Aarush and Jang, Won-Dong and Wang, Xueying and others},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={66--76},
      year={2020},
      organization={Springer}
  }
  
