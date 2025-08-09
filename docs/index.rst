PyABSA Documentation
=======================================

PyABSA is an open-source framework focusing on Aspect-Based Sentiment Analysis (ABSA) and related tasks, providing end-to-end capabilities for training, inference, augmentation, and visualization.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: 0_intro/introduction
      :link-type: doc
      :class-card: sd-shadow-sm

      Learn the goals, scope, and typical use cases of PyABSA.

      - Installation
      - Quick start
      - FAQ

   .. grid-item-card:: User Guide
      :link: 2_config/customize_config
      :link-type: doc
      :class-card: sd-shadow-sm

      Configuration, data, annotation, and visualization.

      - Supported tasks and data formats
      - Customizing configuration
      - Datasets and annotation
      - Metrics visualization

   .. grid-item-card:: Advanced Topics
      :link: 5_augmentation/absc
      :link-type: doc
      :class-card: sd-shadow-sm

      Improve performance and robustness.

      - Text augmentation
      - Adversarial robustness
      - BPE and pretraining

   .. grid-item-card:: Tutorials & Demos
      :link: 6_tutorials/Aspect_Sentiment_Classification
      :link-type: doc
      :class-card: sd-shadow-sm

      Reproduce and get started quickly with notebooks.

      - Classification / Extraction / Triplet
      - Flask inference demos

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   0_intro/introduction.md
   0_intro/installation.md

.. toctree::
   :maxdepth: 2
   :caption: Quick start

   1_quick_start/absc.md
   1_quick_start/ate.md
   1_quick_start/atesc.md
   1_quick_start/annotate.md
   1_quick_start/adversarial.md
   1_quick_start/autoaug.md
   1_quick_start/bpe.md

.. toctree::
   :maxdepth: 2
   :caption: User guide

   8_supported_tasks/tasks.md
   2_config/customize_config.md
   7_datasets/customized_datasets.md
   6_tutorials/metrics_visualization.md

.. toctree::
   :maxdepth: 2
   :caption: Tutorials and notebooks

   6_tutorials/Aspect_Sentiment_Classification.ipynb
   6_tutorials/Aspect_Term_Extraction.ipynb
   6_tutorials/Aspect_Sentiment_Triplet_Extraction.ipynb
   3_inference/APC_flask_demo.ipynb
   3_inference/ASTE_flask_demo.ipynb
   3_inference/ATEPC_flask_demo.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Advanced topics

   5_augmentation/absc.md
   5_augmentation/text_classification.md

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Citation

   9_citation/citation.md
