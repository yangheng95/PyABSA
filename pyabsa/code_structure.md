# Directory structure description
```
PyABSA/
├── examples   #  usages of PyABSA
│   ├── aspect_polarity_classification   # tutorials of training & inference of aspect polarity prediction
│   ├──	aspect_term_extraction		 	 # tutorials of training & using ATEPC model to do ATE and APC
│   └──	common_usages 				     #	common usages, such as loading checkpoint in specified device
└── pyabsa   # source code
    ├── config   # default hyper-parameter handler  
    │	├── apc_config     # apc_config_handler
    │	└── atepc_config   # atepc_config_handler
    ├── network    # neural network encoder 
    ├── research   # research-related entries
    │	├──benchmark   # benchmark manager
    │	│   ├── apc_benchmark     # apc benchmark code
    │	│   └──	atepc_benchmark   # atepc benchmark code
    │	└── parameter_search  # hyper-parameter fine-tuning entry
    │       ├── search_param_for_apc     # apc param search module
    │       └──	search_param_for_atepc   # atepc param search module
    ├── tasks  # define the modeling process for a set of ABSA subtasks
    │	├── apc   # apc modeling module
    │   │   ├── __glove__   # compatiable model from PyTorch-ABSA
    │   │   │   ├── dataset_utils          	
    │   │   │   └──	models      
    │   │   ├── dataset_utils   # data preprocess & model input preparation
    │	│   ├── models          # apc models
    │   │   ├── prediction  	# sentiment inference module
    │   │   └──	training        # apc trainer
    │   ├── ate   # in coming 
    │   └── atepc  # atepc modeling module
    │       ├── assets          # original materials of LCF-ATEPC
    │       ├── dataset_utils   # data preprocess & model input preparation
    │       ├── models          # atepc models
    │       ├──	prediction      # aspect extraction & sentiment inference module
    │       └──	training        # aepc trainer
    ├──	utils   # support code
    ├──	absa_dataset.py   # dataset downloading & loading module
    ├──	functional.py     # packaged apc/atepc training & checkpoint loading entries 
    └──	model_utils.py    # checkpoints downloading and model handler module
```