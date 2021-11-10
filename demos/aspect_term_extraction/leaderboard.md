# ATEPC model leaderboard

These experimental results are not rigorous reproduction of corresponding papers. Average accuracy running with 1, 2, 3
random seeds. Due to the limitation of computational resources, we will be glad if you help us finish the leaderboard,
for the sake of providing reference of reproduction. The performance reports of other models are welcome.


|                        |     LAP14    |             |             |     RES14    |             |             |     RES15    |             |             |     RES16    |             |             |
|------------------------|:------------:|:-----------:|:-----------:|:------------:|-------------|-------------|:------------:|-------------|-------------|:------------:|-------------|-------------|
|                        | Max APC Acc: | Max APC F1: | Max ATE F1: | Max APC Acc: | Max APC F1: | Max ATE F1: | Max APC Acc: | Max APC F1: | Max ATE F1: | Max APC Acc: | Max APC F1: | Max ATE F1: |
|     BERT_BASE_ATEPC    |     79.32    |    74.88    |    79.81    |     83.11    |    74.42    |    86.54    |     82.25    |    68.53    |    77.58    |     89.07    |    72.93    |    80.94    |
|  FAST_LCF_ATEPC (cdw)  |  79.43 |    75.73    |  79.25  |     84.09    |    76.71    |     83.4    |   81.99   |    59.43    |    77.83    |   89.85   |   72.06  |   78.89  |
|  FAST_LCF_ATEPC (cdm)  |  79.21 |  74.77 |  79.31  |     84.45    |    76.89    |    86.97    |   82.88   |   61.95  |    77.76    |   90.35   |   73.65  |   79.47  |
|  FAST_LCFS_ATEPC (cdw) |     78.48    |  74.36 |  79.82  |   84.06   |   76.52  |   86.66  |   82.05   |    60.12    |   77.26  |   89.63   |   72.03  |   79.45  |
|  FAST_LCFS_ATEPC (cdm) |     79.16    |  74.75 |    79.56    |   84.54   |     77.1    |   87.02  |   82.05   |   63.41  |    77.79    |   90.02   |   74.62  |   78.79  |
|     LCF_ATEPC (cdw)    |  78.85 |  74.72 |    79.72    |   84.96   |   77.95  |   87.02  |   82.88   |   60.36  |   77.92  |   89.68   |   72.77  |    78.22    |
|     LCF_ATEPC (cdm)    |     78.79    |     75.2    |  79.95  |     84.6     |    76.94    |   87.02  |   82.31   |   59.42  |    77.91    |   90.69   |   74.61  |   78.63  |
|  LCF_ATEPC_LARGE (cdw) |  79.37 |  75.32 |  80.09  |   83.32   |   74.9  |   86.84  |   81.99  |   63.92  |   78.29  |     89.63    |     72.4    |   78.33  |
|  LCF_ATEPC_LARGE (cdm) |      80      |  76.36 |  80.20  |   83.82   |   76.03  |   86.85  |   82.63   |   62.75  |    78.06    |     89.69    |   68.58  |    78.02    |
|    LCFS_ATEPC (cdw)    |  78.38 |  74.41 |    80.15    |   83.83   |    75.82    |   87.06  |   82.24   |     62.6    |   77.66  |     89.91    |    72.51    |    79.16    |
|    LCFS_ATEPC (cdm)    |  78.06 |  73.94 |    79.71    |     83.94    |    76.01    |    86.79    |   82.24   |   60.66  |   77.78  |   89.74   |   72.59  |   78.39  |
| LCFS_ATEPC_LARGE (cdw) |     78.48    |  74.04 |  80.31  |   83.70   |   75.09  |   86.80  |   81.54   |   61.06  |   78.18  |   89.74   |    70.04    |    78.39    |
| LCFS_ATEPC_LARGE (cdm) |     77.64    |  73.06 |    80.34    |     82.9     |   73.97  |   87.03  |     81.16    |   61.57  |   77.85  |     89.8     |   72.21 |   78.02  |

**The above results were obtained in the 1.2.12 version of PyABSA, and all experiments were performed on the Nvidia Tesla
T4 GPU.**
