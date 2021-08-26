# ATEPC model leaderboard

Average accuracy running with 1, 2, 3 random seed. Due to the limitation of computational resources, we will be glad if
you help us finish the leaderboard, for the sake of providing reference of reproduction. The performance reports of
other models are welcome.

## Laptop14 
|         Models       | APC acc |  APC F1 | ATE F1 | 
| :------------------: | :------------: | :-----------: |:------------:|
| LCFS-ATEPC-LARGE (CDW)|    80.45       |      76.67    |    79.54     |
| LCFS-ATEPC (CDW)|    79.56     |     75.45    |    78.93     |
| LCF-ATEPC (CDW)|    79.40      |      75.83    |    78.55     |
| FAST-LCF-ATEPC (CDW)|    79.19      |      75.41    |    78.55     |
| BERT-BASE-ATEPC|    78.83      |      74.56    |    78.34     |
| LCF-ATEPC-LARGE (CDW)|    78.51      |      74.57    |    79.68     |
| FAST-LCFS-ATEPC (CDW)|    78.35      |      74.13    |    78.27    |

**The above results were obtained in the 1.1.9 version of PyABSA, and all experiments were performed on the Nvidia
Tesla T4 GPU.**