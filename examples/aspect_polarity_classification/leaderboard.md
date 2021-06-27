# APC model leaderboard
Average accuracy running with 1, 2, 3 random seed. Due to the limitation of computational resources, 
we will be glad if you help us finish the leaderboard, for the sake of providing reference of reproduction.
The performance reports of other models are welcome.

|      Models          | Laptop14 (acc) |  Rest14 (acc) | Rest15 (acc) | Rest16 (acc) |
| :------------------: | :------------: | :-----------: |:------------:|:------------:|
| SLIDE-LCFS-BERT (CDW)|    81.35       |        88.04  |    85.93     |   92.52      | 
| SLIDE-LCFS-BERT (CDM)|     82.13      |        87.5   |    85.37     |   92.36      |
| SLIDE-LCF-BERT (CDW) |      80.82         |        86.04      |    85.18        |    91.98         |
| SLIDE-LCF-BERT (CDM) |    80.67          |        86.13      |   85.31          |    91.92         |
| FAST-LCF-BERT (CDW) |      80.35	         |      86.4      |       84.5     |    91.06        |
| FAST-LCF-BERT (CDM) |    80.62          |        86.34      |   84.57          |    92.09         |
| FAST-LCFS-BERT (CDW) |      80.56         |       86.4       |    85.62         |    91.49         |
| FAST-LCFS-BERT (CDM) |    80.15           |        86.69      |   84.63         |    91.65         |
| LCF-BERT-LARGE (CDW) |      80.04         |       86.4       |    84.69         |    91.49         |
| LCF-BERT-LARGE (CDM) |    80.25           |        85.89      |   85.13         |    91.22         |
| LCFS-BERT-LARGE (CDW) |      79.31         |       86.7       |    84.26         |    91.54         |
| LCFS-BERT-LARGE (CDM) |    79.94           |        85.54      |   85.74         |    92.2         |
| BERT-BASE |      80.46         |       83.24       |    82.53         |    89.65         |
| BERT-SPC |    80.62          |        86.55      |   84.75          |    91.44         |
| Etc. |      -         |       -       |    -         |    -         |

**The above results were obtained in the 0.8.7.4 version of PyABSA, and all experiments were performed on the Nvidia Tesla T4 GPU.**