# APC model leaderboard
Average accuracy running with 1, 2, 3 random seed. Due to the limitation of computational resources, 
we will be glad if you help us finish the leaderboard, for the sake of providing reference of reproduction.
The performance reports of other models are welcome.

|      Models          | Laptop14 (acc) |  Rest14 (acc) | Rest15 (acc) | Rest16 (acc) |
| :------------------: | :------------: | :-----------: |:------------:|:------------:|
| SLIDE-LCFS-BERT (CDW)|    81.09       |        86.87  |    85.06     |   91.55     | 
| SLIDE-LCFS-BERT (CDM)|     81.14      |        87.02   |    84.69     |   91.71      |
| SLIDE-LCF-BERT (CDW) |      80.82         |        87.08      |    84.69        |    91.60         |
| SLIDE-LCF-BERT (CDM) |    80.20          |        86.64      |   85.37          |    91.49         |
| FAST-LCF-BERT (CDW) |      80.67	         |      86.28      |       84.50     |    91.81        |
| FAST-LCF-BERT (CDM) |    80.98          |        87.11      |   85.43          |    91.96         |
| FAST-LCFS-BERT (CDW) |      80.56         |       86.4       |    85.62         |    91.49         |
| FAST-LCFS-BERT (CDM) |    80.15           |        86.69      |   84.63         |    91.65         |
| LCF-BERT-LARGE (CDW) |      80.04         |       86.4       |    84.69         |    91.49         |
| LCF-BERT-LARGE (CDM) |    80.25           |        85.89      |   85.13         |    91.22         |
| LCFS-BERT-LARGE (CDW) |      79.31         |       86.7       |    84.26         |    91.54         |
| LCFS-BERT-LARGE (CDM) |    79.94           |        85.54      |   85.74         |    92.2         |
| LCF-BERT (CDW) |      80.88         |       86.22       |    83.76         |    91.6         |
| LCF-BERT(CDM) |    79.88           |        85.98      |   84.69         |    91.65         |
| LCFS-BERT(CDW) |      -         |       86.40       |    84.94         |    91.92         |
| LCFS-BERT(CDM) |    80.82           |        87.00      |   84.44         |    92.14         |
| BERT-BASE |      80.46         |       83.24       |    82.53         |    89.65         |
| BERT-SPC |    80.62          |        86.55      |   84.75          |    91.44         |
| Etc. |      -         |       -       |    -         |    -         |

**The above results were obtained in the 0.8.8.0 version of PyABSA, and all experiments were performed on the Nvidia Geforce RTX2080 GPU.**