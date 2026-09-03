<img width="9508" height="6707" alt="400a5e7aa38ba195c0005798264433a4" src="https://github.com/user-attachments/assets/6251d24a-ddad-401b-95e9-e30d0f999651" /># SKED-AVP

## Prediction of Antiviral Peptides

The peptides possessing the potential to inhibit viral infection are considered antiviral peptides (AVPs). Usually, AVPs exert antiviral effects by blocking viral entry, replication, or virus–host interactions. Although experimental approaches can identify AVPs, they are generally time-consuming, laborious, and costly. Therefore, accurate computational tools are urgently needed to accelerate the discovery of antiviral peptides.

In this study, we propose a multi-feature fusion model named **SKED-AVP** for antiviral peptide prediction. SKED-AVP combines **4-mer Word2Vec biological word embeddings** with **Ankh3-XL protein language model representations** to capture both local physicochemical patterns and long-range protein semantic information. Moreover, selective kernel attention and efficient channel attention are introduced to extract multi-scale sequence motifs and recalibrate informative feature channels. Finally, SKED-AVP achieves **94.62% ACC, 91.65% SN, 97.59% SP, 0.8940 MCC, and 0.9772 AUC** on the independent test set. 


<img width="9508" height="6707" alt="架构图" src="https://github.com/user-attachments/assets/9325fe59-0e04-4257-9778-374c4442c38d" />




