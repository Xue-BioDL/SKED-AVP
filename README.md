## Prediction of Antiviral Peptides

Peptides capable of inhibiting viral infections are known as antiviral peptides (AVPs). They exert antiviral activity through diverse mechanisms, including blocking viral entry, interfering with viral replication, and modulating virus–host interactions. However, the experimental identification of AVPs is often time-consuming, labor-intensive, and costly. Therefore, efficient and accurate computational methods are needed to accelerate the discovery of candidate AVPs.

In this study, we propose **SKED-AVP**, a dual-branch deep learning framework based on multi-feature fusion for sequence-based AVP prediction. SKED-AVP integrates **4-mer Word2Vec embeddings**, which capture local sequence and compositional patterns, with **Ankh3-XL protein language model representations**, which encode long-range contextual information. Selective kernel attention (SKA) is used to adaptively capture multiscale sequence patterns, while efficient channel attention (ECA) recalibrates informative feature channels. On the independent test set, SKED-AVP achieved an **Acc of 94.62%, Sn of 91.65%, Sp of 97.59%, MCC of 0.8940, and AUC of 0.9772**.


<img width="3802" height="2665" alt="a3365cd6c3a53b128c6ad0e999594d81" src="https://github.com/user-attachments/assets/6e909900-2b51-4e4b-9d26-3bda60d6de66" />





