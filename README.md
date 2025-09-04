# Domain Adapter for Speaker Embedding Transfer & Edge Deployment

This project designs and evaluates a **domain adapter** that maps embeddings from **ReDimNet** into a new space for **ARP Learning** (Adversarial Representation Perturbation Learning), enabling robust speaker classification and domain adaptation for **edge devices (Raspberry Pi 4)**.

- **Framework:** PyTorch / TorchAudio  
- **Techniques:** Domain Adapter, Mel-spectrogram features, ARP Learning  
- **Target Platform:** Raspberry Pi 4 (real-time inference optimized)  

---

## Features
- **ReDimNet backbone** for extracting robust speaker embeddings.  
- **Domain adapter module**: lightweight fully-connected + normalization → maps embeddings to ARP space.  
- **ARP Learning** for domain generalization (noise, channel shift, cross-language).  
- **Deployment** pipeline for Raspberry Pi 4
<img width="500" height="297" alt="image" src="https://github.com/user-attachments/assets/0e0e195e-3fc1-4373-a97b-028f377a14cb" />

---

## Results

| Dataset & Setting                                    | Accuracy |
|------------------------------------------------------|-----------|
| **TIMIT DR6 subset (10-class open-set)**             | **98.1%** |
| **Self-collected VN dataset (5-class, noise-augmented open-set)** | **92.5%** |

- Accuracy measured on **testing set** after domain adaptation.  
- Strong robustness under **noise augmentation** and **cross-domain evaluation**.  
<img width="498" height="310" alt="image" src="https://github.com/user-attachments/assets/4cb30119-e1bd-4465-8026-7756fb76de86" />



