# Speaker recognition: Few-shot few-class open-set application for door opening and security
Speaker Recognition project using SRPL and ReDimNet model.
Finetuning pretrained speaker ReDimNet-b0 model and build a embedding spaces adapter for metric learn﻿﻿﻿ing (SRPL Tuning).
Achieved 98.1% testing accuracy on a 10-class open-set derived from the DR6 subset of the TIMIT dataset, 92.5% testing accuracy on a 5-class noise-augmented open-set of a self-collected Vietnamese dataset.  

This project leverages a **pretrained ReDimNet-b0** model and enhances speaker embedding learning with **Speaker Reciprocal Points Learning (SRPL)**. Designed for both **clean** and **noise-augmented** datasets, the system adapts embedding spaces for robust open-set speaker recognition.

---

## 🎯 Highlights

- 🔁 Fine-tuned a pretrained **ReDimNet-b0** speaker model
- 🔄 Integrated **SRPL** to adapt embedding space via metric learning
- 🧪 Achieved **98.1%** accuracy on a 10-class open-set from **TIMIT (DR6)**
- 🇻🇳 Achieved **92.5%** on a 5-class **Vietnamese noise-augmented open-set**

---

## 📁 Dataset Structure

Your dataset should follow the structure below:

```
dataset/
├── train/
│   ├── speaker_01/
│   │   ├── file1.wav
│   │   └── file2.wav
│   └── speaker_02/
│       ├── file1.wav
│       └── file2.wav
├── valid/
│   ├── speaker_01/
│   └── speaker_02/
├── outloader/
│   ├── unknown_01/
│   └── unknown_02/
└── test/
    ├── speaker_01/
    ├── speaker_02/
    ├── unknown_01/
    └── unknown_02/
```
Visualization of testing data in new embedding space (using t-SNE)

![image](https://github.com/user-attachments/assets/422c8ff5-47a4-4f0d-b47d-e709b5685baa)

