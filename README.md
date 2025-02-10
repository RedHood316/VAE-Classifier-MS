# Variational Autoencoders with a Classifier Head

## What are Variational Autoencoders (VAEs)?
Variational Autoencoders (VAEs) are a type of deep learning model used for learning meaningful data representations. They encode input data into a **latent space**, which captures essential features in a compressed format. Unlike traditional autoencoders, VAEs use **probabilistic modeling** to generate diverse representations, making them useful for tasks like image generation and classification.

### How VAEs Work:
1. **Encoder:** Converts input images into a latent representation.
2. **Latent Space:** A compressed version of the data, capturing meaningful patterns.
3. **Decoder:** Attempts to reconstruct the original image from the latent representation.
4. **Classifier Head (Optional):** Uses the latent space representation to classify images into specific categories.

![VAE Structure](https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/30060413-4770-48d0-920d-bc9f438c4fcf)

---
## Goal of This Study
This research focuses on comparing two models—**PCAEClassifier** and **BetaVAEClassifier**—for analyzing **brain MRI scans** of **Multiple Sclerosis (MS) patients**. The models aim to:
- Identify **white matter lesions (WMLs)** in MRI scans.
- Classify MRIs into two categories: **with lesions** vs. **without lesions**.
- Improve **accuracy, precision, recall, and F1-score** through different model configurations.

### Key Steps in Data Preparation:
✔ **Data Augmentation** – Enhancing dataset variety.
✔ **Preprocessing** – Cleaning and standardizing images.
✔ **Compression & Normalization** – Making images suitable for deep learning.

---
## Comparing PCAEClassifier and BetaVAEClassifier
### 1️⃣ PCAEClassifier (Deterministic Encoder)
✔ Uses fixed convolutional layers to extract features.
✔ Provides **consistent, deterministic outputs**.
✔ Converges faster during training.

### 2️⃣ BetaVAEClassifier (Probabilistic Encoder)
✔ Uses a **probabilistic encoding** method.
✔ Captures **mean features** and **variability**.
✔ Offers **richer representations** of input images.

![Encoder Types](https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/ad00b7a7-8757-4924-9fa3-053fe4a3ea4b)

---
## Model Performance Comparison
### BetaVAEClassifier Results (with different \(\beta\) values):
| Beta (\(\beta\)) | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|-------------|------------|------------|-----------|-----------|
| 5           | 83.56      | 84.13      | 82.52     | 83.32     |
| 10          | 85.56      | 90.78      | 78.59     | 84.25     |
| 20          | 87.61      | 87.00      | 87.00     | 87.00     |

### PCAEClassifier Results (with different sparsity parameters \(S\)):
| Sparsity (S) | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|-------------|------------|------------|-----------|-----------|
| 5           | 70.82      | 78.11      | 57.57     | 66.29     |
| 10          | 84.08      | 85.00      | 83.00     | 84.00     |
| 20          | 88.82      | 90.38      | 86.40     | 88.34     |

#### 🔎 Key Takeaways:
✔ Increasing \(\beta\) in BetaVAE improves accuracy but affects recall.
✔ Higher sparsity \(S\) in PCAEClassifier enhances accuracy and F1-score.
✔ Model **hyperparameters significantly impact classification performance**.

---
## Architecture Breakdown
### 1️⃣ PCAEClassifier Architecture:
✔ **Encoder:** Uses convolutional layers to extract features.
✔ **Classifier:** Fully connected layers process the latent representation.
✔ **(Optional) Decoder:** Reconstructs images for improved training stability.

![PCAEClassifier](https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/fc9dcef6-f024-4120-ad9b-f4591cddad63)

### 2️⃣ BetaVAEClassifier Architecture:
✔ **Encoder:** Uses convolutional layers and encodes data as a **distribution**.
✔ **Classifier Head:** Uses latent representation to classify MRIs.
✔ **(Optional) Decoder:** Helps regularize training and visualize latent space.

![BetaVAEClassifier](https://github.com/hajami0802/Variational-Autoencoders-Classifier/assets/169827483/e965766a-615d-44f0-8a0c-e30dcba9f63a)

---
## Conclusion
📌 **BetaVAEClassifier** provides richer feature extraction but requires careful tuning of \(\beta\).
📌 **PCAEClassifier** is simpler but benefits from sparsity tuning.
📌 Both models show promise in **medical image analysis**, particularly in **Multiple Sclerosis (MS) classification**.
📌 Hyperparameter selection plays a crucial role in balancing **accuracy, precision, recall, and F1-score**.

---
## References
- Kingma, D. P., & Welling, M. (2019). *An introduction to variational autoencoders*. Foundations and Trends in Machine Learning.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Muslim, A. M., et al. (2022). *Brain MRI dataset of multiple sclerosis with consensus manual lesion segmentation and patient meta information*. Data in Brief.
- Carass, A., et al. (2017). *Longitudinal multiple sclerosis lesion segmentation: resource and challenge*. NeuroImage.

✅ **This study highlights how Variational Autoencoders (VAEs) with classifier heads can enhance MRI-based disease diagnosis. Future work can explore additional deep learning architectures and fine-tune hyperparameters for even better results!** 🚀

