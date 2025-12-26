# Facial Verification using Siamese Neural Network

This project implements a **Facial Verification System** using a **Siamese Neural Network**, built completely from scratch.  
The model learns to determine whether two given face images belong to the same person by learning a similarity metric rather than performing direct classification.

---

## 📌 Project Overview

Facial verification answers the question:

**“Are these two face images of the same person?”**

Instead of assigning class labels, this system:
- Learns feature embeddings for faces
- Compares them using a distance metric
- Predicts similarity between image pairs

The project achieves **~98.5% validation accuracy** on unseen image pairs.

---

## 🧠 Core Concepts

- Siamese Neural Networks  
- One-shot / Few-shot Learning  
- Metric Learning  
- Contrastive Learning  
- Face Image Preprocessing  

---

## 🏗️ Model Architecture

The architecture consists of:
- Two identical CNN branches with shared weights
- Feature embedding extraction for each image
- **L1 distance layer** to compute similarity
- Fully connected layer for final prediction
Image A ──► CNN ──► Embedding A
│
|A - B|
│
Image B ──► CNN ──► Embedding B ──► Dense ──► Similarity Score


---

## 🗂️ Dataset & Preprocessing

- Dataset contains **10,000+ face image pairs**
- Preprocessing performed using **OpenCV**:
  - Face detection
  - Face alignment
  - Image resizing and normalization
- Preprocessing reduced noise and improved training convergence by ~20%

---

## ⚙️ Training Details

- **Loss Function:** Binary Cross Entropy  
- **Distance Metric:** L1 Distance  
- **Optimizer:** Adam  
- **Training Strategy:** Pair-based learning (positive & negative pairs)

---

## 📊 Results

- **Validation Accuracy:** ~98.5%
- Good generalization to unseen identities
- Stable convergence due to contrastive architecture

---

## 📁 Repository Structure
Siamese_Network/
│
├── Facial_Verification_1.ipynb # Model, training & evaluation notebook
├── data/ # Image pairs (not included)
├── models/ # Saved model weights
└── README.md


---
## 📚 References

- Siamese Neural Networks for One-shot Image Recognition  
- Contrastive Learning and Metric Learning research papers  
- FaceNet architecture concepts  

---

## 🔮 Future Improvements

- Triplet loss with hard negative mining  
- Real-time face verification using webcam  
- Model deployment using FastAPI / Streamlit  
- Improved backbone architectures (ResNet / EfficientNet)  

---

## 👩‍💻 Author

**Mahima Bachhav**  
B.Tech CSE, IIT (ISM) Dhanbad  
Interest: Computer Vision & Applied Machine Learning  

🔗 GitHub: https://github.com/Mahima-Bachhav


