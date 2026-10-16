# CodeAlpha: Emotion Recognition from Speech 🎙️🎭

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Librosa](https://img.shields.io/badge/Library-Librosa-green.svg)](https://librosa.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Project Overview
This project was developed as part of the **CodeAlpha Machine Learning Internship (Task 2)**. The primary objective is to build an intelligent speech signal processing and deep learning pipeline capable of classifying human emotional states (*Happy*, *Sad*, *Angry*, *Neutral*, and *Fearful*) from speech audio recordings.

Speech Emotion Recognition (SER) plays a crucial role in human-computer interaction, virtual assistants, call-center sentiment analysis, and mental health monitoring.

---

## 🚀 Key Features & Approach

1. **Speech Signal Processing & Feature Extraction:**
   - **MFCCs (Mel-Frequency Cepstral Coefficients):** Extracted 40 MFCC features to capture spectral characteristics and timbre of speech.
   - **Chroma STFT:** Captured pitch and harmonic variations.
   - **Spectral Contrast:** Analyzed frequency contrast to differentiate emotional speech dynamics.

2. **Data Preprocessing & Normalization:**
   - Aggregated feature vectors using mean summary statistics across time frames.
   - Applied `StandardScaler` to ensure zero-mean and unit variance across spectral descriptors.
   - Categorical label encoding for emotion classes.

3. **Machine Learning & Neural Network Classifiers:**
   - **Random Forest Classifier:** Ensemble decision tree approach providing a strong baseline.
   - **Multi-Layer Perceptron (MLP) Neural Network:** Deep architecture with hidden layers `(128, 64)` to capture complex non-linear acoustic patterns.

4. **Evaluation Metrics:**
   - Evaluated using **Precision**, **Recall**, **F1-Score**, and **Overall Accuracy** across individual emotion categories.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Audio Processing:** Librosa, SoundFile
- **Machine Learning & Data Processing:** Scikit-Learn, NumPy, Pandas

---

## 📂 Repository Structure
```text
CodeAlpha_EmotionRecognition/
│
├── .gitignore                          # Git ignore rules (excludes .venv, checkpoints, etc.)
├── CodeAlpha_EmotionRecognition.ipynb  # Jupyter Notebook with audio feature extraction & model development
├── CodeAlpha_EmotionRecognition.py     # Clean Python executable script
├── README.md                           # Comprehensive Documentation
└── requirements.txt                    # List of required Python packages