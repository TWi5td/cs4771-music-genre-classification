# CS 4771: Undergraduate Project Proposal

## Problem Definition  
I propose to build a machine learning model to classify the genre of a song based on its audio features. This application is useful for music recommendation systems, streaming platforms, and personal music organization. Automating genre classification helps improve user experience by enabling smarter playlists, recommendations, and content discovery.  

## Machine Learning Task  
This is a **classification** task. The model will take in song audio data (or extracted features) and predict one of several music genres (e.g., rock, pop, classical, jazz, etc.).  

## Data  
We will use publicly available datasets, such as:  
- **GTZAN Dataset**: 1,000 audio tracks across 10 genres (rock, pop, jazz, blues, etc.).  
- Optionally, the **Spotify API**, which provides audio features such as tempo, energy, loudness, and key.  

Data preprocessing will include extracting numerical features from raw audio (using libraries like **librosa**) or converting audio into spectrograms for image-based classification.  

## Project Plan  

### Goals  
- Collect and preprocess audio data.  
- Train baseline models (logistic regression, k-NN, random forest) using audio features.  
- Implement a CNN to classify spectrograms and compare performance.  
- Evaluate models with accuracy, precision, recall, and confusion matrices.  
- Deliver a working music genre classifier that accepts audio input and outputs a genre prediction.  

### Potential Challenges  
- Limited dataset size.  
- Genre overlap (songs may reasonably fit into multiple categories).  
- Computational cost of spectrogram-based CNN training.  

# Project Timeline: Music Genre Classification

This timeline outlines weekly milestones for the CS 4771 project.  
Total estimated duration: **5 weeks (~40–50 hours)**.

---

## Step 1 – Setup & Proposal
- [ ] Create GitHub repository and initial project structure.
- [ ] Write project proposal (`proposal/project_proposal.md`).
- [ ] Install dependencies (`librosa`, `scikit-learn`, `matplotlib`, `tensorflow` or `pytorch`).
- [ ] Verify environment by running a test script (e.g., load audio file, extract basic features).

**Estimated Time:** 4–6 hours  

---

## Step 2 – Data Collection & Preprocessing
- [ ] Download **GTZAN dataset** (or alternative).
- [ ] Explore dataset: number of tracks, genre distribution, audio quality.
- [ ] Extract audio features (MFCCs, tempo, chroma, spectral contrast) using **librosa**.
- [ ] Save extracted features into a structured format (CSV, NumPy arrays, or pandas DataFrame).

**Estimated Time:** 8–12 hours  

---

## Step 3 – Baseline Models
- [ ] Train baseline classifiers (logistic regression, k-NN, random forest) on extracted features.
- [ ] Evaluate models using accuracy and confusion matrices.
- [ ] Document results in a Jupyter notebook (`notebooks/baseline_models.ipynb`).

**Estimated Time:** 8 hours  

---

## Step 4 – Deep Learning Model
- [ ] Convert audio to spectrogram images.
- [ ] Build and train a **Convolutional Neural Network (CNN)** on spectrograms.
- [ ] Experiment with regularization (dropout, batch normalization) to reduce overfitting.
- [ ] Compare CNN results with baseline models.

**Estimated Time:** 12–16 hours  

---

## Step 5 – Evaluation & Final Report
- [ ] Evaluate all models (accuracy, precision, recall, F1-score).
- [ ] Analyze misclassifications and genre overlaps.
- [ ] Write final documentation and update `README.md`.
- [ ] Ensure repo is clean and reproducible (requirements.txt, clear folder structure).

**Estimated Time:** 6–10 hours  

---

## Stretch Goals (Optional)
- [ ] Use **Spotify API** to fetch real-world audio features and test the model.  
- [ ] Try other deep learning architectures (RNN, transformer-based).  
- [ ] Deploy as a simple demo app (e.g., Streamlit web app where user uploads a track snippet).  

---
