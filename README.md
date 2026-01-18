# Query-by-Humming-Audio-Classifier

Query-by-Humming: Sequential Audio Classification

An end-to-end Machine Learning project focused on identifying songs from human hums and whistles. This project explores the critical transition from static feature engineering to sequential deep learning using the MLEnd Hums and Whistles II Dataset.

Project Overview

The goal is to solve an 8-class supervised classification problem. The core research challenge was determining if time-collapsed statistical features (Summary-Features) provide enough signal for song identification, or if temporal context (Sequential-Features) is mandatory for success.

Key Findings

Classical ML (Pipeline A): Achieved 23.00% accuracy using PCA-reduced static features.

Sequential DL (Pipeline B): Achieved 38.75% accuracy using a 1D-CNN.

Conclusion: Temporal context is essential; sequence-aware modeling provided a 15.75% performance gain over traditional baselines.

Technical Stack

Audio Processing: Librosa (MFCC, Chroma, ZCR extraction)

Deep Learning: TensorFlow/Keras (1D-Convolutional Neural Networks)

Classical ML: scikit-learn (SVM, Random Forest, XGBoost, PCA)

Data Science: NumPy, Pandas, Matplotlib, Seaborn

Pipeline Architectures

Pipeline A: Static ML

Feature Extraction: 350+ statistical features (means/std of spectral descriptors).

Preprocessing: Standard Scaling + PCA (reduced to 65 components).

Modeling: Trained via Stratified 5-Fold Cross-Validation.

Pipeline B: Sequential Deep Learning

Transformation: Audio converted to $431 \times 13$ MFCC tensors.

Architecture: 1D-CNN with Conv1D blocks, MaxPooling, and Dense layers.

Regularization: Batch Normalization, 0.5 Dropout, and Early Stopping to manage the small dataset ($N=800$).


🚀 How to Use

Environment: Run the provided .ipynb notebook in Google Colab.

Data: Ensure the MLEnd Hums and Whistles II dataset is accessible in your directory.

Execution: Run all cells to reproduce feature extraction, model training, and evaluation plots.

