# Emotion Classification

A machine learning project that trains and compares three CNN architectures for classifying facial emotions from images. Built with TensorFlow/Keras using the [6 Emotions for Image Classification](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes) dataset from Kaggle.

## Models Compared

1. **CNN (No Pooling)** : A basic convolutional network with no pooling or dropout. Serves as a baseline.
2. **CNN (With Pooling & Dropout)** : A more regularized CNN using MaxPooling and Dropout layers to reduce overfitting.
3. **Xception (Fine-tuned)** : A pre-trained Xception model with ImageNet weights, fine-tuned for emotion classification using transfer learning.

## Project Structure

```
├── run.py                        # Main training and evaluation script
├── Output every 5 epochs.xlsx    # Recorded results across training steps
├── archive.zip                   # Dataset archive
└── saved_models/                 # Generated at runtime : trained model checkpoints
```

## Setup

1. Download the [6 Emotions for Image Classification](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes) dataset from Kaggle.
2. Update the `original_data_dir` path in `run.py` to point to your local dataset folder.
3. Run `run.py` : models will train, evaluate, and display loss/accuracy plots. Checkpoints are saved to `saved_models/` and reloaded automatically on subsequent runs.
