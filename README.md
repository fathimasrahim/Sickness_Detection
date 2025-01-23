# Sickness Detection Model

This project involves building a machine learning model to classify audio data into two categories: `sick` or `not_sick`. Using extracted features from audio files, the model employs a neural network for binary classification.

## Dataset Details
The dataset comprises 845 audio files organized into two categories: `sick` and `not_sick`.  
- **Total Samples:** 845  
- **Total Sick Samples:** 385  
- **Total Not Sick Samples:** 460  

Each audio file's path and corresponding label are listed in a CSV file. **Mel Frequency Cepstral Coefficients (MFCCs)** are extracted from each audio file using the `librosa` library. These MFCC features are summarized by taking their mean across the time axis, resulting in a fixed-length vector of 128 features for each file. The features and labels are saved in `.npy` files to ensure efficient reuse in future runs.

## Model Architecture
The model is a fully connected neural network implemented using TensorFlow's Keras API. Below is a summary of the architecture:  
- **Input Layer:** Accepts a 128-dimensional feature vector.  
- **Hidden Layers:** Four layers with ReLU activation and the following neuron counts:  
  - Layer 1: 256 neurons  
  - Layer 2: 128 neurons  
  - Layer 3: 128 neurons  
  - Layer 4: 64 neurons  
- **Dropout Layers:** Applied after each hidden layer with a rate of 30% to reduce overfitting.  
- **Output Layer:** A single neuron with sigmoid activation for binary classification.  

## Training Details
- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Performance Metrics:** Loss and Accuracy  

Validation loss and accuracy are monitored during training to assess the model's learning progress.

## Model Performance
The trained model achieved a test accuracy in the range of **45% to 70%** across different training runs. The variability in accuracy may be due to:  
- The relatively small dataset size.  
- Inherent noise in the audio data influencing the model's performance during training.
