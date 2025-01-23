import pandas as pd
import numpy as np
import os
import csv
import tqdm
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

train= r"C:\Users\srafa\OneDrive\Desktop\sickness_detection\train"

csv_file="data.csv"

# preparing CSV file from the train data with the filename and its class
data=[]

for class_name in ["sick","not_sick"]:
    class_folder=os.path.join(train, class_name)
    if os.path.exists(class_folder):
        for file_name in os.listdir(class_folder):
            if file_name.endswith(".wav"):
                full_path = os.path.join(class_folder, file_name)
                data.append([full_path, class_name])

with open(csv_file, mode="w", newline="") as file:
    writer=csv.writer(file)
    writer.writerow(["File Name", "Class"])
    writer.writerows(data)


label2int={
    "sick":1,
    "not_sick":0
}

# Function to extract features from .wav files
def extract_features(file_path, vector_length=128):
    """Extract MFCC features from an audio file."""
    try:
        audio_data, sr = librosa.load(file_path, sr=None)  # Load audio
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=vector_length)  # Extract MFCCs
        return np.mean(mfccs.T, axis=0)  # Average across time axis to get a single feature vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(vector_length)  # Return zero vector on error


def load_data(vector_length=128):
    """ Load sickness detection dataset.
    If the features and labels are already saved in 'results' folder, load them directly.
    otherwise process the data from csv file and save features and labels for future use."""
    #make sure that the results folder exists
    if not os.path.isdir("results"):
        os.mkdir("results")

    features_path="results/features.npy"
    labels_path="results/labels.npy"
    # if features & labels already loaded individually and bundled, load them from there instead
    if os.path.isfile(features_path) and os.path.isfile(labels_path):
        X=np.load(features_path)
        y=np.load(labels_path)
        return X, y
    
    #read dataframe
    df=pd.read_csv(r"C:\Users\srafa\OneDrive\Desktop\sickness_detection\data.csv")
    n_samples=len(df)   #total samples
    n_sick_samples= len(df[df["Class"]=="sick"])  #total sick samples
    n_notsick_samples=len(df[df["Class"]=="not_sick"])  #total not sick samples
    print("Total Samples:", n_samples)
    print("Total sick samples:",n_sick_samples)
    print("Total not sick samples:", n_notsick_samples)
    
    #initialize an empty array for all audio features
    X=np.zeros((n_samples,vector_length))
    y=np.zeros((n_samples, 1))  # initialize an empty array for all audio labels (1 for sick and 0 for not sick)

    for i, (filename,label) in tqdm.tqdm(enumerate(zip(df["File Name"],df["Class"])), "Loading data", total=n_samples):
        features = extract_features(filename, vector_length)
        X[i]=features
        y[i]=label2int[label]

    # save the audio features and labels into files
    np.save(features_path,X)
    np.save(labels_path,y)
    return X,y

def splitdata(X, y, test_size=0.1, valid_size=0.1):
    # split training set and testing set
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size,random_state=7)

    # split training set and validation set
    X_train, X_valid, y_train, y_valid=train_test_split(X_train, y_train, test_size=valid_size, random_state=7)
    #return a dictionary of values
    return{
        "X_train":X_train,
        "X_valid":X_valid,
        "X_test":X_test,
        "y_train":y_train,
        "y_valid":y_valid,
        "y_test":y_test
    }

def create_model(vector_length=128):
    """
    Create and compile a model for sickness detection.
    """
    model=Sequential()
    model.add(Dense(256,input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function, 0 means not sick, 1 means sick
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's sick/not sick classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    # print summary of the model
    model.summary()
    return model
