import os
import numpy as np
import pandas as pd
import librosa
from librosa import feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint

import os
top_db1 = 20

def make_predictions(model, file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    features = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index


def make_predictions1(model, file_path):
    y1, sr1 = librosa.load(file_path, sr=22050)
    y_t1, sr_t1 = librosa.effects.trim(y1, top_db=top_db1)
    mel = feature.melspectrogram(y=y_t1, sr=22050)
    mel_scaled = np.mean(mel.T, axis=0)
    features = mel_scaled.reshape(1, mel_scaled.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index


features = []
labels = []

# Specify the directory path
directory = 'arabic_dataset_small'  # 'arabic_dataset'
model_name = 'mfcc_100_trim_small_20.h5'  #'arabic1.h5'#'mel_spec_trim_model1.h5'
target_sr = 22050
nepo = 20

# Get all items (files and folders) in the directory
items = os.listdir(directory)

# Filter out only the folders
folders = sorted([item for item in items if os.path.isdir(os.path.join(directory, item))])
print(folders)


# Print the list of folders
print("Folders in the directory:", model_name)
ind = 0

# Load the audio file and resample it
file_names = []

for folder in folders:

    # Specify the directory path
    subdirectory = directory + "/" + folder + "/"

    # Get all items (files and folders) in the directory
    items = os.listdir(subdirectory)

    # Filter out only the files
    files = [item for item in items if os.path.isfile(os.path.join(subdirectory, item))]

    # Print the list of files
    for file in files:
        y1, sr1 = librosa.load(subdirectory + file, sr=target_sr)
        file_names.append(subdirectory + file)

        y_t1, sr_t1 = librosa.effects.trim(y1, top_db=top_db1)  # удаление тишины

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y1, sr=target_sr, n_mfcc=100)  # , n_mfcc=40
        mfccs_scaled = np.mean(mfccs, axis=1)
        #mfccs_scaled1 = np.std(mfccs, axis=1)
        #mfccs_ans = np.concatenate((mfccs_scaled, mfccs_scaled1), axis=0)
        #print(mfccs_scaled.shape, mfccs_scaled1.shape, mfccs_ans.shape)

        #mel = librosa.feature.melspectrogram(y=y_t1, sr=target_sr, n_mels=128)  # n_mels=40
        #mel_scaled = np.mean(mel, axis=1)
        #mel_scaled1 = np.std(mel, axis=1)
        #mel_ans = np.concatenate((mel_scaled, mel_scaled1), axis=0)
        #print(mel_scaled.shape, mel_scaled1.shape, mel_ans.shape)

        # Append features and labels
        features.append(mfccs_scaled)
        #features.append(mfccs_ans)
        #features.append(mel_ans)
        #all_ans = np.concatenate((mfccs_ans, mel_ans), axis=0)
        #features.append(all_ans)
        labels.append(ind)
    ind += 1

features = np.array(features)
labels = np.array(labels)

print(features.shape)
print(labels.shape)

#"""

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.2, random_state=42,
                                                    stratify=labels_onehot)


def proverka1(fff):
    print(len(fff))
    #print(fff[0])
    temp_mas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in fff:
        for j in range(28):
            if int(i[j]) != 0:
                temp_mas[j] += 1
    print(temp_mas)
    print(sum(temp_mas))


input_shape = (X_train.shape[1], 1)
model = Sequential()
model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define a callback to save the best model during training
checkpoint_callback2 = ModelCheckpoint(filepath=model_name, monitor='val_accuracy', save_best_only=True)  # 'val_loss'


model.fit(X_train, y_train, batch_size=25, epochs=nepo,
          validation_data=(X_test, y_test), verbose=1, callbacks=[checkpoint_callback2])

#i = 0
#cnt = 0
## Print the list of files
#for file_path in file_names:

#    predicted_class_index = make_predictions1(model, file_path)

#    if labels[i] == predicted_class_index:
#        cnt += 1
#    i += 1

#    print("Predicted Label:", predicted_class_index)
#print(cnt / len(file_names))

#"""


