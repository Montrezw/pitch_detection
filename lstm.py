import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sarcasm import Sarcasm
from datasets import (GeneratorBasedBuilder, Version, DownloadManager, SplitGenerator, Split,
    Features, Value, BuilderConfig, DatasetInfo)


# Instantiate your dataset class
sarcasm = Sarcasm()

# Build the datasets
sarcasm.download_and_prepare()

# Access the datasets for training, validation, and testing
dataset_train = sarcasm.as_dataset(split='train')
dataset_validation = sarcasm.as_dataset(split='validation')
dataset_test = sarcasm.as_dataset(split='test')

# Convert datasets to pandas DataFrames
df_train = dataset_train.to_pandas()
df_validation = dataset_validation.to_pandas()
df_test = dataset_test.to_pandas()

# Extract comments & labels
X_train_text = df_train['comments'].tolist()
y_train = df_train['contains_slash_s'].tolist()
X_val_text = df_validation['comments'].tolist()
y_val = df_validation['contains_slash_s'].tolist()
X_test_text = df_test['comments'].tolist()
y_test = df_test['contains_slash_s'].tolist()

max_words = 10000 # Maximum number of words to keep
tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_val_seq = tokenizer.texts_to_sequences(X_val_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

max_len = 100

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print(f"Shape of X_train_pad: {X_train_pad.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_val_pad: {X_val_pad.shape}")
print(f"Shape of y_val: {y_val.shape}")
print(f"Shape of X_test_pad: {X_test_pad.shape}")
print(f"Shape of y_test: {y_test.shape}")

embedding_dim = 128

model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    LSTM(units=128, return_sequences=False),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid') # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 32

history = model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_pad, y_val))

loss, accuracy = model.evaluate(X_test_pad, y_test, batch_size=batch_size)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
