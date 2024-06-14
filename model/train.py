import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
import json

# Download Shakespeare's Sonnets from Project Gutenberg
url = 'https://www.gutenberg.org/cache/epub/1041/pg1041.txt'
response = requests.get(url)
sonnets_text = response.text

# Extract the sonnets (they start after line 124 and end before line 2802)
sonnets = sonnets_text.split('\n')[124:2802]
sonnets = [line.strip() for line in sonnets if line.strip() != '']
sonnets_text = ' '.join(sonnets)

# Create character mappings
chars = sorted(list(set(sonnets_text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Save the character mappings to a file
with open('char_mappings.json', 'w') as f:
    json.dump({'chars': chars, 'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)

# Create sequences of characters
maxlen = 40  # Length of sequences
step = 3     # Step size to create overlapping sequences

sequences = []
next_chars = []

for i in range(0, len(sonnets_text) - maxlen, step):
    sequences.append(sonnets_text[i:i + maxlen])
    next_chars.append(sonnets_text[i + maxlen])

# Vectorization
X = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool_)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Define the model
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(X, y, batch_size=128, epochs=30)

# Save the model
model.save('poetry_generator_model.h5')

print('Model training and saving completed.')
