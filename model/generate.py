import numpy as np
from tensorflow.keras.models import load_model
import json

# Load the character mappings from the file
with open('char_mappings.json', 'r') as f:
    mappings = json.load(f)
    chars = mappings['chars']
    char_to_idx = {k: int(v) for k, v in mappings['char_to_idx'].items()}
    idx_to_char = {int(k): v for k, v in mappings['idx_to_char'].items()}

# Load the trained model
model = load_model('poetry_generator_model.h5')  # Adjust path if necessary

# Function to generate poetry
def generate_poetry(seed_text, num_chars_to_generate=400, temperature=1.0):
    generated_text = seed_text
    for _ in range(num_chars_to_generate):
        x_pred = np.zeros((1, len(seed_text), len(chars)))
        
        # Vectorize the current sequence (seed_text)
        for t, char in enumerate(seed_text):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1.0
            else:
                # Handle unseen characters by using a placeholder or the most common character
                x_pred[0, t, char_to_idx[' ']] = 1.0
        
        # Predict the next character probabilities
        preds = model.predict(x_pred, verbose=0)[0]
        
        # Apply temperature to the predicted probabilities
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Sample the next character based on the predicted probabilities
        next_index = np.random.choice(len(chars), p=preds)
        next_char = idx_to_char[next_index]
        
        # Append the predicted character to the generated text
        generated_text += next_char
        
        # Shift the seed text to include the predicted character and truncate to maxlen
        seed_text = seed_text[1:] + next_char
    
    return generated_text

# Example usage
if __name__ == "__main__":
    maxlen = 40  # This should match the maxlen used during training
    
    # Example seed text
    seed_text = "From fairest creatures we desire increase,"
    
    generated_poetry = generate_poetry(seed_text=seed_text, num_chars_to_generate=400, temperature=0.5)
    print(generated_poetry)
