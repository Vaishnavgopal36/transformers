

import torch
import re
import json
import os
from src.model import EncoderClassifier

# ==============================================================================
# PART 1: HYPERPARAMETERS & HELPERS
# ==============================================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Hyperparameters (MUST MATCH THE SAVED MODEL)
D_MODEL = 128
NUM_HEADS = 8
D_FF = 512
NUM_ENCODER_LAYERS = 4
NUM_CLASSES = 4
MAX_SEQ_LEN = 256

# ----- Helper Functions -----
def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

class_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# ==============================================================================
# PART 2: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- Step 1: Load the Vocabulary ---
    VOCAB_PATH = 'text_classifier/vocab.json'
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: Vocabulary file not found at {VOCAB_PATH}")
        exit()

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary loaded from {VOCAB_PATH}. Size: {VOCAB_SIZE}")

    # --- Step 2: Load the Trained Model ---
    MODEL_SAVE_PATH = 'text_classifier/ag_news_transformer.pth'
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        exit()

    model = EncoderClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_ENCODER_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- Step 3: The Interactive Loop ---
    print("\n--- AG News Classifier ---")
    print("Enter a news headline to classify its category.")

    while True:
        text = input("\nHeadline (or type 'quit' to exit): ")
        if text.lower() == 'quit':
            break

        tokens = ['[CLS]'] + simple_tokenizer(text)
        token_ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
        if len(token_ids) < MAX_SEQ_LEN:
            token_ids += [vocab["<pad>"]] * (MAX_SEQ_LEN - len(token_ids))
        else:
            token_ids = token_ids[:MAX_SEQ_LEN]
            
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        mask = (input_tensor != vocab["<pad>"]).to(device)

        with torch.no_grad():
            logits = model(input_tensor, mask=mask)

        pred_index = torch.argmax(logits, dim=1).item()
        category = class_map.get(pred_index, "Unknown")

        print(f"--> Predicted Category: {category}")

    print("\nExiting. Goodbye!")