# vision_language_model/inference.py

import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

# Import our custom modules
from src.model import VisionEncoder, TransformerDecoder, VLM, sample_next_token
from src.dataset import SimpleTokenizer

def generate_caption(vlm_model, image_tensor, tokenizer, text_prompt, device, max_len=50):
    vlm_model.eval()
    end_token_id = tokenizer.vocab['<end>']
    
    full_prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
    prompt_ids = tokenizer.tokenize(full_prompt)
    generated_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    with torch.no_grad():
        image_memory = vlm_model.vision_projection(vlm_model.vision_encoder(image_tensor.unsqueeze(0)))
        prompt_embed = vlm_model.text_embedding(generated_ids)
        memory = torch.cat([image_memory, prompt_embed], dim=1)
        
        current_ids = generated_ids
        for _ in range(max_len):
            tgt_mask = torch.triu(torch.ones(current_ids.size(1), current_ids.size(1), device=device), diagonal=1).bool()
            output = vlm_model.text_decoder(current_ids, memory, tgt_mask)
            next_logits = output[0, -1, :]
            
            prev_ids_for_penalty = current_ids.squeeze(0).tolist()[len(prompt_ids):]
            next_tok = sample_next_token(next_logits, generated_ids=prev_ids_for_penalty)
            
            if next_tok == end_token_id:
                break
            current_ids = torch.cat([current_ids, torch.tensor([[next_tok]], device=device)], dim=1)
    
    return tokenizer.ids_to_sentence(current_ids.squeeze(0).tolist()[len(prompt_ids):])


def main():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters (must match the trained model)
    D_MODEL = 256
    NUM_HEADS = 8
    D_FF = 1024
    NUM_DECODER_LAYERS = 4
    
    # Paths
    VOCAB_PATH = 'finetuned_vocab.json'
    MODEL_PATH = 'vision_language_model/models/vlm_spin_best_model.pth'

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate a caption for an image using the VLM.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--prompt', type=str, default="Describe the image.", help='Text prompt for the model.')
    args = parser.parse_args()

    # --- Setup ---
    tokenizer = SimpleTokenizer(VOCAB_PATH)
    VOCAB_SIZE = len(tokenizer.vocab)
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Load Model ---
    vision_encoder = VisionEncoder(model_name='vit_tiny_patch16_224')
    text_decoder = TransformerDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_DECODER_LAYERS)
    vlm = VLM(vision_encoder, text_decoder, text_dim=D_MODEL).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
        return
        
    vlm.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded successfully.")

    # --- Generate ---
    try:
        image = Image.open(args.image).convert("RGB")
        image_tensor = image_transforms(image).to(device)
    except FileNotFoundError:
        print(f"Error: Image not found at {args.image}")
        return

    print(f"\nImage: {args.image}")
    print(f"Prompt: {args.prompt}")
    
    caption = generate_caption(vlm, image_tensor, tokenizer, args.prompt, device)
    
    print("\n--> Generated Caption:")
    print(caption)

if __name__ == "__main__":
    main()