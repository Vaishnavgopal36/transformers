# vision_language_model/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os

# Import our custom modules
from src.model import VisionEncoder, TransformerDecoder, VLM
from src.dataset import Flickr8kVQADataset, SimpleTokenizer

def create_causal_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

def main():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    D_MODEL = 256
    NUM_HEADS = 8
    D_FF = 1024
    NUM_DECODER_LAYERS = 4
    BATCH_SIZE = 16
    MAX_LEN = 60
    WARMUP_EPOCHS = 10
    FINETUNE_EPOCHS = 5
    WARMUP_LR = 1e-4
    FINETUNE_LR = 1e-6
    
    # Paths (relative to the root `transformer_portfolio` folder)
    IMAGE_DIR = 'flickr8k/Images'
    CAPTIONS_FILE = 'flickr8k/captions.txt'
    VOCAB_PATH = 'finetuned_vocab.json'
    MODEL_DIR = 'vision_language_model/models'
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'vlm_spin_best_model.pth')

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Setup Data ---
    tokenizer = SimpleTokenizer(VOCAB_PATH)
    VOCAB_SIZE = len(tokenizer.vocab)
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = Flickr8kVQADataset(IMAGE_DIR, CAPTIONS_FILE, image_transforms, tokenizer, MAX_LEN)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    def collate_fn(batch):
        images, q, a_in, a_out = zip(*batch)
        return torch.stack(images, 0), torch.stack(q, 0), torch.stack(a_in, 0), torch.stack(a_out, 0)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"Dataset ready: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # --- Setup Model, Loss, and Optimizers ---
    vision_encoder = VisionEncoder(model_name='vit_tiny_patch16_224')
    text_decoder = TransformerDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_DECODER_LAYERS)
    vlm = VLM(vision_encoder, text_decoder, text_dim=D_MODEL).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<pad>'])
    
    optimizer_warmup = torch.optim.AdamW(
        list(vlm.text_decoder.parameters()) + list(vlm.vision_projection.parameters()), lr=WARMUP_LR
    )
    optimizer_finetune = torch.optim.AdamW(vlm.parameters(), lr=FINETUNE_LR)

    # --- Training Loop ---
    best_val_loss = float('inf')

    # Stage 1: Decoder Warm-up
    print("\n--- STAGE 1: DECODER WARM-UP ---")
    for epoch in range(WARMUP_EPOCHS):
        vlm.vision_encoder.eval()
        vlm.text_decoder.train()
        vlm.vision_projection.train()
        for param in vlm.vision_encoder.parameters():
            param.requires_grad = False

        total_train_loss = 0
        for images, q_ids, a_in_ids, a_out_ids in tqdm(train_dataloader, desc=f"Warm-up Epoch {epoch+1}"):
            images, q_ids, a_in_ids, a_out_ids = images.to(device), q_ids.to(device), a_in_ids.to(device), a_out_ids.to(device)
            tgt_mask = create_causal_mask(a_in_ids.size(1), device)
            
            logits = vlm(images, q_ids, a_in_ids, tgt_mask)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), a_out_ids.view(-1))
            
            optimizer_warmup.zero_grad()
            loss.backward()
            optimizer_warmup.step()
            total_train_loss += loss.item()
        
        print(f"Avg Train Loss: {total_train_loss / len(train_dataloader):.4f}")
        # (A full implementation would include a validation loop here to save the best model)

    # Stage 2: Full Fine-tuning
    print("\n--- STAGE 2: FULL FINE-TUNING ---")
    for epoch in range(FINETUNE_EPOCHS):
        vlm.train()
        for param in vlm.vision_encoder.parameters():
            param.requires_grad = True

        total_train_loss = 0
        for images, q_ids, a_in_ids, a_out_ids in tqdm(train_dataloader, desc=f"Finetune Epoch {epoch+1}"):
            images, q_ids, a_in_ids, a_out_ids = images.to(device), q_ids.to(device), a_in_ids.to(device), a_out_ids.to(device)
            tgt_mask = create_causal_mask(a_in_ids.size(1), device)
            
            logits = vlm(images, q_ids, a_in_ids, tgt_mask)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), a_out_ids.view(-1))
            
            optimizer_finetune.zero_grad()
            loss.backward()
            optimizer_finetune.step()
            total_train_loss += loss.item()
            
        avg_loss = total_train_loss / len(train_dataloader)
        print(f"Avg Finetune Loss: {avg_loss:.4f}")

        # Save model after each finetuning epoch
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(vlm.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved to {BEST_MODEL_PATH}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()