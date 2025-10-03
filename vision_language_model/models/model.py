# vision_language_model/src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

# --- Model Components ---

class SPINMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, top_k_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.top_k_heads = top_k_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None, is_cross_attention=False):
        B, _, _ = Q.size()
        Q_proj = self.W_Q(Q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K_proj = self.W_K(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V_proj = self.W_V(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        
        attn = F.softmax(scores, dim=-1)
        
        if is_cross_attention and not self.training:
            num_image_tokens = 197 # For ViT-Tiny
            image_attention_scores = attn[:, :, -1, :num_image_tokens].mean(dim=-1)
            top_k_indices = torch.topk(image_attention_scores, self.top_k_heads, dim=-1).indices
            suppression_mask = torch.zeros_like(image_attention_scores)
            suppression_mask.scatter_(1, top_k_indices, 1.0)
            V_proj = V_proj * suppression_mask.view(B, self.num_heads, 1, 1)

        out = torch.matmul(attn, V_proj).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_O(out)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SPINMultiHeadAttention(d_model, num_heads)
        self.cross_attn = SPINMultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(Q=x, K=memory, V=memory, is_cross_attention=True)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask):
        tgt_embed = self.pos_enc(self.embedding(tgt))
        for layer in self.layers:
            tgt_embed = layer(tgt_embed, memory, tgt_mask)
        return self.output_linear(tgt_embed)

class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.model.embed_dim

    def forward(self, x):
        return self.model.forward_features(x)

# --- Main VLM Model ---

class VLM(nn.Module):
    def __init__(self, vision_encoder, text_decoder, text_dim):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.vision_projection = nn.Linear(vision_encoder.embed_dim, text_dim)
        self.text_embedding = text_decoder.embedding

    def forward(self, image, question_ids, answer_ids, tgt_mask):
        vision_memory = self.vision_projection(self.vision_encoder(image))
        question_embed = self.text_embedding(question_ids)
        memory = torch.cat([vision_memory, question_embed], dim=1)
        return self.text_decoder(answer_ids, memory, tgt_mask)
        
# --- Generation Helper ---

def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.1, generated_ids=None):
    logits = logits / max(1e-8, temperature)
    if repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            if token_id < len(logits):
                logits[token_id] = logits[token_id] / repetition_penalty if logits[token_id] > 0 else logits[token_id] * repetition_penalty
    if top_k is not None and top_k > 0:
        _, topk_idx = torch.topk(logits, min(top_k, logits.size(-1)))
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_idx, logits.gather(-1, topk_idx))
        logits = mask
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 0] = False
        sorted_logits[cutoff] = float('-inf')
        logits.scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()