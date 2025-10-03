# text_classifier/src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :].to(x.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q, self.W_K, self.W_V, self.W_O = [nn.Linear(d_model, d_model) for _ in range(4)]
    def forward(self, x, mask=None):
        B, L, _ = x.size()
        Q, K, V = [l(x).view(B, L, self.num_heads, self.d_k).transpose(1,2) for l, x in zip((self.W_Q, self.W_K, self.W_V), (x,x,x))]
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2); scores = scores.masked_fill(mask == 0, float("-1e9"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, L, self.num_heads * self.d_k)
        return self.W_O(out), attn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        x_attn, _ = self.mha(x, mask)
        x = self.norm1(x + x_attn)
        x = self.norm2(x + self.ff(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__(); self.layers = nn.ModuleList([TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
    def forward(self, x, mask=None):
        for layer in self.layers: x = layer(x, mask)
        return x, None

class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, input_ids, mask=None):
        embeds = self.embedding_layer(input_ids)
        embeds_pos = self.pos_enc(embeds)
        enc_out, _ = self.encoder(embeds_pos, mask)
        return self.classifier(enc_out[:, 0, :])