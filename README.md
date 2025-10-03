# Transformer-Based Models

Welcome to my repo of Transformer-based models! This repository showcases a step-by-step journey into the world of the Transformer architecture, starting with fundamental components and building up to advanced applications in both Natural Language Processing (NLP) and Computer Vision.

Each project is self-contained and demonstrates an end-to-end workflow, from model construction to training and inference.

---

## 🚀 Projects Overview

This repository contains three distinct projects, each building upon the core concepts of the Transformer.

### 1. Text Classifier
A Transformer Encoder model built from scratch in PyTorch to classify news headlines into categories. This project focuses on the fundamental self-attention mechanism and text processing pipeline.
- **[Read More & View Code](./text_classifier/readme.md)**

### 2. Vision Encoder (ViT)
An implementation of a Vision Transformer (ViT) that uses transfer learning to fine-tune a large, pre-trained model for a specific image classification task (CIFAR-10). This project explores the application of Transformers to the vision domain.
- **[Read More & View Code](./vision_encoder/readme.md)**

### 3. Vision-Language Model (VLM)
The most advanced project, this VLM combines a vision encoder and a text decoder for Visual Question Answering (VQA). It features a custom implementation of the SPIN attention mechanism to reduce model "hallucinations."
- **[Read More & View Code](./vision_language_model/readme.md)**

---

## 🧠 Core Architecture: The "Attention Is All You Need" Transformer

All projects in this repository are based on the principles introduced in the seminal paper "Attention Is All You Need" by Vaswani et al. (2017). The diagram below illustrates this foundational encoder-decoder architecture.

![Original Transformer Architecture](./assets/attention_research_1.png)

### The Encoder (Left Side)

The Encoder's job is to process an input sequence (e.g., a sentence) and build a rich, context-aware numerical representation of it.

-   **1. Input & Positional Encoding:** The input words are first converted into vectors (embeddings). Since the model has no inherent sense of order, a "positional encoding" vector is added to each word embedding to give it information about its position in the sequence.
-   **2. Multi-Head Self-Attention:** This is the core of the Transformer. For each word, the self-attention mechanism allows it to "look at" every other word in the input sequence. It calculates an "attention score" to determine how relevant each of the other words is to understanding the current word. This process builds a deeply contextual representation for every word. "Multi-head" means this is done multiple times in parallel, allowing the model to capture different types of contextual relationships simultaneously.
-   **3. Feed-Forward Network:** After attention, each word's representation is passed through a simple, identical feed-forward neural network for further processing.
-   **Add & Norm:** Each block contains residual connections ("Add") and layer normalization ("Norm") to help with training stability and information flow. The encoder consists of a stack of these identical layers (`Nx`).

### The Decoder (Right Side)

The Decoder's job is to take the Encoder's output and generate a new sequence (e.g., the translated sentence) one word at a time.

-   **1. Masked Multi-Head Self-Attention:** The decoder also uses self-attention, but with a crucial difference: it is "masked." When predicting the next word, the decoder is only allowed to look at the words it has *already* generated in the output sequence. It is prevented from "cheating" by looking ahead.
-   **2. Encoder-Decoder Attention (Cross-Attention):** This is the bridge between the encoder and decoder. In this step, the decoder pays attention to the **encoder's output**. This allows it to consider the full context of the *input* sentence when deciding which word to generate next in the *output* sentence. This is where the model learns alignments between different languages or modalities.
-   **3. Final Layers:** The output of the final decoder block is passed through a Linear layer and a Softmax function. This produces a probability distribution over the entire vocabulary, and the word with the highest probability is chosen as the next word in the sequence.

---

## 🛠️ Technology Stack
- Python
- PyTorch
- `timm` (PyTorch Image Models)
- Jupyter
- Matplotlib & Pillow
