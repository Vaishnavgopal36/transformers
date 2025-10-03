# Vision-Language Model (VLM) with SPIN for Hallucination Reduction

## 🎯 Project Goal

This project implements a sophisticated Vision-Language Model (VLM) for Visual Question Answering (VQA). The model is built from scratch using PyTorch and combines a Vision Transformer (ViT) encoder with a custom Transformer decoder.

The key innovation of this project is the implementation of **SPIN (Image-Guided Head Suppression)**, a technique designed to mitigate model "hallucinations." By forcing the text decoder to ground its output in the visual information provided by the image, SPIN helps the model generate more accurate and contextually relevant answers.

---

## 🛠️ Technology Stack

- **Python 3.8+**
- **PyTorch:** The core deep learning framework.
- **`timm` (PyTorch Image Models):** Used for the pre-trained Vision Transformer backbone.
- **Pillow (PIL):** Used for loading and manipulating images.

---

## 🧠 Architecture Overview

The model is an encoder-decoder architecture designed to process both an image and a text prompt (a question) to generate a text-based answer.

### Explanation of Components:

1.  **Vision Encoder:**
    -   We use a pre-trained **`vit_tiny_patch16_224`** model from `timm` as our vision backbone.
    -   Its role is to take an input image and transform it into a sequence of vector representations (embeddings), where each vector corresponds to a patch of the original image.

2.  **Multimodal Fusion:**
    -   The input text question is also converted into a sequence of embeddings using a standard `nn.Embedding` layer.
    -   These two sequences (image embeddings and question embeddings) are then **concatenated** to form a single, unified "memory" sequence. This combined memory provides the full context (both visual and textual) to the decoder.

3.  **Transformer Decoder with SPIN:**
    -   This is the generative part of the model. It's a custom-built Transformer decoder that produces the answer one word at a time.
    -   It consists of two main attention mechanisms in each layer:
        -   **Masked Self-Attention:** Allows the decoder to look at the words it has *already generated* in the answer sequence to maintain coherence.
        -   **Cross-Attention:** This is the most critical step. For each new word it generates, the decoder "looks at" the entire combined memory (image + question) to decide which word comes next.

4.  **SPIN (Image-Guided Head Suppression):**
    -   **Problem:** Standard VLMs sometimes "hallucinate" by generating text that is plausible but factually disconnected from the image. This happens when the cross-attention mechanism learns to ignore the visual embeddings and relies only on the text part of its memory.
    -   **Solution:** SPIN modifies the cross-attention mechanism during inference.
        1.  It calculates how much attention each "head" in the multi-head attention is paying to the image embeddings.
        2.  It identifies the **top-K heads** that are most focused on the image.
        3.  It then creates a "suppression mask" that **zeros out the output of all other heads**—the ones that were ignoring the image.
    -   **Result:** This forces the decoder to base its next word prediction primarily on information that is grounded in the visual input, significantly reducing hallucinations.

---

## 💿 Dataset and Preprocessing

-   **Dataset:** The model is trained on the **Flickr8k** dataset.
-   **Synthetic VQA:** To train the model for a question-answering task, the original image captions are converted into a synthetic VQA format. We pair each image with a generic question (e.g., "Describe the image in detail.") and use the original caption as the ground-truth answer.
-   **Prompt Templating:** We use a `USER: <image>\n{question}\nASSISTANT:` template during training and inference. This structured format helps the model understand the task is conversational and prevents it from simply repeating the input question.

---

## ⚙️ Workflow & Usage

1.  **Setup the Environment:**
    -   Create a virtual environment and install the required packages from a `requirements.txt` file (you can combine the lists from the previous projects).
    -   Place the `flickr8k` data folder and `finetuned_vocab.json` file in the root directory.

2.  **Train the Model:**
    -   To train the model from scratch, run the training script from the root `transformer_portfolio` directory:
        ```bash
        python vision_language_model/train.py
        ```
    -   This script will execute the two-stage training process (warm-up and fine-tuning) and save the best-performing model to **`vision_language_model/models/vlm_spin_best_model.pth`**. This will take a considerable amount of time.

3.  **Run Inference:**
    -   Once the model is trained, you can use the inference script to ask questions about an image.
    -   Run the command from the root directory, providing a path to an image and an optional prompt.
        ```bash
        python vision_language_model/inference.py --image path/to/your/image.jpg --prompt "What is the person doing in the image?"
        ```

---