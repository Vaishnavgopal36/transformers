# Vision Encoder: Fine-Tuning a Vision Transformer (ViT)

## 🎯 Project Goal

The goal of this project is to demonstrate the process of **transfer learning** with a modern, transformer-based vision model. We take a large Vision Transformer (ViT) pre-trained on the massive ImageNet dataset and fine-tune it to perform a specific image classification task on a smaller dataset, CIFAR-10.

The final output is a specialized vision encoder (`vit_cifar10_finetuned.pth`) that is highly effective at recognizing images from the 10 CIFAR-10 classes.

---

## 🛠️ Technology Stack

- **Python 3.8+**
- **PyTorch:** The core deep learning framework.
- **`torchvision`:** Provides popular datasets, model architectures, and common image transformations for computer vision.
- **`timm` (PyTorch Image Models):** A state-of-the-art library for accessing pre-trained computer vision models.
- **Pillow (PIL):** Used for loading and manipulating images during inference.
- **NumPy & Matplotlib:** For data handling and visualization.

---

## ⚙️ Setup and Installation

To get this project running locally, it is recommended to use a virtual environment.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/transformer_portfolio.git](https://github.com/YourUsername/transformer_portfolio.git)
    cd transformer_portfolio
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate it (Windows)
    .\venv\Scripts\activate

    # Activate it (macOS/Linux)
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install all required libraries using the `requirements.txt` file.
    ```bash
    pip install -r vision_encoder/requirements.txt
    ```

---

## Workflow & Usage

The project is divided into two main parts: training and inference.

### Step 1: Training the Model

The entire training process is contained within the Jupyter Notebook.

1.  **Navigate to the notebooks directory** and open `vit_finetuning.ipynb`.
2.  **Run all cells** in the notebook. This will:
    - Download the CIFAR-10 dataset to a `./data` folder.
    - Load the pre-trained ViT model.
    - Fine-tune the model for 3 epochs.
3.  **Output:** Upon completion, the trained model weights will be saved to **`vision_encoder/models/vit_cifar10_finetuned.pth`**.

### Step 2: Running Inference

Once the model is trained, you can use the `inference.py` script to classify any image.

1.  Place an image you want to test in the `vision_encoder/test_images/` folder.
2.  From the **root `transformer_portfolio` directory**, run the script with the `--image` argument:
    ```bash
    python vision_encoder/inference.py --image vision_encoder/test_images/your_image.jpg
    ```
    The script will then print the predicted class and the model's confidence level.

---

## 🧠 Architecture Overview: Vision Transformer (ViT)

The Vision Transformer (ViT) adapts the hugely successful Transformer architecture (originally designed for natural language processing) to computer vision tasks. Instead of processing words, it processes image patches.

### Conceptual Diagram Placeholder

*(Here, you would ideally embed an image, either created by you or found online with proper attribution, illustrating the ViT architecture. A good diagram would look something like this:)*

```
[Input Image (e.g., 224x224)]
        |
        V
[Divide into Patches (e.g., 16x16)] -- These are treated like "words"
        |
        V
[Linear Projection (Embeddings)] -- Each patch is flattened & projected to a vector
        |
        V
[Add Positional Embeddings] -- To retain spatial info, just like words
        |
        V
[Add CLS Token] ------------------- A special learnable token prepended to the sequence
        |                               (This token will later represent the whole image)
        V
[SEQUENCE OF EMBEDDINGS (CLS + Patches)]
        |
        V
[N x Transformer Encoder Blocks] -- The core of the model
    (Each Block Contains: Multi-Head Attention, LayerNorm, Feed-Forward Network, LayerNorm)
        |
        V
[Output from Transformer Encoder] -- A sequence of processed embeddings
        |
        V
[Extract CLS Token Embedding] ------ The CLS token's output is taken for classification
        |
        V
[Classification Head (Linear Layer)] -- Maps CLS embedding to output classes (10 for CIFAR-10)
        |
        V
[Predicted Class Probabilities]
```

### Explanation of Components:

1.  **Image Patching:**
    -   Unlike Convolutional Neural Networks (CNNs) that process images pixel by pixel or small regions, ViT first **divides the input image into fixed-size patches** (e.g., 16x16 pixels).
    -   These patches are flattened and treated as a sequence of "tokens" or "words."

2.  **Linear Embedding:**
    -   Each flattened patch is then **linearly projected** into a higher-dimensional space to obtain patch embeddings. This converts the pixel values into a vector representation that the Transformer can understand.

3.  **Positional Embeddings:**
    -   Since Transformers inherently lack information about the *order* or *position* of tokens, **positional embeddings** are added to the patch embeddings. These are learnable vectors that encode the spatial coordinates of each patch, allowing the model to understand where each patch came from in the original image.

4.  **\[CLS] Token:**
    -   A special, learnable **classification token (\[CLS] token)** is prepended to the sequence of patch embeddings.
    -   After passing through the Transformer Encoder, the output embedding corresponding to this \[CLS] token is used to represent the entire image for classification. It effectively "summarizes" the image's content.

5.  **Transformer Encoder Blocks:**
    -   The core of the ViT consists of multiple identical **Transformer Encoder Blocks**.
    -   Each block is composed of:
        -   **Multi-Head Self-Attention (MHSA):** This mechanism allows the model to weigh the importance of different image patches to each other. It effectively captures global dependencies between distant parts of the image. "Multi-head" means it performs this attention operation multiple times in parallel, capturing different kinds of relationships.
        -   **Layer Normalization:** Applied before and after the attention and feed-forward layers to stabilize training.
        -   **Feed-Forward Network (FFN):** A simple neural network applied independently to each token embedding, allowing the model to process information further.
    -   These blocks process the sequence of patch embeddings, allowing information to flow and interact across the entire image.

6.  **Classification Head:**
    -   Finally, the output embedding of the **\[CLS] token** from the last Transformer Encoder block is passed through a simple **linear layer (classifier head)**.
    -   This layer maps the learned image representation to the desired output classes (in our case, 10 for CIFAR-10), producing logits that can be converted into class probabilities.

### Fine-Tuning Process:

-   **Base Model:** We specifically use the `vit_base_patch16_224` model, which was pre-trained on the massive ImageNet-1k dataset (1000 classes). This provides a strong foundation of visual features.
-   **Transfer Learning:** Instead of training from scratch (which would require immense data and computational power), we leverage this pre-trained knowledge.
-   **Head Replacement:** The only architectural change we make is replacing the original 1000-class classification head with a new `nn.Linear` layer that has 10 output neurons, matching the CIFAR-10 dataset's classes. The weights of the pre-trained ViT encoder itself are retained and then fine-tuned slightly during training.
-   **Dataset:** CIFAR-10, consisting of 60,000 32x32 colour images in 10 classes, with 6,000 images per class. We resize them to 224x224 to match the ViT's input expectation.

---

## 📈 Results

After 3 epochs of training, the model achieves a high level of accuracy on the CIFAR-10 test set. For detailed loss and accuracy metrics for each epoch, please refer to the output cells of the `vit_finetuning.ipynb` notebook.

*(After you run the notebook again, you can update this section with your exact accuracy score!)*