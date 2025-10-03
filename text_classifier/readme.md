# Transformer Text Classifier for AG News

This project is a Transformer Encoder model, built from scratch using PyTorch, to classify news headlines from the AG News dataset into one of four categories: World, Sports, Business, and Sci/Tech.

## Features
- **Transformer Architecture:** A custom-built Transformer Encoder model without relying on high-level libraries like Hugging Face.
- **Interactive Interface:** A command-line script (`inference.py`) allows for real-time classification of new headlines.
- **Modular Code:** The model architecture is cleanly separated (`src/model.py`) from the inference logic (`inference.py`) for better reusability and readability.

## Setup and Installation

To run this project, you'll need Python and PyTorch.

1.  **Prerequisites:**
    - Python 3.8+
    - PyTorch

2.  **Install PyTorch:**
    If you don't have PyTorch installed, follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

3.  **Required Files:**
    Make sure the trained model `ag_news_transformer.pth` and the vocabulary file `vocab.json` are in this directory.

## How to Use

To run the interactive classifier, navigate to the `transformer` directory in your terminal and run the following command:

```bash
python text_classifier/inference.py