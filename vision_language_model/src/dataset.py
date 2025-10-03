# vision_language_model/src/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import re
import random

class SimpleTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        if '<image>' not in self.vocab: self.vocab['<image>'] = len(self.vocab)
        if '<start>' not in self.vocab: self.vocab['<start>'] = len(self.vocab)
        if '<end>' not in self.vocab: self.vocab['<end>'] = len(self.vocab)
        if '<pad>' not in self.vocab: self.vocab['<pad>'] = 0
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        text = text.replace('<image>', ' <image> ')
        tokens = re.findall(r'<image>|\b\w+\b|\S', text.lower())
        return [self.vocab.get(t, self.vocab.get('<unk>', 1)) for t in tokens]

    def ids_to_sentence(self, ids):
        words = []
        for i in ids:
            word = self.inv_vocab.get(i, '<unk>')
            if word in ['<start>', '<pad>', '<image>']: continue
            if word == '<end>': break
            words.append(word)
        return ' '.join(words)

class Flickr8kVQADataset(Dataset):
    def __init__(self, image_dir, captions_file, transform, tokenizer, max_len):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._load_data(captions_file)
        self.questions = [
            "Describe the image in detail.",
            "What is happening in this picture?",
            "Provide a detailed description of the image.",
            "What is in the image?",
            "Can you describe this image for me?"
        ]

    def _load_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                image_file, caption = parts[0], ','.join(parts[1:])
                data.append({'image': image_file, 'caption': caption.lower()})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['image'])
        image = self.transform(Image.open(image_path).convert("RGB"))
        
        question = random.choice(self.questions)
        answer = item['caption']
        
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
        prompt_tokens = self.tokenizer.tokenize(prompt)
        answer_tokens = self.tokenizer.tokenize(answer)
        
        answer_input_ids = [self.tokenizer.vocab['<start>']] + answer_tokens
        answer_target_ids = answer_tokens + [self.tokenizer.vocab['<end>']]

        question_ids = (prompt_tokens + [self.tokenizer.vocab['<pad>']] * self.max_len)[:self.max_len]
        answer_input_ids = (answer_input_ids + [self.tokenizer.vocab['<pad>']] * self.max_len)[:self.max_len]
        answer_target_ids = (answer_target_ids + [self.tokenizer.vocab['<pad>']] * self.max_len)[:self.max_len]
        
        return image, torch.tensor(question_ids), torch.tensor(answer_input_ids), torch.tensor(answer_target_ids)