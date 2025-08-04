# Multi-Modal-Hate-Speech-Detection-in-Memes
BERT-ResNet fusion model with cross-attention for content moderation

# Multimodal Meme Harmfulness Classification

Detects harmful memes using both visual and textual content with BERT and ResNet.

## Overview
- **Problem**: Binary classification of memes as harmful/non-harmful
- **Model**: Fusion of BERT (text) and ResNet50 (image) features
- **Dataset**: 9,000 labeled memes (8500 train, 500 validation)
- **Key Techniques**: Class imbalance handling, multimodal fusion

## Results
| Metric     | Score |
|------------|-------|
| Accuracy   | 0.59  |
| F1-Score   | 0.59  |
| Precision  | 0.59  |
| Recall     | 0.59  |

![Confusion Matrix](results/confusion_matrix.png)

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt

Dependencies
Python 3.8+

PyTorch 1.12+

Transformers 4.18+

text

2. **requirements.txt**:
torch==2.0.1
torchvision==0.15.2
transformers==4.30.2
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.7.1
Pillow==9.5.0
tqdm==4.65.0

text

3. **src/model.py** (Core component):
```python
import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as models

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.img_proj = nn.Linear(2048, 768)
        self.text_proj = nn.Linear(768, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768*2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_out.last_hidden_state[:, 0, :]
        img_features = self.resnet(image).flatten(1)
        img_features = self.img_proj(img_features)
        combined = torch.cat([text_features, img_features], dim=1)
        return self.classifier(combined)

def predict_meme(image_path, text):
    """Predict harmfulness of a meme"""
    # Preprocessing and prediction logic
    ...
