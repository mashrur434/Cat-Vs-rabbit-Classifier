from flask import Flask, request,render_template, jsonify

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io




app = Flask(__name__)

# Define the model class (exact copy from the notebook)
class BinaryCNN_RGB(nn.Module):
    def __init__(self):
        super(BinaryCNN_RGB, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  # 14x14
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*14*14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 1)   # raw logits output
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BinaryCNN_RGB().to(device)
model.load_state_dict(torch.load('cat-vs-rabbit.pth', weights_only=True, map_location=device))
model.eval()

# Define the transform (from the notebook's predict_image function)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classes
classes = ['cat', 'rabbit']

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logit = model(img_tensor).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()  # Probability
            pred_class_idx = 1 if prob > 0.5 else 0
            pred_class = classes[pred_class_idx]
        
        return jsonify({
            'class': pred_class,
            'probability': prob,
            'logit': logit
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)