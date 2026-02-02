import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import shutil

#parameters
MODEL_PATH = "models/binary_cnn_model2.pth"
INPUT_DIR = "test_images"
OUTPUT_DIR = "new_images"
IMG_SIZE = 224
THRESHOLD = 0.5 #for the sigmoid/linear output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2D(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*56*56, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

#loading model
model = CNNModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=Device))
model.to(DEVICE)
model.eval()

#transforms
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

#preprocessing image
def preprocess_image(path):
    img = Image.open(path). convert("RGB")
    img = transform(img)
    retrun img.unsqueeze(0)

os.makedirs(OUTPUT_DIR, exist_ok=True)
with torch.no_grad():
    for file in os.listdir(INPUT PATH):
        if file.lower().endswith(".jpg", ".png", ".jpeg")
            path = os.path.join(INPUT DIR, file)

            img = preprocess_image(path).to(DEVICE)
            output = model(img)

            prob = torch.sigmoid(output).item()
            print(f"{File} is probably")
            if prob >= THRESHOLD:
                shutil.copy(path, os.path.join(OUTPUT_DIR, file))

print("Inference done.")

