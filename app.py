import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr


class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# 2. Setup environment, load classes maps and weights
with open("classes.json", "r") as f:
    classes = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyCNN(num_classes=len(classes))
model.load_state_dict(torch.load("plantvillage_cnn.pth", map_location=device))
model.to(device)
model.eval()

# 3. Match the transformations you used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 4. Define the inference function for Gradio
def predict_image(inp_img):
    if inp_img is None:
        return "Please upload an image."
    
    # Convert incoming numpy array or path into a PIL Image and transform
    image = Image.fromarray(inp_img.astype('uint8'), 'RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Get all class probabilities for a beautiful bar chart output
    confidences = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    return confidences

# 5. Launch the Gradio App interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="PlantVillage Disease Classifier",
    description="Upload a photo of a plant leaf to identify the species and check for signs of disease."
)

if __name__ == "__main__":
    interface.launch()