from PIL import Image
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(file):
    image = Image.open(file).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return image, tensor
