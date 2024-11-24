import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import SimpleCNN
import glob
import random

def get_latest_model():
    model_files = glob.glob('model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def predict_digit(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

def display_prediction(image, prediction, actual=None):
    plt.imshow(image.squeeze(), cmap='gray')
    title = f'Prediction: {prediction}'
    if actual is not None:
        title += f' (Actual: {actual})'
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Load the model
    model = SimpleCNN()
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Load MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Get 2 random images
    indices = random.sample(range(len(test_dataset)), 5)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices, 1):
        image, label = test_dataset[idx]
        prediction = predict_digit(model, image.unsqueeze(0))
        
        plt.subplot(1, 5, i)
        display_prediction(image, prediction, label)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 