import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import SimpleCNN
import pytest
import glob

def get_latest_model():
    model_files = glob.glob('model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def test_model_architecture():
    model = SimpleCNN()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters")
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_accuracy():
    # Set device to CPU only
    device = torch.device("cpu")
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Load model
    model = SimpleCNN().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%"

if __name__ == "__main__":
    pytest.main([__file__]) 