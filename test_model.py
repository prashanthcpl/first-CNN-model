import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import SimpleCNN
import pytest
import glob
import numpy as np

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

def test_model_robustness_to_noise():
    """Test model's robustness to input noise"""
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Load a single test image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    image, label = test_dataset[0]
    
    # Add Gaussian noise
    noise_level = 0.1
    noisy_image = image + torch.randn_like(image) * noise_level
    
    with torch.no_grad():
        original_pred = model(image.unsqueeze(0))
        noisy_pred = model(noisy_image.unsqueeze(0))
        
        _, original_class = torch.max(original_pred, 1)
        _, noisy_class = torch.max(noisy_pred, 1)
    
    assert original_class == noisy_class, "Model prediction changed significantly with noise"

def test_output_probabilities():
    """Test if model outputs valid probability distributions"""
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Test with random input
    test_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = torch.softmax(model(test_input), dim=1)
        
        # Check if probabilities sum to 1
        prob_sum = output.sum().item()
        assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities sum to {prob_sum}, should be 1.0"
        
        # Check if all probabilities are between 0 and 1
        assert torch.all((output >= 0) & (output <= 1)), "Some probabilities are outside [0,1]"

def test_batch_consistency():
    """Test if model predictions are consistent across different batch sizes"""
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Create test input
    test_input = torch.randn(10, 1, 28, 28)
    
    with torch.no_grad():
        # Process all at once
        batch_output = model(test_input)
        _, batch_preds = torch.max(batch_output, 1)
        
        # Process one by one
        individual_preds = []
        for i in range(10):
            output = model(test_input[i:i+1])
            _, pred = torch.max(output, 1)
            individual_preds.append(pred.item())
    
    individual_preds = torch.tensor(individual_preds)
    assert torch.all(batch_preds == individual_preds), "Predictions vary with batch size"

if __name__ == "__main__":
    pytest.main([__file__]) 