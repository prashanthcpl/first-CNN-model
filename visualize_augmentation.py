import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

def get_transforms():
    # Original transform (no augmentation)
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Augmented transform
    augmented_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            scale=(0.9, 1.1),
            translate=(0.1, 0.1)
        ),
        transforms.ToTensor(),
    ])
    
    return basic_transform, augmented_transform

def show_augmented_samples(n_samples=5):
    # Get transforms
    basic_transform, augmented_transform = get_transforms()
    
    # Load dataset
    dataset = datasets.MNIST('./data', train=True, download=True, transform=None)
    
    # Create figure
    plt.figure(figsize=(2*n_samples, 4))
    
    # Get random indices
    indices = random.sample(range(len(dataset)), n_samples)
    
    for idx, sample_idx in enumerate(indices):
        # Get original image
        image, label = dataset[sample_idx]
        
        # Apply transforms
        original = basic_transform(image)
        augmented = augmented_transform(image)
        
        # Plot original
        plt.subplot(2, n_samples, idx + 1)
        plt.imshow(original.squeeze(), cmap='gray')
        plt.title(f'Original: {label}')
        plt.axis('off')
        
        # Plot augmented
        plt.subplot(2, n_samples, n_samples + idx + 1)
        plt.imshow(augmented.squeeze(), cmap='gray')
        plt.title('Augmented')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_augmented_samples(5) 