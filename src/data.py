import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=256):

    # Define data augmentation + normalization pipeline
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),          # Randomly crop with padding for robustness
        transforms.RandomHorizontalFlip(),              # Flip images horizontally
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),     # Random brightness/contrast changes
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize pixel values
    ])

    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        './data', train=True, download=True, transform=transform
    )

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=transform
    )

    # Create DataLoader for training (with shuffling)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Create DataLoader for testing (no shuffle)
    testloader = DataLoader(testset, batch_size=batch_size)

    # Return both loaders
    return trainloader, testloader