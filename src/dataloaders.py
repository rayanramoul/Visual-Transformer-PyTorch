from torch.utils.data import DataLoader

from torchvision.transforms  import ToTensor
from torchvision.datasets import MNIST

def read_dataset():
    transform = ToTensor() 
    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    return train_loader, test_loader