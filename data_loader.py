# data_loader.py
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, test_dir=None, batch_size=32):
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(10),
               transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = None
    test_loader = None

    if train_dir:
        train_data = datasets.ImageFolder(root=train_dir, transform=transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    if test_dir:
        test_data = datasets.ImageFolder(root=test_dir, transform=transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
