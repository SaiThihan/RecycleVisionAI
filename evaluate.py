# evaluate.py
import torch
from data_loader import get_data_loaders
from model import load_model

def evaluate(model, test_loader, criterion):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

if __name__ == "__main__":
    test_dir = './images/test'
    model_path = 'best_model.pth'
    model = load_model(model_path)
    _, test_loader = get_data_loaders(None, test_dir)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate(model, test_loader, criterion)
