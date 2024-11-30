import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import RecycleClassifier
from torch.utils.data import random_split

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=15, patience=3):
    model.train()
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate(model, val_loader, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss}')

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == "__main__":
    train_dir = './images/train'
    
    train_loader, _ = get_data_loaders(train_dir=train_dir)  # Only get the train loader
    
    # Split the training data into train and validation sets
    dataset = train_loader.dataset  # This is the dataset object from ImageFolder
    val_size = int(0.2 * len(dataset))  # 20% for validation
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    
    # Create new loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)

    model = RecycleClassifier()
    model.apply(initialize_weights)

    class_weights = torch.tensor([1.0, 2.0])  
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=3)
    print("Training complete!")