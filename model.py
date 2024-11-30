#model.py
import torch
import torch.nn as nn
import torchvision.models as models

class RecycleClassifier(nn.Module):
    def __init__(self):
        super(RecycleClassifier, self).__init__()
        self.features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features.fc = nn.Linear(self.features.fc.in_features, 2)  # Binary classification

    def forward(self, x):
        x = self.features(x)
        return x

def load_model(path):
    model = RecycleClassifier()
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model
