# model.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1_layer = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2_layer = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1_layer = nn.Linear(32 * 7 * 7, 128)
        self.fc2_layer = nn.Linear(128, 10)

    def forward(self, inputs):
        x = F.relu(self.conv1_layer(inputs))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2_layer(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1_layer(x))
        x = self.fc2_layer(x)
        return x
    
    