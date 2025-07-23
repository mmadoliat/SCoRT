import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformation to apply to the data
data_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to range [-1, 1]
])

# Download MNIST dataset and apply the transformation
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=data_transform, download=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=data_transform, download=False)

# Define data loaders to load the data in batches during training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate model and move to device
cnn_model = SimpleCNN().to(device)

# Define loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Train model
for epoch in range(5):
    cnn_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluate on test data
cnn_model.eval()
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = cnn_model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted_labels == labels).sum().item()

accuracy = 100 * correct_predictions / total_samples
print(f"Accuracy of test set: {accuracy:.2f}%")

# Save the model
torch.save(cnn_model.state_dict(), 'cnn_model.pth')
