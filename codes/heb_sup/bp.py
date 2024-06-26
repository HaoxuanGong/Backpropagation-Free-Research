import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

def load_MNIST_data(train_batch_size=64, test_batch_size=64):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    training_data_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=data_transformation),
        batch_size=train_batch_size,
        shuffle=True
    )

    testing_data_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=data_transformation),
        batch_size=test_batch_size,
        shuffle=False
    )

    return training_data_loader, testing_data_loader

class SimpleNN(nn.Module):
    def __init__(self, layer_sizes, bias=True, device=None, d_type=None, is_hinge_loss=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias).to(device=device, dtype=d_type))
        self.activation = torch.nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.num_of_epochs = 2
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def train(self, train_loader):
        for epoch in range(self.num_of_epochs):
            for i, (images, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if (i+1) % 300 == 0:
                    print(f"Epoch {epoch+1}/{self.num_of_epochs}, Step {i+1}/{len(train_loader)}, Loss: {loss.item()}")

    def predict(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy




# Load MNIST data
train_loader, test_loader = load_MNIST_data()

# Define the model
model = SimpleNN([784, 500, 500])
model.train(train_loader)

# Evaluate the model
accuracy = model.predict(test_loader)
print(f"Test Accuracy: {accuracy}")
