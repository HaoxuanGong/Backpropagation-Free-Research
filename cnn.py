import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

# Define a simple VGG-like CNN with label encoding
class VGGLikeCNN(nn.Module):
    def __init__(self, num_classes=10, use_learned_matrix=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_learned_matrix = use_learned_matrix
        self.dims = 32  # For CIFAR-10, images are 32x32
        
        # Define the learned or random matrix
        if use_learned_matrix:
            self.embedding = nn.Embedding(num_classes, self.dims * self.dims)
        else:
            self.embedding = nn.Parameter(torch.randn(num_classes, self.dims, self.dims))
        
        # Define VGG-like layers
        self.features = nn.Sequential(
            self._make_layer(3, 64, 2),
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 3),
            self._make_layer(256, 512, 3)
        )
        
        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x, y):
        # Encode labels and concatenate with image
        if self.use_learned_matrix:
            embedding = self.embedding(y).view(-1, 1, self.dims, self.dims)
        else:
            embedding = self.embedding[y]
        
        # Print shapes for debugging
        print(f"x shape: {x.shape}")
        print(f"embedding shape: {embedding.shape}")
        
        # Ensure correct dimensions before concatenation
        if x.size(1) != embedding.size(1):
            raise ValueError("Channel dimensions do not match for concatenation.")
        
        x = torch.cat([x, embedding], dim=1)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x




# Data loading function
def load_CIFAR10_data(train_batch_size=64, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
    ])

    train_loader = DataLoader(
        CIFAR10('./data', train=True, download=True, transform=transform),
        batch_size=train_batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        CIFAR10('./data', train=False, download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader

# Main function to run the training
def main():
    train_loader, test_loader = load_CIFAR10_data()
    model = VGGLikeCNN(num_classes=10, use_learned_matrix=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):  # Number of epochs
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x, y)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Evaluate model on test set
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                output = model(x, y)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        print(f"Epoch {epoch+1}: Accuracy {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
