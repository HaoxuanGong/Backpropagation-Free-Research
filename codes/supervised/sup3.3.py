# Import necessary libraries
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm  # For displaying progress bars during loops
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

# Function to load MNIST training and testing data with specific transformations
def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    # Define transformations: convert to tensor, normalize, and flatten the images
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    # Load training data
    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    # Load testing data
    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

# Function to modify input data by overlaying label information onto the image data
def overlay_y_on_x(x, y):
    """Overlay the label `y` onto the image data `x` by setting the first 10 pixels
    to zero and then encoding the label as a max value at the label's index."""
    x_ = x.clone()
    x_[:, :10] *= 0.0  # Zero out the first 10 pixels
    x_[range(x.shape[0]), y] = x.max()  # Set the pixel at the label's index to the max value in `x`
    return x_

# Define the neural network model
class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        # Dynamically create layers based on the `dims` list
        self.layers = [Layer(dims[d], dims[d + 1]).cuda() for d in range(len(dims) - 1)]

    # Function to predict labels for input data `x`
    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    # Function to train the model with positive and negative examples
    def train(self, x_pos, x_neg):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)

# Custom layer class inheriting from `nn.Linear`
class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    # Custom forward pass applying ReLU activation
    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    # Custom training routine for this layer
    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # Loss function designed to push positive samples above and negative samples below a threshold
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

# Function to visualize a sample from the dataset
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)  # Reshape flattened image back to 2D
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

# Main script to run the training and evaluation
if __name__ == "__main__":
    torch.manual_seed(1234)  # Set seed for reproducibility
    train_loader, test_loader = MNIST_loaders()  # Load data

    # Initialize the network and prepare data
    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()  # Move data to GPU
    x_pos = overlay_y_on_x(x, y)  # Prepare positive samples
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])  # Prepare negative samples using random labels
    
    # Visualize samples before training
    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)
    
    net.train(x_pos, x_neg)  # Train the model

    # Evaluate the model on training data
    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    # Evaluate the model on test data
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()
    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
