import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    """
    transform = Compose([...]) defines a set of operations that later will be used to transform the data

    ToTensor() transforms a PIL image or a Numpy ndarray to a PyTorch tensor, converting data to between 0 - 1
        Tensor is a flexible, multidimensional array for representing data

    Normalize() takes a mean of 0.1307 and a standard deviation of 0.3081. This allows the data to center
    more around 0.

    Lambda() takes a function as input, which flattens the data to a 1-dimensional array

    """

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    """
    DataLoader prepares data from MNIST database
    batch_size is the number of samples that will be loaded and passed through a network at one time
    """

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """
    Replace the first 10 pixels of data [x] with one-hot-encoded label [y] in each row
    x.shape[0] returns the number of rows in the x tensor
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):
    """
    dims refer to the number of neurons in each layer, and therefore the architecture of the network
    Layer is defined by two consecutive numbers in the dims list
    """

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

    """
    Evaluate which label (out of 10) results in the highest "goodness" score when the data is processed through 
    network with that label overlay onto the input data
    """

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            # Overlay label onto x
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                # Passes h (overlay data) to layers
                h = layer(h)
                # ISSUE
                # What does mean(1) do, and why?
                # Isn't the goodness calculated as the direct sum of the squared activities?
                # Isn't the neural activities from the first layer excluded as it might get too abstract to contribute
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):

    """
    Layer Setup
    """
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    """
    x.norm(2, 1, keepdim=True)
    2: Taking the L2 norm
    1: A dimension of 1 means the norm will be computed for each row
    keepdim=true: preserving the original dimension
    
    Remove magnitude information, maintaining the direction information
    
    torch.mm() performs matrix multiplication between weights and normalized input to produce
    output with bias added
    """
    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    """
    tqdm is a library that provides progress bars, giving you a visual indication of how many epochs are completed
    
    """
    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.

            # -g.pos + self.threshold: small loss value when g_pos is greater than threshold
            # g_neg - self.threshold: small loss value when g.neg is smaller than threshold
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            # ISSUE
            # HOW ?
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


if __name__ == "__main__":
    # This sets the seed for generating random numbers. It ensures that the results are reproducible
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net([784, 500, 500, 500, 500])
    # gets the first batch of images (x) and their labels (y) from the training data loader
    x, y = next(iter(train_loader))
    # moves the images and labels to GPU to enable faster computation
    x, y = x.cuda(), y.cuda()
    x_pos = overlay_y_on_x(x, y)
    # generates a random permutation of indices for x to shuffle the labels, which will be used to generate negative
    # data
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)

    net.train(x_pos, x_neg)

    print('training accuracy:', net.predict(x).eq(y).float().mean().item())

    # Evaluation
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('testing accuracy:', net.predict(x_te).eq(y_te).float().mean().item())
