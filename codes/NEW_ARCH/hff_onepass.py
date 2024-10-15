import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST , FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
#from load_data import prepare_data

# Define the number of classes and epochs globally
NUM_CLASSES = 10
EPOCHS = 1000
TRAIN_BATCH_SIZE = 50000
TEST_BATCH_SIZE = 10000
SHUFFLE = 1

class HLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.hebbian_weights = nn.Parameter(torch.ones(NUM_CLASSES, out_features).cuda())  # num_classes x number of neurons
        self.activation = nn.Tanh()
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.hebbian_optimizer = Adam([self.hebbian_weights], lr=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        output = torch.mm(normalized_input, self.weight.T) + self.bias.unsqueeze(0)
        return self.activation(output)

    def compute_goodness(self, labels, hebbian_weights, output):
        positive_goodness =  (torch.mm(labels, hebbian_weights) * output).mean(1)
        negative_goodness = (torch.mm(1 - labels, hebbian_weights) * output).mean(1) / labels.size(1) - 1
        return positive_goodness , negative_goodness

    def balanced_loss(self, positive_goodness, negative_goodness, alpha=4.0):
        delta = positive_goodness - negative_goodness
        return torch.log(1 + torch.exp(-alpha * delta)).mean()
    
    def soft_plus(self, positive_goodness, negative_goodness):
        return torch.log(1 + torch.exp(torch.cat([- positive_goodness + 0.0, negative_goodness- 0.0]))).mean()

    def train_layer(self, positive_input, pos_labels, i):
        for _ in tqdm(range(EPOCHS), desc=f'Training Layer {i}'):
            # Forward Pass -----------------------------------------------------
            positive_output = self.forward(positive_input)
            positive_goodness , negative_goodness = self.compute_goodness(pos_labels, self.hebbian_weights, positive_output)
            #print((negative_goodness).mean(0))
            #positive_goodness , negative_goodness = self.compute_goodness(neg_labels, self.hebbian_weights, negative_output)
            # ---------------------------------------------------------------------   
            #loss = self.balanced_loss(positive_goodness, negative_goodness)
            loss = self.soft_plus(positive_goodness, negative_goodness)
            self.optimizer.zero_grad()
            self.hebbian_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.hebbian_optimizer.step()

        return self.forward(positive_input).detach()

class HFF(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList([HLayer(layers_config[i], layers_config[i+1]).cuda() for i in range(len(layers_config) - 1)])
    
    def create_pos_data(self, data, label):
        positive_data = data.clone()
        pos_label = torch.zeros(data.size(0), NUM_CLASSES)
        for i in range(positive_data.shape[0]):
            pos_label[i][label[i]] = 1.0
        return pos_label.cuda()

    def train_network(self, training_data, training_data_label):
        pos = self.create_pos_data(training_data, training_data_label)  # Positive data
        positive_labels = nn.Parameter(pos)
        for epoch in range(SHUFFLE):
            print(f'Epoch {epoch + 1}')
            goodness_pos = training_data
            for i, layer in enumerate(self.layers):
                goodness_pos = layer.train_layer(goodness_pos, positive_labels, i)

    def predict(self, input_data):
        goodness_per_label = []  # Store goodness for all layers
        for layer_idx, layer in enumerate(self.layers):
            input_data = layer(input_data)  # Process data through the layer
            hebbian_weights = layer.hebbian_weights  # [num_classes, num_neurons]
            goodness_value = torch.mm(input_data, hebbian_weights.T)  # Gij = Ain x Hijn
            goodness_per_label.append(goodness_value)
        total_goodness = torch.stack(goodness_per_label, dim=0).sum(dim=0)  # Gj = sum(Gij)
        return total_goodness.argmax(dim=1)

# Data loading function for MNIST
def load_MNIST_data(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))  # Flattening the 28x28 images into 1D vectors
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


# Prepare the data
def prepare_data():
    torch.manual_seed(4321)
    training_data_loader, testing_data_loader = load_MNIST_data()
    training_data, training_data_label = next(iter(training_data_loader))
    testing_data, testing_data_label = next(iter(testing_data_loader))
    testing_data, testing_data_label = testing_data.cuda(), testing_data_label.cuda()
    training_data, training_data_label = training_data.cuda(), training_data_label.cuda()
    return training_data, training_data_label, testing_data, testing_data_label


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.manual_seed(1234)
    training_data, training_data_label, testing_data, testing_data_label = prepare_data()
    network = HFF([784, 512 , 512]).cuda()  # Use num_classes
    network.train_network(training_data, training_data_label)  # Train the network
    print("Training Accuracy: ", network.predict(training_data).eq(training_data_label).float().mean().item())
    print("Testing Accuracy: ", network.predict(testing_data).eq(testing_data_label).float().mean().item())
