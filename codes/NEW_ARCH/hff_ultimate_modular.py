import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#from load_data import prepare_data

# Define the number of classes and epochs globally
NUM_CLASSES = 10
EPOCHS = 300
TRAIN_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 1000
SHUFFLE = 1

class HLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.hebbian_weights = nn.Parameter(torch.ones(NUM_CLASSES, out_features).cuda())  # num_classes x number of neurons
        self.activation = nn.ReLU()
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.hebbian_optimizer = RMSprop([self.hebbian_weights], lr=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        output = torch.mm(normalized_input, self.weight.T) + self.bias.unsqueeze(0)
        return self.activation(output)

    def compute_goodness(self, label, hebbian_weight, output):
        return (torch.mm(label, hebbian_weight) * output).mean(1)
    
    def balanced_loss(self, positive_goodness, negative_goodness, alpha=4.0):
        delta = positive_goodness - negative_goodness
        return torch.log(1 + torch.exp(-alpha * delta)).mean()
    
    def soft_plus(self, positive_goodness, negative_goodness):
        return torch.log(1 + torch.exp(torch.cat([-positive_goodness + 2.0, negative_goodness - 2.0]))).mean()


    def train_layer(self, positive_input, negative_input, pos_labels, neg_labels, i):
        for _ in tqdm(range(EPOCHS), desc=f'Training Layer {i}'):
            # Forward Pass -----------------------------------------------------
            positive_output = self.forward(positive_input)  # Forward pass
            negative_output = self.forward(negative_input)
            positive_goodness = self.compute_goodness(pos_labels, self.hebbian_weights, positive_output)
            negative_goodness = self.compute_goodness(neg_labels, self.hebbian_weights, negative_output)
            # ---------------------------------------------------------------------   
            #loss = self.balanced_loss(positive_goodness, negative_goodness)
            loss = self.soft_plus(positive_goodness, negative_goodness)
            self.optimizer.zero_grad()
            self.hebbian_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.hebbian_optimizer.step()

        return self.forward(positive_input).detach(), self.forward(negative_input).detach()

class HFF(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList([HLayer(layers_config[i], layers_config[i+1]).cuda() for i in range(len(layers_config) - 1)])
    
    def create_neg_data(self, data, label, seed=None):
        if seed is not None:
            random.seed(seed)            
        negative_data = data.clone()
        neg_label = torch.zeros(data.size(0), NUM_CLASSES)
        for i in range(negative_data.shape[0]):
            possible_answers = list(range(NUM_CLASSES))
            possible_answers.remove(label[i].item())
            false_label = random.choice(possible_answers)
            #negative_data[i, false_label] = negative_data.max() # commonet for no onehot encoding at all
            neg_label[i, false_label] = 1.0
        return negative_data.cuda(), neg_label.cuda()
    
    def create_pos_data(self, data, label):
        positive_data = data.clone()
        pos_label = torch.zeros(data.size(0), NUM_CLASSES)
        for i in range(positive_data.shape[0]):
            #positive_data[i][label[i]] = positive_data.max() # commonet for no onehot encoding at all
            pos_label[i][label[i]] = 1.0
        return positive_data.cuda() , pos_label.cuda()

    def train_network(self, training_data, training_data_label):
        pos_data, pos = self.create_pos_data(training_data, training_data_label)  # Positive data
        neg_data, neg = self.create_neg_data(training_data, training_data_label)  # Negative data
        positive_labels = nn.Parameter(pos)
        negative_labels = nn.Parameter(neg)
        for epoch in range(SHUFFLE):  # Training epochs
            print(f'Epoch {epoch + 1}')
            goodness_pos = pos_data
            goodness_neg = neg_data
            for i, layer in enumerate(self.layers):
                goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, positive_labels, negative_labels, i)

    def predict(self, input_data):
        goodness_per_label = []
        for layer in self.layers:
            input_data = layer(input_data)  # process the data once for all labels
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
    network = HFF([784, 500, 500]).cuda()  # Use num_classes
    network.train_network(training_data, training_data_label)  # Train the network
    print("Training Accuracy: ", network.predict(training_data).eq(training_data_label).float().mean().item())
    print("Testing Accuracy: ", network.predict(testing_data).eq(testing_data_label).float().mean().item())
