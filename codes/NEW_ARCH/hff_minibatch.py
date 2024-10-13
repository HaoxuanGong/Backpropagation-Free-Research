import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import numpy as np

# Define the number of classes and epochs globally
num_classes = 10
epochs_layer = 200  # Number of outer epochs for negative sample reshuffling

class HLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.hebbian_weights = nn.Parameter(torch.ones(num_classes, out_features).cuda())  # num_classes x number of neurons
        self.activation = nn.ReLU()
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.hebbian_optimizer = Adam([self.hebbian_weights], lr=0.015)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        output = torch.mm(normalized_input, self.weight.T) + self.bias.unsqueeze(0)
        return self.activation(output)

    def compute_goodness(self, label, hebbian_weight, output):
        return (torch.mm(label, hebbian_weight) * output).mean(1)

    def train_layer(self, positive_input, negative_input, pos_labels, neg_labels , i):
        for epoch in (range(epochs_layer)): # epoch for porcessing input
            positive_output = self.forward(positive_input)  # Forward pass
            negative_output = self.forward(negative_input)
            positive_goodness = self.compute_goodness(pos_labels, self.hebbian_weights, positive_output)
            negative_goodness = self.compute_goodness(neg_labels, self.hebbian_weights, negative_output)
            
            loss = torch.log(1 + torch.exp(torch.cat([
                -positive_goodness + 0.0,  # threshold = 0
                negative_goodness - 0.0
            ]))).mean()

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
        negative_data[:, :num_classes] = 0.0
        for i in range(negative_data.shape[0]):
            possible_answers = list(range(num_classes))
            possible_answers.remove(label[i])
            false_label = random.choice(possible_answers)
            negative_data[i][false_label] = negative_data.max()
        return negative_data
    
    def create_pos_data(self, data, label, seed=None):
        if seed is not None:
            random.seed(seed)

        positive_data = data.clone()
        positive_data[:, :num_classes] = 0.0
        for i in range(positive_data.shape[0]):
            positive_data[i][label[i]] = positive_data.max()
        return positive_data

    def train_network(self, training_data_loader, neg_shuffle):
        # Outer loop for reshuffling negative samples
        for epoch in range(neg_shuffle):
            for batch_idx, (training_data, training_data_label) in enumerate(tqdm(training_data_loader, desc=f'Training Network Epoch {epoch + 1}')):
                # the 50000 batch size is now slice down to mini batches
                training_data, training_data_label = training_data.cuda(), training_data_label.cuda()
                goodness_pos = self.create_pos_data(training_data, training_data_label) # [512 ]
                goodness_neg = self.create_neg_data(training_data, training_data_label)
                positive_labels = nn.Parameter(goodness_pos[:, :num_classes].cuda())
                negative_labels = nn.Parameter(goodness_neg[:, :num_classes].cuda())

                # Mini-batch training loop
                for i, layer in enumerate(self.layers):
                    goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, positive_labels, negative_labels, i)

    def test_network(self, testing_data_loader):
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for testing_data, testing_data_label in testing_data_loader:
                testing_data, testing_data_label = testing_data.cuda(), testing_data_label.cuda()
                
                predictions = self.predict(testing_data)
                total_correct += predictions.eq(testing_data_label).sum().item()
                total_samples += testing_data_label.size(0)

        accuracy = total_correct / total_samples
        print(f"Testing Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def predict(self, input_data): # one pass version like mentioned in our paper
        goodness_per_label = []
        for layer in self.layers:
            input_data = layer(input_data)  # process the data once for all labels
            hebbian_weights = layer.hebbian_weights  # [num_classes, num_neurons]
            # compute goodness for all labels at once
            goodness_value = torch.mm(input_data, hebbian_weights.T)  # Gij = Ain x Hijn
            goodness_per_label.append(goodness_value)
        total_goodness = torch.stack(goodness_per_label, dim=0).sum(dim=0)  # Gj = sum(Gij)
        return total_goodness.argmax(dim=1)

    def mark_data(self, input_data, label):
        marked_data = input_data.clone()
        marked_data[:, :num_classes] = 0.0
        marked_data[:, label] = 1
        return marked_data

# Data loading function for MNIST
def load_MNIST_data(train_batch_size=5000, test_batch_size=512):
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
    return training_data_loader, testing_data_loader

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.manual_seed(1234)
    training_data_loader, testing_data_loader = prepare_data()
    network = HFF([784, 800, 800]).cuda()
    network.train_network(training_data_loader, neg_shuffle=10)
    network.test_network(testing_data_loader)
