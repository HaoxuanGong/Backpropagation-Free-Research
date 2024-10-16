import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR100 , EMNIST # emnist has 62 classes
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

# Define the number of classes and epochs globally
num_classes = 10
EPOCHS = 800  # Number of outer epochs for negative sample reshuffling
SHUFFLE = 4
NUM_CLASSES = 10


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
    
    # def compute_goodness(self, labels, hebbian_weights, output):
    #     positive_goodness =  (torch.mm(labels, hebbian_weights) * output).mean(1)
    #     negative_goodness = (torch.mm(1 - labels, hebbian_weights / 10) * output).mean(1)
    #     return positive_goodness , negative_goodness

    def train_layer(self, input, pos_labels, neg_labels , i):
        for _ in (range(EPOCHS)):
            output = self.forward(input)
            #positive_goodness , negative_goodness = self.compute_goodness(pos_labels, self.hebbian_weights, output)
            positive_goodness = self.compute_goodness(pos_labels, self.hebbian_weights, output)
            negative_goodness = self.compute_goodness(neg_labels, self.hebbian_weights, output)
            
            loss = torch.log(1 + torch.exp(torch.cat([
                -positive_goodness + 0.0,  # threshold = 0
                negative_goodness - 0.0
            ]))).mean()

            self.optimizer.zero_grad()
            self.hebbian_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.hebbian_optimizer.step()

        return self.forward(input).detach()


class HFF(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList([HLayer(layers_config[i], layers_config[i+1]).cuda() for i in range(len(layers_config) - 1)])

    def create_labels(self, label):
        batch_size = label.size(0)
        neg_label = torch.zeros(batch_size, NUM_CLASSES, device=label.device)
        random_neg_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=label.device)
        random_neg_labels = torch.where(random_neg_labels == label, (random_neg_labels + 1) % NUM_CLASSES, random_neg_labels)
        neg_label.scatter_(1, random_neg_labels.unsqueeze(1), 1.0)
        pos_label = torch.zeros(batch_size, NUM_CLASSES, device=label.device)
        pos_label.scatter_(1, label.unsqueeze(1), 1.0)
        
        return pos_label , neg_label

    def train_network(self, train_loader):
        for epoch in range(SHUFFLE):  # Training epochs
            for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
                positive_labels , negative_lables = self.create_labels(batch_labels.cuda())
                goodness_pos = batch_data.cuda()
                for i, layer in enumerate(self.layers):
                    goodness_pos = layer.train_layer(goodness_pos, positive_labels, negative_lables , i) 


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

# Data loading function for MNIST
def load_MNIST_data(train_batch_size=2048, test_batch_size=512):
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

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

# Prepare the data
def prepare_data():
    torch.manual_seed(4321)
    training_data_loader, testing_data_loader = load_MNIST_data()
    return training_data_loader, testing_data_loader

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.manual_seed(1234)
    training_data_loader, testing_data_loader = prepare_data()
    network = HFF([784, 2048, 1024]).cuda()
    network.train_network(training_data_loader)
    network.test_network(testing_data_loader)
