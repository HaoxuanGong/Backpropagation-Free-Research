import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import time

# Define the number of classes and epochs globally
num_classes = 10
epochs = 50

class HLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.hebbian_weights = nn.Parameter(torch.ones(num_classes, out_features).cuda())  # num_classes x number of nuerons
        self.activation = nn.ReLU()
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.hebbian_optimizer = Adam([self.hebbian_weights], lr=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        output = torch.mm(normalized_input, self.weight.T) + self.bias.unsqueeze(0)
        return self.activation(output)

    def compute_goodness(self, label, hebbian_weight, output):
        return (torch.mm(label, hebbian_weight) * output).mean(1)

    def train_layer(self, positive_input, negative_input, pos_labels, neg_labels):
        for _ in tqdm(range(epochs)):
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

    def create_data(self, data, label, seed=None):
        if seed is not None:
            random.seed(seed)

        positive_data = data.clone()
        positive_data[:, :num_classes] = 0.0
        for i in range(positive_data.shape[0]):
            positive_data[i][label[i]] = 1.0

        negative_data = data.clone()
        negative_data[:, :num_classes] = 0.0
        for i in range(negative_data.shape[0]):
            possible_answers = list(range(num_classes))
            possible_answers.remove(label[i])
            false_label = random.choice(possible_answers)
            negative_data[i][false_label] = 1.0

        return positive_data , negative_data

    def train_network(self, training_data, training_data_label):
        for epoch in range(1):
            print(f'Epoch {epoch + 1}')
            goodness_pos, goodness_neg = self.create_data(training_data, training_data_label) #  postivtve and negative generation
            positive_labels = nn.Parameter(goodness_pos[:, :num_classes].cuda())
            negative_labels = nn.Parameter(goodness_neg[:, :num_classes].cuda())

            for i, layer in enumerate(self.layers):
                print(f'Training Layer {i}...')
                goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, positive_labels, negative_labels)

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
    
    # def predict(self, input_data): # multiple pass version
    #     goodness_per_label = []
    #     for label in range(num_classes): 
    #         marked_data = self.mark_data(input_data, label)
    #         goodness_values = []
    #         for layer in self.layers:
    #             marked_data = layer(marked_data)
    #             hebbian_weight = layer.hebbian_weights[label, :]
    #             goodness_value = (marked_data * hebbian_weight).mean(1)
    #             goodness_values.append(goodness_value)
    #         goodness_per_label.append(torch.sum(torch.stack(goodness_values), dim=0).unsqueeze(1))
    #     goodness_per_label = torch.cat(goodness_per_label, 1)
    #     return goodness_per_label.argmax(dim=1)


    def mark_data(self, input_data, label):
        marked_data = input_data.clone()
        marked_data[:, :num_classes] = 0.0
        marked_data[:, label] = 1
        return marked_data


# Data loading function for MNIST
def load_MNIST_data(train_batch_size=50000, test_batch_size=10000):
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
    network = HFF([784, 1000, 1000]).cuda()  # Use num_classes
    network.train_network(training_data, training_data_label) # this now only take in traing data and label, postive and neg is generated within network
    print("Training Accuracy: ", network.predict(training_data).eq(training_data_label).float().mean().item())
    print("Testing Accuracy: ", network.predict(testing_data).eq(testing_data_label).float().mean().item())

