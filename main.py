
import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam


number_of_epochs = 1000

hidden_layer_neurons = 500

class Layer(nn.Linear):
    hebbian_weights_layer_one = nn.Parameter(torch.ones(10, hidden_layer_neurons).cuda())
    hebbian_weights_layer_two = nn.Parameter(torch.ones(10, hidden_layer_neurons).cuda())
    def __init__(self, in_features, out_features, bias=True, device=None, d_type=None, is_hinge_loss=False):
        super().__init__(in_features, out_features, bias, device, d_type)
        self.activation = torch.nn.ReLU()
        self.learning_rate = 0.02
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self.threshold = 2.0
        self.num_of_epochs = number_of_epochs
        self.is_hinge_loss = is_hinge_loss
        self.hebbian_optimizer = Adam([Layer.hebbian_weights_layer_two, Layer.hebbian_weights_layer_one], lr=0.02)
        self.kernel_size = 3
        self.in_channels = 1  # Assuming grayscale images, hence 1 channel
        self.image_size = int(in_features ** 0.5)  # Assuming input is flattened image
        self.block_size = 10
        self.neighborhood_size = 400
         # Create a block-wise connectivity mask
        #self.mask = self.create_block_wise_mask(in_features, out_features, self.block_size).cuda()
        self.mask = self.create_neighbour_mask(in_features, out_features).cuda()
        #self.weight = nn.Parameter(torch.randn(out_features, in_features).cuda())
        #self.bias = nn.Parameter(torch.randn(out_features).cuda())
        #print(self.weight.shape)

    def compute_hebbian_activity(self, labels, values, layer_num):
        if layer_num == 0:
            hebbian_value = torch.mm(values.pow(2) * self.layer_weights[layer_num], self.hebbian_weights_layer_one.T) * labels
        else:
            hebbian_value = torch.mm(values, self.hebbian_weights_layer_two.T) * labels
        return hebbian_value.mean(1).cuda()

    def create_block_wise_mask(self, in_features, out_features, block_size):
        # weight matrix W dimensions [out_features, in_features]
        # W[i, j] is the connection(weight) between:        
        # j-th neuron in the input layer   
        # The i-th neuron in the output layer   
        # this is block wise connection, so 0-9 in input connect to 0-9 in output
        # 10-19 input to 10-19 output
        mask = torch.zeros(out_features, in_features)
        for i in range(0, in_features, block_size):
            start_index = i
            end_index = min(i + block_size, in_features)
            mask[start_index:end_index, start_index:end_index] = 1
        return mask
    
    def create_neighbour_mask(self, in_features, out_features):
        # neighbour connection, so 0 connects to 0-10, 1 connect to 1-11
        mask = torch.zeros(out_features, in_features)
        for i in range(out_features):
            # start and end indices for the neighborhood
            start_index = max(0, i - self.neighborhood_size // 2)
            end_index = min(in_features, i + self.neighborhood_size // 2 + 1)
            mask[i, start_index:end_index] = 1
        return mask

    def forward(self, input: Tensor) -> Tensor:
        normalized_input = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        limited_weight = self.weight * self.mask
        output = torch.mm(normalized_input, limited_weight.T) + self.bias.unsqueeze(0)
        return self.activation(output)

    def balanced_loss(self, positive_goodness, negative_goodness, alpha=8.0):
        delta = positive_goodness - negative_goodness
        per_instance_loss = torch.log(1 + torch.exp(-alpha * delta))
        return per_instance_loss.mean()  
    
    def apply_sparse_weights(self):
        # Apply random sparsity
        mask = torch.rand_like(self.weight) > self.sparsity
        self.register_buffer('sparsity_mask', mask)
        self.weight.data *= self.sparsity_mask

    def exponential_hinge_loss(self, positive_goodness, negative_goodness, delta=1.0, is_second_phase=False):
        if is_second_phase:
            threshold = self.threshold * 2
        else:
            threshold = self.threshold
        positive_loss = torch.exp(torch.clamp(delta - (positive_goodness - threshold), min=0)) - 1
        negative_loss = torch.exp(torch.clamp(delta - (threshold - negative_goodness), min=0)) - 1
        return torch.cat([positive_loss, negative_loss]).mean()

    def plot_goodness(self, positive_goodness, negative_goodness):
        plt.figure(figsize=(10, 5))
        plt.plot(positive_goodness, label='Positive Goodness', color='b')
        plt.plot(negative_goodness, label='Negative Goodness', color='r')
        plt.xlabel('Epoch')
        plt.ylabel('Goodness Value')
        plt.title('Change in Goodness During Training')
        plt.legend()
        plt.grid(True)
        plt.show()

    def train_layer(self, positive_input, negative_input, layer_num,gt):
        positive_goodness_history = []
        negative_goodness_history = []
        for _ in tqdm(range(self.num_of_epochs)):
            positive_output = self.forward(positive_input)  # Shape: [batch_size, 500]
            negative_output = self.forward(negative_input)  
            if layer_num == 0:
                # positive_goodness = (torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_one_zeroth) * positive_output + torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_one) * positive_output.pow(2) + torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_one_quadra) * positive_output.pow(4) + torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_one_third) * positive_output.pow(3)).mean(1)
                positive_goodness = (torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_one) * positive_output.pow(2)).mean(1)
                # negative_goodness = (torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_one_zeroth) * negative_output + torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_one) * negative_output.pow(2) + torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_one_quadra) * negative_output.pow(4) + torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_one_third) * negative_output.pow(3)).mean(1)
                negative_goodness = (torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_one) * negative_output.pow(2)).mean(1)
            else:
                # Second Layer
                # positive_goodness = (torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_two_zeroth) * positive_output + torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_two) * positive_output.pow(2) + torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_two_quadra) * positive_output.pow(4) + torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_two_third) * positive_output.pow(3)).mean(1)
                positive_goodness = (torch.mm(Network.positive_labels, Layer.hebbian_weights_layer_two) * positive_output.pow(2)).mean(1)
                # negative_goodness = (torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_two_zeroth) * positive_output + torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_two) * negative_output.pow(2) + torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_two_quadra) * negative_output.pow(4) + torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_two_third) * negative_output.pow(3)).mean(1)
                negative_goodness = (torch.mm(Network.negative_labels, Layer.hebbian_weights_layer_two) * negative_output.pow(2)).mean(1)
            
            positive_goodness_history.append(positive_goodness.mean().item())
            negative_goodness_history.append(negative_goodness.mean().item())

            if self.is_hinge_loss:
                loss = self.exponential_hinge_loss(positive_goodness, negative_goodness)
            else:
                loss = self.balanced_loss(positive_goodness, negative_goodness)

            self.optimizer.zero_grad()
            self.hebbian_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.hebbian_optimizer.step()
        
        self.plot_goodness(positive_goodness_history, negative_goodness_history)
        return self.forward(positive_input).detach(), self.forward(negative_input).detach()


class Network(nn.Module):
    # hebbian_weights = nn.Parameter(torch.ones(10, 774).cuda())
    positive_labels = []
    negative_labels= []
    def __init__(self, dimension_configs):
        super().__init__()
        self.layers = []
        for i in range(len(dimension_configs) - 1):
            self.layers += [Layer(dimension_configs[i], dimension_configs[i + 1]).cuda()]
            
    def balanced_loss(self, positive_goodness, negative_goodness, alpha=4.0):
        delta = positive_goodness - negative_goodness
        per_instance_loss = torch.log(1 + torch.exp(-alpha * delta))
        return per_instance_loss.mean()  

    def mark_data(self, data, label):
        marked_data = data.clone().cuda()
        marked_data[:, :10] = 0
        marked_data[torch.arange(marked_data.size(0)), label] = 1
        return marked_data
    
    
    def predict(self, input_data):
        goodness_per_label = []
        for label in range(10):
            marked_data = self.mark_data(input_data, label)
            hebbian_weight_one = Layer.hebbian_weights_layer_one[label, :]
            hebbian_weight_two = Layer.hebbian_weights_layer_two[label, :]
        
            goodness = []
            for layer_num, layer in enumerate(self.layers):
                marked_data = layer(marked_data)
                if layer_num == 0:
                    goodness_value = (marked_data.pow(2) * hebbian_weight_one).mean(1)
                else:
                    goodness_value = (marked_data.pow(2) * hebbian_weight_two).mean(1)
                goodness.append(goodness_value)

            goodness_per_label.append(torch.sum(torch.stack(goodness), dim=0).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(dim=1)
    
    def compute_hebbian_activity(self, values):
        labels = values[:, :10].cuda()
        hebbian = values[:, 10:].cuda()
        hebbian_value = torch.mm(hebbian, self.hebbian_weights.T) * labels
        return hebbian_value.mean(1).cuda()
    
    
    def train_network(self, positive_goodness, negative_goodness,label):
        goodness_pos, goodness_neg = positive_goodness, negative_goodness
        positive_labels = goodness_pos[:, :10]
        negative_labels = negative_goodness[:, :10]

        Network.positive_labels = nn.Parameter(positive_labels.cuda())
        Network.negative_labels = nn.Parameter(negative_labels.cuda())
        for i, layer in enumerate(self.layers):
            print('Training Layer', i, '...')
            goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, i , label)
            


def load_CIFAR10_data(train_batch_size=30000, test_batch_size=6000):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        Lambda(lambda x: torch.flatten(x))
    ])

    training_data_loader = DataLoader(
        CIFAR10('./data/', train=True, download=True, transform=data_transformation),
        batch_size=train_batch_size,
        shuffle=False
    )

    testing_data_loader = DataLoader(
        CIFAR10('./data/', train=False, download=True, transform=data_transformation),
        batch_size=test_batch_size,
        shuffle=False
    )

    return training_data_loader, testing_data_loader

def load_FashionMNIST_data(train_batch_size=50000, test_batch_size=10000):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.2860,), (0.3530,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    training_data_loader = DataLoader(
        FashionMNIST('./data/', train=True, download=True, transform=data_transformation),
        batch_size=train_batch_size,
        shuffle=False
    )

    testing_data_loader = DataLoader(
        FashionMNIST('./data/', train=False, download=True, transform=data_transformation),
        batch_size=test_batch_size,
        shuffle=False
    )

    return training_data_loader, testing_data_loader

def load_MNIST_data(train_batch_size=5000, test_batch_size=100):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    training_data_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=data_transformation),
        batch_size=train_batch_size,
        shuffle=False
    )

    testing_data_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=data_transformation),
        batch_size=test_batch_size,
        shuffle=False
    )

    return training_data_loader, testing_data_loader


def create_positive_data(data, label):
    positive_data = data.clone()
    positive_data[:, :10] = 0.0

    for i in range(positive_data.shape[0]):
        positive_data[i][label[i]] = 1.0

    return positive_data


def create_negative_data(data, label, seed=None):
    if seed is not None:
        random.seed(seed)

    negative_data = data.clone()
    negative_data[:, :10] = 0.0

    for i in range(negative_data.shape[0]):
        possible_answers = list(range(10))
        possible_answers.remove(label[i])
        false_label = random.choice(possible_answers)
        negative_data[i][false_label] = 1.0

    return negative_data


def prepare_data():
    torch.manual_seed(4321)
    training_data_loader, testing_data_loader = load_FashionMNIST_data()

    training_data, training_data_label = next(iter(training_data_loader))

    testing_data, testing_data_label = next(iter(testing_data_loader))
    testing_data, testing_data_label = testing_data.cuda(), testing_data_label.cuda()

    print(f"Training Data: ", training_data)
    print(f"Training Data Label: ", training_data_label)

    training_data, training_data_label = training_data.cuda(), training_data_label.cuda()

    positive_data = create_positive_data(training_data, training_data_label)
    print(f"Positive Data: ", positive_data)

    negative_data = create_negative_data(training_data, training_data_label, seed=1234)
    print(f"Negative Data: ", negative_data)

    return positive_data, negative_data, training_data, training_data_label, testing_data, testing_data_label


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.manual_seed(1234)
    positive_data, negative_data, training_data, training_data_label, testing_data, testing_data_label = prepare_data()
    network = Network([784, 500, 500]).cuda() #3072
    network.train_network(positive_data, negative_data , training_data_label)

    print("Training Accuracy: ", network.predict(training_data).eq(training_data_label).float().mean().item())
    print("Testing Accuracy: ", network.predict(testing_data).eq(testing_data_label).float().mean().item())