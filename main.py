# This code is based on the implementation of Mohammad Pezeshki available at
# https://github.com/mohammadpz/pytorch_forward_forward and licensed under the MIT License.
# Modifications/Improvements to the original code have been made by Jmaes Gong, Bruce LI.
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.optim import Adam
from network_self import Network

def load_CIFAR10_data(train_batch_size=50000, test_batch_size=10000):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Lambda(lambda x: torch.flatten(x))
    ])

    training_data_loader = DataLoader(
        CIFAR10('./data/', train=True, download=True, transform=data_transformation),
        batch_size=train_batch_size,
        shuffle=True
    )

    testing_data_loader = DataLoader(
        CIFAR10('./data/', train=False, download=True, transform=data_transformation),
        batch_size=test_batch_size,
        shuffle=False
    )

    return training_data_loader, testing_data_loader

def load_KMNIST_data(train_batch_size=50000, test_batch_size=10000):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.1010,), (0.9510,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    training_data_loader = DataLoader(
        KMNIST('./data/', train=True, download=True, transform=data_transformation),
        batch_size=train_batch_size,
        shuffle=True
    )

    testing_data_loader = DataLoader(
        KMNIST('./data/', train=False, download=True, transform=data_transformation),
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
        shuffle=True
    )

    testing_data_loader = DataLoader(
        FashionMNIST('./data/', train=False, download=True, transform=data_transformation),
        batch_size=test_batch_size,
        shuffle=False
    )

    return training_data_loader, testing_data_loader

def load_MNIST_data(train_batch_size=50000, test_batch_size=10000):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
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


def prepare_data(dataset='FashionMNIST'):
    torch.manual_seed(1234)
    if dataset == 'MNIST':
        training_data_loader, testing_data_loader = load_MNIST_data()
    elif dataset == 'FashionMNIST':
        training_data_loader, testing_data_loader = load_FashionMNIST_data()
    elif dataset == 'KMNIST':
        training_data_loader, testing_data_loader = load_KMNIST_data()
    elif dataset == 'CIFAR10':
        training_data_loader, testing_data_loader = load_CIFAR10_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    training_data, training_data_label = next(iter(training_data_loader))

    testing_data, testing_data_label = next(iter(testing_data_loader))
    testing_data, testing_data_label = testing_data.cuda(), testing_data_label.cuda()

    training_data, training_data_label = training_data.cuda(), training_data_label.cuda()

    print(f"Creating Positive and Negative Data for : {dataset}")
    positive_data = create_positive_data(training_data, training_data_label)
    negative_data = create_negative_data(training_data, training_data_label, seed=1234)

    return positive_data, negative_data, training_data, training_data_label, testing_data, testing_data_label

if __name__ == "__main__":
    torch.manual_seed(1234)
    # dataset='MNIST', 'FashionMNIST', 'KMNIST', 'CIFAR10'
    positive_data, negative_data, training_data, training_data_label, testing_data, testing_data_label = prepare_data(dataset='MNIST') 

    # network init
    network = Network([784, 50, 50]).cuda()
    number_of_epochs = 300

    # Override default hyperparameters if needed
    # adjust hyperparameters in layer_self.py
    for layer in network.layers:
        layer.num_of_epochs = number_of_epochs

    network.train_network(positive_data, negative_data, training_data_label, tsne=False) # True for t-Sne visualizaton of layer output

    print("Training Accuracy: ", network.predict(training_data).eq(training_data_label).float().mean().item())

    print("Testing Accuracy: ", network.predict(testing_data).eq(testing_data_label).float().mean().item())