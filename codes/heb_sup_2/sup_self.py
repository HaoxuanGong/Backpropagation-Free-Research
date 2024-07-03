import psutil
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from network_self import Network
import random
import time

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage_bytes = process.memory_info().rss
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # Convert to MB
    return memory_usage_mb

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

def train_network_and_measure_time(network, positive_data, negative_data):
    return network.train_network(positive_data, negative_data)


def load_KMNIST_data(train_batch_size=60000, test_batch_size=10000):
    data_transformation = Compose([
        ToTensor(),
         Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
    ])

    training_data_loader = DataLoader(
        KMNIST('./data/', train=True, download=True, transform=data_transformation),
        batch_size=train_batch_size,
        shuffle=False
    )

    testing_data_loader = DataLoader(
        KMNIST('./data/', train=False, download=True, transform=data_transformation),
        batch_size=test_batch_size,
        shuffle=False
    )

    return training_data_loader, testing_data_loader

def load_FashionMNIST_data(train_batch_size=60000, test_batch_size=10000):
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

def load_MNIST_data(train_batch_size=50000, test_batch_size=10000):
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
    torch.manual_seed(1234)
    training_data_loader, testing_data_loader = load_KMNIST_data()

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
    torch.manual_seed(1234)
    positive_data, negative_data, training_data, training_data_label, testing_data, testing_data_label = prepare_data()

    # Measure memory usage and time before training
    memory_before = measure_memory_usage()
    print(f"Memory Usage before training: {memory_before:.2f} MB")

    # Train the network and measure time
    network = Network([784, 500, 500]).cuda()
    trained_network, execution_time = train_network_and_measure_time(network, positive_data, negative_data)
    print(f"Training Execution Time: {execution_time:.2f} seconds")

    # Measure memory usage after training
    memory_after = measure_memory_usage()
    print(f"Memory Usage after training: {memory_after:.2f} MB")

    # Optionally, print memory difference
    memory_diff = memory_after - memory_before
    print(f"Memory increase during training: {memory_diff:.2f} MB")

    # Evaluate accuracy
    print("Training Accuracy: ", trained_network.predict(training_data).eq(training_data_label).float().mean().item())
    print("Testing Accuracy: ", trained_network.predict(testing_data).eq(testing_data_label).float().mean().item())