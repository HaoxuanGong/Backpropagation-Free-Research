import psutil
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from network_self import Network
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage_bytes = process.memory_info().rss
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # Convert to MB
    return memory_usage_mb

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

def prepare_data():
    torch.manual_seed(1234)
    training_data_loader, testing_data_loader = load_MNIST_data()

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

def tsne_visualization(data, labels, title, cmap='hsv'):
    tsne = TSNE(n_components=2, random_state=1234)
    data_2d = tsne.fit_transform(data.cpu().numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels.cpu().numpy(), cmap=cmap, alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(1234)
    positive_data, negative_data, training_data, training_data_label, testing_data, testing_data_label = prepare_data()

    # Measure memory usage before training
    # memory_before = measure_memory_usage()
    # print(f"Memory Usage before training: {memory_before:.2f} MB")

    # Train the network
    network = Network([784, 2000, 2000, 2000, 2000]).cuda()
    network.train_network(positive_data, negative_data)

    # Visualize using t-SNE
    #predictions = network.predict(testing_data)
    #print(predictions)
    #tsne_visualization(training_data, training_data_label, "t-SNE Visualization of MNIST Data")
    #tsne_visualization(testing_data, predictions, "t-SNE Visualization of Network Predictions")

    print("Training Accuracy: ", network.predict(training_data).eq(training_data_label).float().mean().item())

    print("Testing Accuracy: ", network.predict(testing_data).eq(testing_data_label).float().mean().item())
