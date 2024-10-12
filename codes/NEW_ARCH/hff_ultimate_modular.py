import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam

epochs = 400
class HebbianLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.hebbian_weights = nn.Parameter(torch.ones(10, out_features).cuda())  # 10 classes
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
            positive_output = self.forward(positive_input)  # Shape: [batch_size, 500]
            negative_output = self.forward(negative_input)
            positive_goodness = self.compute_goodness(pos_labels, self.hebbian_weights, positive_output)
            negative_goodness = self.compute_goodness(neg_labels, self.hebbian_weights, negative_output)
            
            loss = torch.log(1 + torch.exp(torch.cat([
                -positive_goodness + 0.0, # threshold = 0
                negative_goodness - 0.0
            ]))).mean()

            self.optimizer.zero_grad()
            self.hebbian_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.hebbian_optimizer.step()

        return self.forward(positive_input).detach(), self.forward(negative_input).detach()


class HebbianNetwork(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList([HebbianLayer(layers_config[i], layers_config[i+1]).cuda() for i in range(len(layers_config) - 1)])

    def predict(self, input_data):
        goodness_per_label = []
        for label in range(10): 
            marked_data = self.mark_data(input_data, label)  # Mark the data with the current label
            goodness_values = []
            for i, layer in enumerate(self.layers):
                marked_data = layer(marked_data)
                hebbian_weight = layer.hebbian_weights[label, :]
                goodness_value = (marked_data * hebbian_weight).mean(1)
                goodness_values.append(goodness_value)
            goodness_per_label.append(torch.sum(torch.stack(goodness_values), dim=0).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(dim=1)


    def mark_data(self, input_data, label):
        marked_data = input_data.clone()
        marked_data[:, :10] = 0
        marked_data[:, label] = 1
        return marked_data
    
    def create_negative_data(self, data, label, seed=None):
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
    

    def train_network(self, positive_goodness, negative_goodness, training_data, training_data_label):
        for epoch in (range(5)):
            print(f'Epoch {epoch + 1}')
            goodness_pos, goodness_neg = positive_goodness, negative_goodness
            negative_goodness = self.create_negative_data(training_data, training_data_label)
            goodness_neg = negative_goodness
            positive_labels = goodness_pos[:, :10]
            negative_labels = negative_goodness[:, :10]

            positive_labels = nn.Parameter(positive_labels.cuda())
            negative_labels = nn.Parameter(negative_labels.cuda())
            for i, layer in enumerate(self.layers):
                print('Training Layer', i, '...')
                goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, positive_labels , negative_labels)

            

def load_MNIST_data(train_batch_size=5000, test_batch_size=1000):
    data_transformation = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))  # Flattening the 28x28 images into 1D vectors
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


def create_data(data, label, seed=None):
    if seed is not None:
        random.seed(seed)
    positive_data = data.clone()
    positive_data[:, :10] = 0.0
    for i in range(positive_data.shape[0]):
        positive_data[i][label[i]] = 1.0

    negative_data = data.clone()
    negative_data[:, :10] = 0.0
    for i in range(negative_data.shape[0]):
        possible_answers = list(range(10))
        possible_answers.remove(label[i])
        false_label = random.choice(possible_answers)
        negative_data[i][false_label] = 1.0
    return positive_data , negative_data


def prepare_data():
    torch.manual_seed(4321)
    training_data_loader, testing_data_loader = load_MNIST_data()
    training_data, training_data_label = next(iter(training_data_loader))
    testing_data, testing_data_label = next(iter(testing_data_loader))
    testing_data, testing_data_label = testing_data.cuda(), testing_data_label.cuda()
    # print(f"Training Data: ", training_data)
    print(f"Training Data Label: ", training_data_label)
    training_data, training_data_label = training_data.cuda(), training_data_label.cuda()
    positive_data , negative_data = create_data(training_data, training_data_label)
    
    return positive_data, negative_data, training_data, training_data_label, testing_data, testing_data_label

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.manual_seed(1234)
    positive_data, negative_data, training_data, training_data_label, testing_data, testing_data_label = prepare_data()
    network = HebbianNetwork([784, 500, 200 , 300]).cuda()  # MNI
    network.train_network(positive_data, negative_data, training_data, training_data_label)
    print("Training Accuracy: ", network.predict(training_data).eq(training_data_label).float().mean().item())
    print("Testing Accuracy: ", network.predict(testing_data).eq(testing_data_label).float().mean().item())