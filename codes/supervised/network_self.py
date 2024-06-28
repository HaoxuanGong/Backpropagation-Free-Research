import torch
import torch.nn as nn
from layer_self import Layer
from layer_self import number_of_epochs
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt


def plot_metrics(neg_min, neg_max, neg_mean, pos_min, pos_max, pos_mean):
    epochs = range(1, len(neg_min) + 1)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, neg_min, 'b', label='Min Negative Sum')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Min Negative Sum')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, neg_max, 'r', label='Max Negative Sum')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Max Negative Sum')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, neg_mean, 'g', label='Mean Negative Sum')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Mean Negative Sum')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, pos_min, 'c', label='Min Positive Sum')
    plt.plot(epochs, pos_max, 'm', label='Max Positive Sum')
    plt.plot(epochs, pos_mean, 'y', label='Mean Positive Sum')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Positive Sum Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()

class Network(nn.Module):

    def __init__(self, dimension_configs):
        super().__init__()
        self.layers = []
        self.layer_weights = nn.Parameter(torch.ones(2))
        for i in range(len(dimension_configs) - 1):
            self.layers += [Layer(dimension_configs[i], dimension_configs[i + 1]).cuda()]



    def mark_data(self, data, label):
        marked_data = data.clone().cuda()
        marked_data[:, :10] = 0
        marked_data[torch.arange(marked_data.size(0)), label] = 1
        return marked_data

    def predict(self, input_data):
        goodness_per_label = []
        for label in range(10):
            marked_data = self.mark_data(input_data, label)
            goodness = []
            for layer_num, layer in enumerate(self.layers):
                marked_data = layer(marked_data)
                goodness_value = marked_data.pow(2).mean(1) * self.layer_weights[layer_num].item()
                # goodness_value = marked_data.pow(2).mean(1)
                goodness.append(goodness_value)
            goodness_per_label.append(sum(goodness).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(dim=1)

    def train_network(self, positive_goodness, negative_goodness):
        # Lists to store min, max, and mean values
        negative_min_list, negative_max_list, negative_mean_list = [], [], []
        positive_min_list, positive_max_list, positive_mean_list = [], [], []

        optimizer = Adam(self.parameters(), lr=0.01)
        goodness_pos, goodness_neg = positive_goodness, negative_goodness
        for i, layer in enumerate(self.layers):
            print('Training Layer', i, '...')
            goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, i)

        for _ in tqdm(range(400)):
            optimizer.zero_grad()
            second_layer_negative_input = self.layers[0].forward(negative_goodness)
            first_layer_negative_goodness_sum = second_layer_negative_input.pow(2).mean(1)

            second_layer_negative_goodness_sum = self.layers[1].forward(second_layer_negative_input).pow(
                2).mean(1)

            negative_sum = first_layer_negative_goodness_sum * self.layer_weights[
                0] + second_layer_negative_goodness_sum * self.layer_weights[1]

            neg_loss = self.layers[0].exponential_negative_hinge_loss(negative_sum)
            neg_loss.backward()
            negative_min_list.append(negative_sum.min().item())
            negative_max_list.append(negative_sum.max().item())
            negative_mean_list.append(negative_sum.mean().item())
            optimizer.step()

        for _ in tqdm(range(400)):
            optimizer.zero_grad()
            second_layer_positive_input = self.layers[0].forward(positive_goodness)
            first_layer_positive_goodness_sum = second_layer_positive_input.pow(2).mean(1)

            second_layer_positive_goodness_sum = self.layers[1].forward(second_layer_positive_input).pow(
                2).mean(1)
            positive_sum = first_layer_positive_goodness_sum * self.layer_weights[
                0] + second_layer_positive_goodness_sum * self.layer_weights[1]

            pos_loss = self.layers[0].exponential_positive_hinge_loss(positive_sum)
            pos_loss.backward()
            # Store min, max, and mean values
            positive_min_list.append(positive_sum.min().item())
            positive_max_list.append(positive_sum.max().item())
            positive_mean_list.append(positive_sum.mean().item())
            optimizer.step()

        # with torch.no_grad():
        #     self.layer_weights[0] = 0.8
        #     self.layer_weights[1] = 1.2

        print(self.layer_weights)
        plot_metrics(negative_min_list, negative_max_list, negative_mean_list, positive_min_list, positive_max_list,
                     positive_mean_list)
