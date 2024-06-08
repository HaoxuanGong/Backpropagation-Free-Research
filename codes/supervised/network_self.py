import torch
import torch.nn as nn
from codes.supervised.layer_self import Layer


class Network(nn.Module):

    def __init__(self, dimension_configs):
        super().__init__()
        self.layers = []
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
            for layer in self.layers:
                marked_data = layer(marked_data)
                goodness += [marked_data.pow(2).sum(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(dim=1)

    def train_network(self, positive_goodness, negative_goodness):
        goodness_pos, goodness_neg = positive_goodness, negative_goodness
        for i, layer in enumerate(self.layers):
            print('Training Layer', i, '...')
            goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, i)
