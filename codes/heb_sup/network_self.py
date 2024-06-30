import torch
import torch.nn as nn
from layer_self import Layer
from layer_self import number_of_epochs
from tqdm import tqdm
from torch.optim import Adam


class Network(nn.Module):

    def __init__(self, dimension_configs):
        super().__init__()
        self.layers = []
        #self.layer_weights = nn.Parameter(torch.ones(2))
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
                layer_weights = layer.layer_weights[layer_num, :]
                goodness_value = (marked_data.pow(2) * layer_weights).mean(1)
                # goodness_value = marked_data.pow(2).mean(1)
                goodness.append(goodness_value)
            goodness_per_label.append(torch.sum(torch.stack(goodness), dim=0).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(dim=1)

    def train_network(self, positive_goodness, negative_goodness):
        #optimizer = Adam(self.parameters(), lr=0.001)
        goodness_pos, goodness_neg = positive_goodness, negative_goodness
        for i, layer in enumerate(self.layers):
            print('Training Layer', i, '...')
            goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, i)

        # for _ in tqdm(range(50)):
        #     optimizer.zero_grad()
        #     first_layer_positive_goodness_sum = self.layers[0].forward(positive_goodness).pow(2).mean(1)
        #     first_layer_negative_goodness_sum = self.layers[0].forward(negative_goodness).pow(2).mean(1)

        #     second_layer_positive_goodness_sum = self.layers[1].forward(self.layers[0].forward(positive_goodness)).pow(
        #         2).mean(1)
        #     second_layer_negative_goodness_sum = self.layers[1].forward(self.layers[0].forward(negative_goodness)).pow(
        #         2).mean(1)

        #     positive_sum = first_layer_positive_goodness_sum * self.layer_weights[
        #         0] + second_layer_positive_goodness_sum * self.layer_weights[1]
        #     negative_sum = first_layer_negative_goodness_sum * self.layer_weights[
        #         0] + second_layer_negative_goodness_sum * self.layer_weights[1]

        #     loss = self.layers[0].soft_plus_loss(positive_sum, negative_sum, is_second_phase=True)
        #     loss.backward()
        #     optimizer.step()


        # with torch.no_grad():
        #     self.layer_weights[0] = 0.8
        #     self.layer_weights[1] = 1.2

        #print(self.layer_weights)


