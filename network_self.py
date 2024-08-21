import torch
import torch.nn as nn
from layer_self import Layer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
            for layer_num, layer in enumerate(self.layers):
                marked_data = layer(marked_data)
                layer_weights = layer.nasf[layer_num, :]
                goodness_value = (marked_data.pow(2) * layer_weights).mean(1)
                goodness.append(goodness_value)
            goodness_per_label.append(torch.sum(torch.stack(goodness), dim=0).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(dim=1)
    
    def tsne_visualization(self,data, labels, title, cmap='tab10'):
        tsne = TSNE(n_components=2, random_state=1234)
        data_2d = tsne.fit_transform(data.cpu().numpy())

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels.cpu().numpy(), cmap=cmap, alpha=0.6)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def train_network(self, positive_goodness, negative_goodness,training_data_label,tsne):
        goodness_pos, goodness_neg = positive_goodness, negative_goodness
        for i, layer in enumerate(self.layers):
            print('Training Layer', i, '...')
            goodness_pos, goodness_neg = layer.train_layer(goodness_pos, goodness_neg, i)
            if tsne:
                print("generating t-Sne visualization")
                self.tsne_visualization(goodness_pos, training_data_label, i)