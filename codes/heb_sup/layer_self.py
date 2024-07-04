import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam

number_of_epochs = 200


class Layer(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, d_type=None, is_hinge_loss=False):
        super().__init__(in_features, out_features, bias, device, d_type)
        self.activation = torch.nn.ReLU()
        self.learning_rate = 0.06
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self.layer_weights = nn.Parameter(torch.ones(2, 500))
        self.threshold = 2.0
        self.num_of_epochs = number_of_epochs
        self.is_hinge_loss = is_hinge_loss
        self.weight_optimizer = Adam([self.layer_weights], lr=0.15)  # Optimizer for layer_weights

    def forward(self, input: Tensor) -> Tensor:
        normalized_input = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        output = torch.mm(normalized_input, self.weight.T) + self.bias.unsqueeze(0)
        return self.activation(output)

    def hinge_loss(self, positive_goodness, negative_goodness, delta=1.0, is_second_phase=False):
        if is_second_phase:
            threshold = self.threshold * 2
        else:
            threshold = self.threshold
        positive_loss = torch.clamp(delta - (positive_goodness - threshold), min=0)
        negative_loss = torch.clamp(delta - (threshold - negative_goodness), min=0)
        return torch.cat([positive_loss, negative_loss]).mean()

    def exponential_hinge_loss(self, positive_goodness, negative_goodness, delta=1.0, is_second_phase=False):
        if is_second_phase:
            threshold = self.threshold * 2
        else:
            threshold = self.threshold
        positive_loss = torch.exp(torch.clamp(delta - (positive_goodness - threshold), min=0)) - 1
        negative_loss = torch.exp(torch.clamp(delta - (threshold - negative_goodness), min=0)) - 1
        return torch.cat([positive_loss, negative_loss]).mean()

    def soft_plus_loss(self, positive_goodness, negative_goodness, is_second_phase=False):
        if is_second_phase:
            threshold = self.threshold * 3
        else:
            threshold = self.threshold
        return torch.log(1 + torch.exp(torch.cat([
            -positive_goodness + threshold,
            negative_goodness - threshold]))).mean()

    def contrastive_loss(self, positive_goodness, negative_goodness, margin=1.0):
        # Contrastive loss: positive_goodness should be higher, negative_goodness should be lower
        positive_loss = (positive_goodness - 1).pow(2)  # We want positive_goodness close to 1
        negative_loss = torch.clamp(margin - negative_goodness, min=0).pow(2)  # We want negative_goodness far from 1
        loss = (positive_loss + negative_loss).mean()
        return loss

    def margin_ranking_loss(self, positive_scores, negative_scores):
        squared_diff = (positive_scores - negative_scores).pow(2)
        loss = squared_diff.mean()
        return loss

    def plot_goodness(self, positive_goodness_history, negative_goodness_history,
                      positive_unaltered_goodness_history, negative_unaltered_goodness_history):
        epochs = range(1, self.num_of_epochs + 1)

        plt.figure(figsize=(14, 7))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, positive_goodness_history, label='Altered Positive Goodness')
        plt.plot(epochs, positive_unaltered_goodness_history, label='Unaltered Positive Goodness')
        plt.legend()
        plt.title('Positive Goodness Comparison Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Goodness')

        plt.subplot(2, 1, 2)
        plt.plot(epochs, negative_goodness_history, label='Altered Negative Goodness')
        plt.plot(epochs, negative_unaltered_goodness_history, label='Unaltered Negative Goodness')
        plt.legend()
        plt.title('Negative Goodness Comparison Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Goodness')

        plt.tight_layout()
        plt.show()

    def train_layer(self, positive_input, negative_input, layer_num):
        positive_goodness_history = []
        negative_goodness_history = []
        positive_unaltered_goodness_history = []
        negative_unaltered_goodness_history = []
        for _ in tqdm(range(self.num_of_epochs)):
            positive_output = self.forward(positive_input)  # Shape: [batch_size, 500]
            negative_output = self.forward(negative_input)
            layer_weight_row = self.layer_weights[layer_num, :]

            positive_goodness = (positive_output.pow(2) * layer_weight_row).mean(1)  # Shape: [batch_size]
            negative_goodness = (negative_output.pow(2) * layer_weight_row).mean(1)

            positive_unaltered_goodness = positive_output.pow(2).mean(1)
            negative_unaltered_goodness = negative_output.pow(2).mean(1)

            positive_goodness_history.append(positive_goodness.mean().item())
            negative_goodness_history.append(negative_goodness.mean().item())
            positive_unaltered_goodness_history.append(positive_unaltered_goodness.mean().item())
            negative_unaltered_goodness_history.append(negative_unaltered_goodness.mean().item())
            if self.is_hinge_loss:
                loss = self.exponential_hinge_loss(positive_goodness, negative_goodness)
            else:
                loss = self.soft_plus_loss(positive_goodness, negative_goodness)

            self.optimizer.zero_grad()
            self.weight_optimizer.zero_grad()  # Clear gradients for layer_weights
            loss.backward()
            self.optimizer.step()
            self.weight_optimizer.step()  # Update layer_weights

        print(self.layer_weights)
        self.plot_goodness(positive_goodness_history, negative_goodness_history,
                           positive_unaltered_goodness_history, negative_unaltered_goodness_history)
        return self.forward(positive_input).detach(), self.forward(negative_input).detach()
