import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam

class Layer(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, d_type=None, is_hinge_loss=False):
        super().__init__(in_features, out_features, bias, device, d_type)
        # adjust hyperparameters here
        self.activation = torch.nn.ReLU()
        self.learning_rate = 0.07 # network optimzer lr
        self.nasf_learning_rate = 0.1 # neuron activity scaling factors optimzer lr
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self.nasf = nn.Parameter(torch.ones(2, 50)) # neuron activity scaling factors
        self.nasf_optimizer = Adam([self.nasf], lr=self.nasf_learning_rate)  # Optimizer for layer_weights
        self.threshold = 2.0
        self.num_of_epochs = 200
        self.is_hinge_loss = is_hinge_loss

    def forward(self, input: Tensor) -> Tensor:
        normalized_input = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        output = torch.mm(normalized_input, self.weight.T) + self.bias.unsqueeze(0)
        return self.activation(output)

    def hinge_loss(self, positive_goodness, negative_goodness, delta=1.0):
        positive_loss = torch.clamp(delta - (positive_goodness - self.threshold), min=0)
        negative_loss = torch.clamp(delta - (self.threshold - negative_goodness), min=0)

        return torch.cat([positive_loss, negative_loss]).mean()

    def exponential_hinge_loss(self, positive_goodness, negative_goodness, delta=1.0):
        positive_loss = torch.exp(torch.clamp(delta - (positive_goodness - self.threshold), min=0)) - 1
        negative_loss = torch.exp(torch.clamp(delta - (self.threshold - negative_goodness), min=0)) - 1
        return torch.cat([positive_loss, negative_loss]).mean()

    def soft_plus_loss(self, positive_goodness, negative_goodness):
        return torch.log(1 + torch.exp(torch.cat([-positive_goodness + self.threshold, negative_goodness - self.threshold]))).mean()

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
            layer_weight_row = self.nasf[layer_num, :]

            positive_goodness = (positive_output.pow(2) * layer_weight_row).mean(1)  # Shape: [batch_size]
            negative_goodness = (negative_output.pow(2) * layer_weight_row).mean(1)

            positive_unaltered_goodness = positive_output.pow(2).mean(1)
            negative_unaltered_goodness = negative_output.pow(2).mean(1)

            positive_goodness_history.append(positive_goodness.mean().item())
            negative_goodness_history.append(negative_goodness.mean().item())
            positive_unaltered_goodness_history.append(positive_unaltered_goodness.mean().item())
            negative_unaltered_goodness_history.append(negative_unaltered_goodness.mean().item())
            if self.is_hinge_loss:
                loss = self.hinge_loss(positive_goodness, negative_goodness)
            else:
                loss = self.soft_plus_loss(positive_goodness, negative_goodness)

            self.optimizer.zero_grad()
            self.nasf_optimizer.zero_grad()  # Clear gradients for layer_weights
            loss.backward()
            self.optimizer.step()
            self.nasf_optimizer.step()  # Update layer_weights

        print(self.nasf)
        #self.plot_goodness(positive_goodness_history, negative_goodness_history,positive_unaltered_goodness_history, negative_unaltered_goodness_history)
        return self.forward(positive_input).detach(), self.forward(negative_input).detach()
