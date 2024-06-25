import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from torch.optim import Adam

number_of_epochs = 200


class Layer(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, d_type=None, is_hinge_loss=True):
        super().__init__(in_features, out_features, bias, device, d_type)
        self.activation = torch.nn.ReLU()
        self.learning_rate = 0.08
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self.threshold = 2.0
        self.num_of_epochs = number_of_epochs
        self.is_hinge_loss = is_hinge_loss

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

    def soft_plus_loss(self, positive_goodness, negative_goodness, is_second_phase=False):
        if is_second_phase:
            threshold = self.threshold * 2
        else:
            threshold = self.threshold
        return torch.log(1 + torch.exp(torch.cat([
            -positive_goodness + threshold,
            negative_goodness - threshold]))).mean()

    def train_layer(self, positive_input, negative_input, layer_num):
        for _ in tqdm(range(self.num_of_epochs)):
            positive_goodness = self.forward(positive_input).pow(2).mean(1)
            negative_goodness = self.forward(negative_input).pow(2).mean(1)
            if self.is_hinge_loss:
                loss = self.hinge_loss(positive_goodness, negative_goodness)
            else:
                loss = self.soft_plus_loss(positive_goodness, negative_goodness)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.forward(positive_input).detach(), self.forward(negative_input).detach()

    # def second_phase_train(self, positive_input, negative_input, layer_weights):
    #
    #     for _ in tqdm(range(self.num_of_epochs)):
    #         first_layer_positive_goodness_sum = self.forward(positive_input).pow(2).mean(1)
    #         first_layer_negative_goodness_sum = self.forward(negative_input).pow(2).mean(1)
    #
    #         second_layer_positive_goodness_sum = self.forward(self.forward(positive_input)).pow(2).mean(1)
    #         second_layer_negative_goodness_sum = self.forward(self.forward(negative_input)).pow(2).mean(1)
    #
    #         positive_sum = first_layer_negative_goodness_sum * layer_weights[0] + second_layer_positive_goodness_sum * layer_weights[1]
    #         negative_sum = first_layer_positive_goodness_sum * layer_weights[0] + second_layer_negative_goodness_sum * layer_weights[1]
    #
    #         loss = self.hinge_loss(positive_sum, negative_sum, is_second_phase=True)

