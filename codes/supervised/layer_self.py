import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from torch.optim import Adam


class Layer(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, d_type=None, is_hinge_loss=True):
        super().__init__(in_features, out_features, bias, device, d_type)
        self.activation = torch.nn.ReLU()
        self.optimizer = Adam(self.parameters(), lr=0.06)
        self.threshold = 2.0
        self.num_of_epochs = 500
        self.is_hinge_loss = is_hinge_loss

    def forward(self, input: Tensor) -> Tensor:
        normalized_input = input / (input.norm(2, 1, keepdim=True) + 1e-4)
        output = torch.mm(normalized_input, self.weight.T) + self.bias.unsqueeze(0)
        return self.activation(output)

    def hinge_loss(self, positive_goodness, negative_goodness, delta=1.0):
        positive_loss = torch.clamp(delta - (positive_goodness - self.threshold), min=0)
        negative_loss = torch.clamp(delta - (self.threshold - negative_goodness), min=0)
        return torch.cat([positive_loss, negative_loss]).mean()

    def soft_plus_loss(self, positive_goodness, negative_goodness):
        return torch.log(1 + torch.exp(torch.cat([
            -positive_goodness + self.threshold,
            negative_goodness - self.threshold]))).mean()

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






