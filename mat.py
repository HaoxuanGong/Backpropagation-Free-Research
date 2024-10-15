import torch

# Example inputs
batch_size = 4
num_classes = 3
num_features = 5

# Labels (one-hot encoded)
labels = torch.tensor([[0., 0., 1.],  # Label for the first sample is class 2    [0.9395, 0.3481, 0.7388, 0.2050, 0.4010]]) # Weights for class 2
                       [1., 0., 0.],  # Label for the second sample is class 0   [0.5123, 0.6419, 0.2562, 0.7383, 0.1919],  # Weights for class 0
                       [0., 1., 0.],  # Label for the third sample is class 1    [0.0925, 0.5745, 0.2769, 0.1002, 0.8380],  # Weights for class 1
                       [0., 0., 1.]]) # Label for the fourth sample is class 2   [0.9395, 0.3481, 0.7388, 0.2050, 0.4010]]) # Weights for class 2

# Hebbian weights (weights per class)
hebbian_weights = torch.tensor([[0.5123, 0.6419, 0.2562, 0.7383, 0.1919],  # Weights for class 0
                                [0.0925, 0.5745, 0.2769, 0.1002, 0.8380],  # Weights for class 1
                                [0.9395, 0.3481, 0.7388, 0.2050, 0.4010]]) # Weights for class 2

# Output activations from the network
output = torch.tensor([[0.5, 0.2, 0.3, 0.7, 0.9],  # Activations for sample 1
                       [0.4, 0.6, 0.5, 0.1, 0.3],  # Activations for sample 2
                       [0.3, 0.9, 0.1, 0.2, 0.4],  # Activations for sample 3
                       [0.6, 0.8, 0.2, 0.5, 0.7]]) # Activations for sample 4

positive_goodness =  torch.mm(labels, hebbian_weights)
negative_goodness = torch.mm(1 - labels, hebbian_weights)
# Step 3: Compute the average of the negative weights
num_neg_classes = labels.size(1) - 1  # Number of incorrect classes
average_negative_weights = negative_goodness / num_neg_classes
print(average_negative_weights)