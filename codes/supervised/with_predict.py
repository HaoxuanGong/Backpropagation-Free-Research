import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

class Config:
    # Neural network architecture
    dims = [784, 500, 500]  # Adjust the dimensions here
    
    # Training parameters
    learning_rate = 0.03  # Adjust the learning rate here
    threshold = 2.0  # Adjust the threshold here
    num_epochs = 1000  # Adjust the number of epochs here
    train_batch_size = 50000  # Adjust the train batch size here
    test_batch_size = 10000  # Adjust the test batch size here

class BlackBoxModel(nn.Module):
    def __init__(self):
        super(BlackBoxModel, self).__init__()
        self.layer1 = nn.Linear(500, 100)  # Adjust sizes as needed
        self.layer2 = nn.Linear(100, 500)  # Adjust sizes to match your architecture
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return x

def MNIST_loaders():
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True, download=True, transform=transform),
        batch_size=Config.train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False, download=True, transform=transform),
        batch_size=Config.test_batch_size, shuffle=False)

    return train_loader, test_loader

def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Layer(Config.dims[d], Config.dims[d + 1]).cuda() for d in range(len(Config.dims) - 1)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            x_pos, x_neg = layer.train(x_pos, x_neg)

class Layer(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=Config.learning_rate)
        self.threshold = Config.threshold
        self.num_epochs = Config.num_epochs

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def load_and_preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert to tensor
    image = TF.to_tensor(image)

    # Normalize using the same mean and standard deviation as MNIST
    image = TF.normalize(image, [0.1307], [0.3081])

    # Flatten the image
    image = torch.flatten(image)

    # Add a batch dimension
    image = image.unsqueeze(0)

    return image

def predict_single_image(image, model):
    # Predict the label for the image
    prediction = model.predict(image)
    return prediction.item()

if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net()
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)
    
    net.train(x_pos, x_neg)
    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()
    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())

    # Load and preprocess your JPEG image =====================================================================
    image_path = './5.png'  # Update this path
    processed_image = load_and_preprocess_image(image_path)
    processed_image = processed_image.cuda()  # Make sure it's on the same device as your model

    # Predict
    predicted_label = predict_single_image(processed_image, net)
    print(f"Predicted label: {predicted_label}")
