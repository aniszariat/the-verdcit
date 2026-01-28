import torch
import torch.nn as nn
from activationFunction import GELU


# A neural network to illustrate shortcut connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # 1 Implements five layers
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            # 2 Compute the output of the current layer
            layer_output = layer(x)
            # 3 Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
# 1 Specifies random seed for the initial weights for reproducibility
torch.manual_seed(123)
# initialize a neural network without shortcut connections
# model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
# initialize a neural network with shortcut connections
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)


# we implement a function that computes the gradients in the model’s backward pass:
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.0]])
    loss = nn.MSELoss()
    # Calculates loss based on how close the target and output are
    loss = loss(output, target)
    # Backward pass to calculate the gradients
    loss.backward()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


print_gradients(model_without_shortcut, sample_input)
