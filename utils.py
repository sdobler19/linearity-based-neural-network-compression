import torch.nn as nn
import torch
import numpy as np
import keras
import pandas as pd
from Preparation.OpenML_prep import evaluate
from compression.compression import model_compression, get_activation_rates


# Function to check if all weights and bias of a neuron are non-negative
def neuron_all_positive(neuron_weights, bias):
    return all(neuron_weights >= 0) and (bias >= 0)


# Function to identify neurons in a layer with all non-negative weights and bias
def check_layer(layer): 
    linear_in_layer = [neuron_all_positive(neuron_weights, bias) for neuron_weights, bias in zip(layer.weight,layer.bias)]
    idx_list = [idx for idx, linear in enumerate(linear_in_layer) if linear]
    return idx_list


# Checks each linear layer in a model to find provably linear neurons
def check_provable_linearity(model):  
    out = []
    for layer_idx, layer in enumerate(model.layers.children()):
        if not isinstance(layer, torch.nn.modules.linear.Linear):
            continue
        elif check_layer(layer):
            out.append([layer_idx, check_layer(layer)])

    return out


# Class to convert a Keras model to a PyTorch model
class keras2torch_converter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList()

        # Convert each Keras layer to a PyTorch equivalent
        for lay in model.layers:
            self.layers.append(nn.Linear(lay.input.shape[1], lay.output.shape[1], lay.use_bias))
            
            if lay.activation == keras.src.activations.relu:
                self.layers.append(nn.ReLU())
            elif lay.activation == keras.src.activations.sigmoid:
                self.layers.append(nn.Sigmoid())
            elif lay.activation == keras.src.activations.softmax:
                self.layers.append(nn.Softmax())
            elif lay.activation == keras.src.activations.tanh:
                self.layers.append(nn.Tanh())
            elif lay.activation == keras.src.activations.Linear:
                continue 
            else:
                assert Exception(f"Activation Function: {lay.activation} not implemented")


        # transfering weights
        for i in range(len(self.layers)):
            if not isinstance(self.layers[i], torch.nn.Linear):
                continue
            self.layers[i].weight.data = torch.from_numpy(np.transpose(model.get_weights()[i]))
            self.layers[i].bias.data = torch.from_numpy(model.get_weights()[i + 1])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    # Forward pass needed for the 
    def extended_relu_forward(self, x):
        out = list()
        for layer in self.layers:
            x = layer(x)
            if not isinstance(layer, torch.nn.Linear):
                out.append(x)
        return out


# Evaluate model accuracy on a data loader
def loader_eval(loader, model):
    score = 0
    count = 0
    for data, label in loader:
        pred = model.forward(data)
        score += (torch.argmax(pred, 1) == label).float().sum()
        count += len(label)
    return score/count


# Perform model compression and track performance across thresholds
def compression_loop(model, prune_data, test_data, finegrain = 5, layerthreshold = "best"):
    steps = [step/100 for step in range(5, 105, finegrain)]

    actrates = get_activation_rates(model, prune_data) # Get activation rates previously to save computation cost

    n_weights_lay = []
    n_weights_skip = []
    loss = []
    for compression_rate in steps:
        com_model = model_compression(model)
        model.act_rate = actrates
        com_model.compression(compression_rate, prune_data, layerthreshold)
        layer, skip = com_model.get_num_weights(True)
        n_weights_lay.append(layer)
        n_weights_skip.append(skip)
        if isinstance(test_data, torch.utils.data.dataloader.DataLoader):
            loss.append(loader_eval(test_data, com_model).item())
        else:
            X_eval, y_eval = test_data
            loss.append(evaluate(X_eval, y_eval, com_model))

    plotting_data = pd.DataFrame({"compression_thresholds": steps, 
                                "n_weights_lay": n_weights_lay, "n_weights_skip" : n_weights_skip, 
                                "Loss": loss})

    plotting_data["n_weights"] = plotting_data["n_weights_lay"] + plotting_data["n_weights_skip"]

    return plotting_data