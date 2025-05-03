import torch.nn as nn
import copy
import torch

# Compute the activation rates (frequency of activation) of each neuron in each linear layer
def get_activation_rates(model, val_data):
    layers_dim = get_model_dim(model)
    activation_count = [torch.as_tensor([0] * width, dtype=torch.float32) for width in layers_dim]
    
    # Iterate through each data sample in the validation data
    for line_idx in range(val_data.shape[0]):
        activations = model.extended_relu_forward(torch.as_tensor(val_data.iloc[line_idx], dtype=torch.float32))
        
        # Count activated neurons (binary step function)
        for i in range(len(activation_count)):
            activation_count[i] += step_function(activations[i])
    
    # Normalize by number of samples to get activation rate
    for i in range(len(activation_count)):
        activation_count[i] /= val_data.shape[0]

    return activation_count

def get_model_dim(model):
    layers_width = []
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            layers_width.append(model.layers[i].out_features)
    return layers_width

def step_function(x):
    out = x > 0
    return out.int()

# Class for compressing a PyTorch model using skip connections
class model_compression(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = copy.deepcopy(model)
        self.layers = self.model.layers  # List of model layers
        self.skip_connection = dict()  # Stores skip connections
        self.act_rate = None  # To cache activation rates
 
        self.indices_linear_layers = [idx for idx, x in enumerate(self.model.layers) if isinstance(x, torch.nn.modules.linear.Linear)]

    # Apply compression by identifying and skipping neurons with high linearity
    def compression(self, threshold, val_data, layer_threshold=0):
        if self.act_rate is None:
            self.act_rate = get_activation_rates(self.model, val_data)

        # Get indices of neurons above threshold in each layer
        linear_neurons = [torch.where(layer >= threshold)[0] for layer in self.act_rate]
        linear_neurons[-1] = torch.tensor([])  # Skip the final layer

        for idx, lin_neurons_layer in enumerate(linear_neurons):
            if self.layer_threshold_interpretation(len(lin_neurons_layer), idx, layer_threshold):
                self.get_skip_connection_weights(idx, lin_neurons_layer)
                self.remove_linear_neurons(idx, lin_neurons_layer)

    # Builds skip connection weights from one layer to the next, skipping linear neurons
    def get_skip_connection_weights(self, layer, neurons):
        layer_idx = self.indices_linear_layers[layer]
        next_layer_idx = self.indices_linear_layers[layer + 1]

        # Prepare input feature count for skip connection
        skip_con_in_features = self.layers[layer_idx].in_features
        if self.indices_linear_layers[layer - 1] in self.skip_connection:
            skip_con_in_features += self.skip_connection[self.indices_linear_layers[layer - 1]].in_features

        # Initialize new linear skip connection
        self.skip_connection[layer_idx] = nn.Linear(skip_con_in_features, self.layers[next_layer_idx].out_features, bias=False)
        self.skip_connection[layer_idx].weight.data.zero_()

        # Compute weights for skip connection
        for next_layer_neuron in range(self.layers[next_layer_idx].out_features):
            for neuron in neurons:
                next_layer_weight = self.layers[next_layer_idx].weight.data[next_layer_neuron][neuron]

                if self.indices_linear_layers[layer - 1] in self.skip_connection:
                    tmp_skip_contribution = torch.cat((
                        self.layers[layer_idx].weight.data[neuron],
                        self.skip_connection[self.indices_linear_layers[layer - 1]].weight.data[neuron]
                    ))
                else:
                    tmp_skip_contribution = self.layers[layer_idx].weight.data[neuron]

                self.skip_connection[layer_idx].weight.data[next_layer_neuron] += tmp_skip_contribution * next_layer_weight
                self.layers[next_layer_idx].bias.data[next_layer_neuron] += self.layers[layer_idx].bias.data[neuron] * next_layer_weight

    # Removes identified linear neurons from current and next layers
    def remove_linear_neurons(self, layer, neurons):
        layer_idx = self.indices_linear_layers[layer]
        next_layer_idx = self.indices_linear_layers[layer + 1]

        # Shrink current layer
        new_layer = nn.Linear(self.layers[layer_idx].in_features, self.layers[layer_idx].out_features - len(neurons))
        if new_layer.out_features:
            new_layer.weight.data = torch.stack(
                [weights for idx, weights in enumerate(self.layers[layer_idx].weight.data) if idx not in neurons], dim=0
            )
            new_layer.bias.data = torch.stack(
                [bias for idx, bias in enumerate(self.layers[layer_idx].bias.data) if idx not in neurons], dim=0
            )
        self.layers[layer_idx] = new_layer

        # Adjust next layer inputs
        new_next_layer = nn.Linear(
            self.layers[next_layer_idx].in_features - len(neurons),
            self.layers[next_layer_idx].out_features
        )
        if new_next_layer.in_features:
            new_next_layer.weight.data = torch.stack([
                torch.stack([neuron for idx, neuron in enumerate(layer) if idx not in neurons], dim=0)
                for layer in self.layers[next_layer_idx].weight.data
            ], dim=0)
            new_next_layer.bias.data = self.layers[next_layer_idx].bias.data
        self.layers[next_layer_idx] = new_next_layer

        # Update skip connections
        if self.indices_linear_layers[layer - 1] in self.skip_connection:
            new_skip_con_layer = nn.Linear(
                self.skip_connection[self.indices_linear_layers[layer - 1]].in_features,
                self.skip_connection[self.indices_linear_layers[layer - 1]].out_features - len(neurons),
                bias=False
            )
            if new_skip_con_layer.out_features:
                new_skip_con_layer.weight.data = torch.stack([
                    weights for idx, weights in enumerate(
                        self.skip_connection[self.indices_linear_layers[layer - 1]].weight.data
                    ) if idx not in neurons
                ], dim=0)
            self.skip_connection[self.indices_linear_layers[layer - 1]] = new_skip_con_layer

    # Sequentially apply compression layer-by-layer with heuristic-based selection
    def seq_compression(self, threshold, val_data):
        self.act_rate = get_activation_rates(self.model, val_data)
        linear_neurons = [torch.where(layer >= threshold)[0] for layer in self.act_rate]
        linear_neurons[-1] = torch.tensor([])
        original_layers = copy.deepcopy(self.layers)
        skip_ideces = []
        improvements = [self.skip_improvement(len(lin_neu), idx) for idx, lin_neu in enumerate(linear_neurons)]

        while any([imp > 0 for imp in improvements]):
            max_impro = torch.argmax(torch.as_tensor(improvements))
            skip_ideces.append(max_impro)
            self.layers = copy.deepcopy(original_layers)
            self.skip_connection = {}
            skip_ideces.sort()

            for i in skip_ideces:
                self.get_skip_connection_weights(i, linear_neurons[i])
                self.remove_linear_neurons(i, linear_neurons[i])

            improvements = [
                0 if idx in skip_ideces else self.skip_improvement(len(lin_neu), idx)
                for idx, lin_neu in enumerate(linear_neurons)
            ]

    # Check if layer passes the linearity threshold (supports int, float, str logic)
    def layer_threshold_interpretation(self, num_lin_neurons_layer, lin_layer_idx, layer_threshold=0):
        if isinstance(layer_threshold, int):
            return num_lin_neurons_layer > layer_threshold
        elif isinstance(layer_threshold, float):
            layer_idx = self.indices_linear_layers[lin_layer_idx]
            return (num_lin_neurons_layer / self.layers[layer_idx].out_features) > layer_threshold
        elif isinstance(layer_threshold, str):
            return self.skip_improvement(num_lin_neurons_layer, lin_layer_idx) > 0
        else:
            raise Exception("No allowed type for the layer threshold")

    # Estimate benefit of applying skip connection
    def skip_improvement(self, num_lin_neurons_layer, lin_layer_idx):
        if not num_lin_neurons_layer:
            return False
        layer_idx = self.indices_linear_layers[lin_layer_idx]
        next_layer_idx = self.indices_linear_layers[lin_layer_idx + 1]
        wid_lay_prev = self.layers[layer_idx].in_features

        if (layer_idx - 2) in self.skip_connection:
            wid_lay_prev += self.skip_connection[layer_idx - 2].out_features

        wid_lay_next = self.layers[next_layer_idx].out_features
        return (wid_lay_prev + wid_lay_next + 1) * num_lin_neurons_layer - (wid_lay_prev * wid_lay_next)

    # Forward pass with skip connections
    def forward(self, x):
          
        tmp_skip_output = None
        tmp_layers = []
        for idx, layer in enumerate(self.layers):
            if (not (tmp_skip_output is None)) and isinstance(layer, torch.nn.modules.linear.Linear):
                x_tmp = layer(x) + tmp_skip_output
                if idx in self.skip_connection.keys():    
                    tmp_skip_output = self.skip_connection[idx](torch.cat((x, torch.cat(tmp_layers, dim = tmp_layers[0].dim() - 1)), dim = tmp_layers[0].dim() - 1))
                    tmp_layers.insert(0, x)
                else:
                    tmp_skip_output = None
                    tmp_layers = []

                x = x_tmp
            else:
                if idx in self.skip_connection.keys():
                    tmp_skip_output = self.skip_connection[idx](x)
                    tmp_layers.append(x)
                x = layer(x)

        return x

    # Count the number of weights in model and skip connections
    def get_num_weights(self, seperate=False, without_in=False):
        layer_count = 0
        for idx, layer in enumerate(self.layers):
            if idx == 0 and without_in:
                continue
            if isinstance(layer, torch.nn.modules.linear.Linear):
                layer_count += layer.weight.data.numel() + layer.bias.data.numel()
        skip_count = sum(skip.weight.data.numel() for skip in self.skip_connection.values())
        return (layer_count, skip_count) if seperate else (layer_count + skip_count)

    # Automatically find compression level to reach target model size (based on rate)
    def seq_fixed_size_compression(self, val_data, rate=0.5, threshold_stepsize=0.05):
        aim = rate * self.get_num_weights(without_in=True)
        original_layers = copy.deepcopy(self.layers)
        activation_threshold = 1
        while aim < self.get_num_weights(without_in=True):
            activation_threshold -= threshold_stepsize
            self.layers = copy.deepcopy(original_layers)
            self.skip_connection = {}
            self.compression(activation_threshold, val_data, "best")