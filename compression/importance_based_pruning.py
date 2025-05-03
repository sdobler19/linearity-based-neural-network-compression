## this is a simple implementation of the pruning described in: https://www.mdpi.com/1999-4893/17/1/48

import torch.nn as nn
import torch
import torch.optim as optim

def step_function(x):
    out = x > 0
    return out.int()

def batch_step_function(x):
    return torch.sum(step_function(x).float(), 0)
    

class Activation_based_pruning(nn.Module):
    def __init__(self, input_size, out_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_size),
            nn.Softmax()
        )

    def train(self, trainloader, epochs, val_loader = None, patients = 2):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.layers.parameters(), lr=0.005, momentum=0.9)

        max_val_acc = 0
        patients_count = 0
        is_looping = True
        for epoch in range(epochs):
            acc = 0
            count = 0
            for X, label in trainloader:
                y = nn.functional.one_hot(label, 10).float()
                y_pred = self.forward(X)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc += (torch.argmax(y_pred, 1) == label).float().sum()
                count += len(label)
            
            print(f"train_acc: {acc / count}")

            if val_loader:
                val_acc = 0
                val_count = 0
                for X, label in val_loader:
                    y_pred = self.forward(X)
                    val_acc += (torch.argmax(y_pred, 1) == label).float().sum()
                    val_count += len(label)
                print(f"epoch: {epoch} --train_acc: {acc/count} -- val_acc: {val_acc/val_count} -- stagnation: {patients_count}")
                if val_acc/val_count > max_val_acc:
                    max_val_acc = val_acc/val_count
                    patients_count = 0
                else:
                    patients_count += 1
                    if patients_count == patients:
                        is_looping = False
                        break

            if not is_looping:
                break

    def train_n_prune(self, trainloader, epochs, target_acc, pruning_perc, pruning_part = 0.05, val_loader = None):
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.layers.parameters(), lr=0.005, momentum=0.9)
        original_modelsize = self.get_modelsize()
        for epoch in range(epochs):
            acc = 0
            count = 0
            for X, label in trainloader:
                y = nn.functional.one_hot(label, 10).float()
                y_pred = self.forward(X)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc += (torch.argmax(y_pred, 1) == label).float().sum()
                count += len(label)
            
            if ((acc/count) > target_acc):
                if ((1 - (self.get_modelsize() / original_modelsize)) <= pruning_perc):
                # prune
                    act_rates = self.get_act_rates(trainloader)
                    threshold = torch.quantile(torch.cat(act_rates), pruning_part)
                    print(threshold)
                    prune_places = [torch.where(lay <= threshold)[0] for lay in act_rates]
                    # add history of removed neurons
                    for idx, neurons in enumerate(prune_places):
                        self.prune(idx, neurons)
                else:
                    break


            if val_loader:
                val_acc = 0
                val_count = 0
                for X, label in val_loader:
                    y_pred = self.forward(X)
                    val_acc += (torch.argmax(y_pred, 1) == label).float().sum()
                    val_count += len(label)
                yield (epoch, (acc/count).item(), (val_acc/val_count).item(), self.get_modelsize())
                print((epoch, (acc/count).item(), (val_acc/val_count).item(), self.get_modelsize()))
            else:
                yield (epoch, (acc/count).item(), self.get_modelsize())
                print((epoch, (acc/count).item(), self.get_modelsize()))
            


    def get_modelsize(self):
        layer_count = 0
        for layer in self.layers:
            if not isinstance(layer, nn.Linear):
                continue
            layer_count += layer.weight.data.numel() + layer.bias.data.numel()

        return layer_count

    def get_act_rates(self, x_loader):
        activation_count = []
        instance_count = 0
        for idx, (line, y) in enumerate(x_loader):
            activations = self.extended_relu_forward(line)
            for i in range(len(activations)):
                if idx == 0:
                    activation_count.append(batch_step_function(activations[i]))
                else:
                    activation_count[i] += batch_step_function(activations[i])
            instance_count += len(y) 

        for i in range(len(activation_count)):
            activation_count[i] /= instance_count
        return activation_count

    def prune(self, layer_idx, neuron_idx):
        linear_layers = [idx for idx, x in enumerate(self.layers) if isinstance(x, nn.Linear)]
        layer = linear_layers[layer_idx]
        if not (layer == max(linear_layers)):
            next_layer = linear_layers[layer_idx + 1]

        new_layer = nn.Linear(self.layers[layer].in_features, self.layers[layer].out_features - len(neuron_idx))
        
        if new_layer.out_features:
            new_layer.weight.data = torch.stack([weights for idx, weights in enumerate(self.layers[layer].weight.data) if not idx in neuron_idx], dim = 0)
            new_layer.bias.data = torch.stack([bias for idx, bias in enumerate(self.layers[layer].bias.data) if not idx in neuron_idx], dim = 0)
        self.layers[layer] = new_layer
        
        if not (layer == max(linear_layers)):
            new_next_layer = nn.Linear(self.layers[next_layer].in_features  - len(neuron_idx), self.layers[next_layer].out_features)
            if new_next_layer.in_features:
                new_next_layer.weight.data = torch.stack([torch.stack([neuron for idx, neuron in enumerate(layer) if not idx in neuron_idx], dim = 0) for layer in self.layers[next_layer].weight.data], dim = 0)
            new_next_layer.bias.data = self.layers[next_layer].bias.data
            self.layers[next_layer] = new_next_layer
        

    def extended_relu_forward(self, x):
        out = list()
        for layer in self.layers:
            x = layer(x)
            if not isinstance(layer, nn.Linear):
                out.append(x)
        return out

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x

