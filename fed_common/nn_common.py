import numpy as np
import math
import torch
import torch.nn as nn


class model_common(nn.Module):
    def __init__(self, dataset, model_choice, ci = None):
        super(model_common, self).__init__()
        self.dataset = dataset
        self.ci = ci
        self.model_choice = model_choice
        
        if self.model_choice == "CNN":    
            self.conv_layers, self.lin_layers = self.get_model(dataset)
        else:
            self.lin_layers = self.get_model(dataset)
        
    def get_model(self, dataset):
        if dataset == "banana":
            return nn.Sequential(
               nn.Linear(2,4),
               nn.Linear(2,4) 
            )
        elif dataset == "MNIST" or dataset == "fashion_MNIST":
            if self.model_choice == "FNN": 
                return nn.Sequential(            
                    nn.Linear(28*28,100),
                    nn.ReLU(),
                    nn.Linear(100,10)
                )
            elif self.model_choice == "CNN" : 
                convLayers = nn.Sequential(
                    nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                linLayers = nn.Sequential(
                    nn.Linear(196, 10)
                    )
                return convLayers, linLayers
        elif dataset == "MNIST_2class":
            if self.model_choice == "FNN":
                return nn.Sequential(
                    nn.Linear(28*28, 100),
                    nn.ReLU(),
                    nn.Linear(100,2) 
                )
            elif self.model_choice == "CNN":
                convLayers = nn.Sequential(
                    nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                linLayers = nn.Sequential(
                    nn.Linear(196, 2)
                    )
                return convLayers, linLayers
        elif dataset == "MNIST_4class" :
            if self.model_choice == "FNN":
                return nn.Sequential(
                    nn.Linear(28*28, 100),
                    nn.ReLU(),
                    nn.Linear(100,4)    
                )
            elif self.model_choice == "CNN":
                convLayers = nn.Sequential(
                    nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                linLayers = nn.Sequential(
                    nn.Linear(196, 4)
                    )
                return convLayers, linLayers
        elif dataset == "A2_PCA" or dataset == "3node" : 
            if self.model_choice == "FNN": 
                return nn.Sequential(
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100,2)    
                )
            elif self.model_choice == "CNN":
                convLayers = nn.Sequential(
                    nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                linLayers = nn.Sequential(nn.Linear(25,2))
                return(convLayers, linLayers)
            else :
                raise ValueError("no known model selection supplied")
        elif dataset == "kinase":
            if self.model_choice == "FNN":
                return nn.Sequential(
                    nn.Linear(8192, 100),
                    nn.ReLU(),
                    nn.Linear(100,2))
        elif dataset == "kinase_PCA":
            if self.model_choice == "FNN":
                return nn.Sequential(
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100,2))
        elif dataset == "kinase_ABL1" or dataset == "kinase_KDR":
            if self.model_choice == "FNN":
                return nn.Sequential(
                    nn.Linear(8192, 1000),
                    nn.ReLU(),
                    nn.Linear(1000,2)) 
        else:
            raise ValueError("no known dataset supplied")

        #forward pass through the net
    def forward(self, input):
        #print(input)
        #print(input.shape)

        
        if self.model_choice == "CNN":
            input = self.conv_layers(input)
            input = input.view(input.shape[0], -1)
        return self.lin_layers(input)
    
    def train(self):
        raise(NotImplementedError("implement training function"))

    def test(self, X_test, y_test):
        correct = 0
        with torch.no_grad():
            #for (x, y) in zip(X_test, y_test):
            output = self.forward(X_test)
            #loss = criterion(output, y)
            # for now, only look at accuracy, using criterion we can expand this later on 
            _, prediction = torch.max(output.data, 1)
            correct += (prediction == y_test).sum().item()
        # return accuracy
        return (correct / X_test.size()[0])

    def set_params(self, params):
        self.load_state_dict(params)
        #for model_layer, param_layer in zip (self.layers, params):
        #    model_layer.load_state_dict(param_layer, strict=True)

    def get_params(self):
        return self.state_dict()
        #parameters = []
        #for layer in self.layers:
        #    parameters.append(layer.state_dict())
        #return parameters