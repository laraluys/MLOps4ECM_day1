import torch.nn as nn

class NN_model(nn.Module):
    def __init__(self, model_params):
        
        super(NN_model, self).__init__()
        self.seq_len = 9 # 9 input values in dataset
        self.criterion = nn.BCELoss()
        self.module_list = nn.ModuleList()
        self.dimensions = model_params["dimensions"]
        self.n_layers = model_params["n_layers"]

        layer1 = nn.Linear(self.seq_len,self.dimensions[0], bias=True)
        self.module_list.append(layer1)
        self.module_list.append(nn.ReLU())

        for i in range(0, self.n_layers - 1):
            layeri = nn.Linear(self.dimensions[i],self.dimensions[i+1], bias=True)
            self.module_list.append(layeri)
            self.module_list.append(nn.ReLU())

        layern = nn.Linear(self.dimensions[-1], 1, bias=True)
        self.module_list.append(layern)
        self.module_list.append(nn.Sigmoid()) # value has to be between 0 and 1
        

    def forward(self, input_tensor):
        x = self.module_list[0](input_tensor)
        for i in range(1, len(self.module_list)):
            x = self.module_list[i](x)
        return x
