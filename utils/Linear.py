import  torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)