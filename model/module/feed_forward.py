import torch.nn as nn
import torch
from utils.gelu import GELU

import torch.nn.functional as F
# class Positionwise_Conv_Forward(nn.Module):
#     "Implements FFN equation."
#
#     def __init__(self, d_model, d_ff, width_list,dropout):
#         super(Positionwise_Conv_Forward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#
#         self.conv= nn.ModuleList(nn.Conv2d(1, 1, kernel_size=(width,d_ff), padding=(int((width-1)/2),0)) for width in width_list)
#
#         self.w_2 = nn.Linear(len(self.conv), d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = GELU()
#
#     def get_weight(self):
#         weights = [self.w_1.weight, self.w_1.bias, self.w_2.weight, self.w_2.bias,self.conv.weight,self.conv.bias]
#         return weights
#
#     def forward(self, x):
#         x = self.w_1 (x)
#         x = x.unsqueeze (1)
#         conv_bits =[]
#         for conv in self.conv:
#             conv_bits.append(self.activation (conv(x)))
#         x = torch.cat(conv_bits,-1)
#         x = x.squeeze (1)
#         x = self.w_2 (self.dropout(x))
#         return x


class Positionwise_Conv_Forward (nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, width, dropout):
        super (Positionwise_Conv_Forward, self).__init__ ()
        self.w_1 = nn.Linear (d_model, d_ff)

        self.conv5 = nn.Conv2d (1, 1, kernel_size=(width, d_ff), padding=(int ((width - 1) / 2), 0))

        self.w_2 = nn.Linear (1, d_model)
        self.dropout = nn.Dropout (dropout)
        self.activation = nn.ReLU()

    def get_weight(self):
        weights = [self.w_1.weight, self.w_1.bias, self.w_2.weight, self.w_2.bias, self.conv.weight, self.conv.bias]
        return weights

    def forward(self, x):
        x = self.w_1 (x)
        x = x.unsqueeze (1)
        x = self.activation(self.conv5(x))
        x = x.squeeze (1)
        x = self.w_2 (self.dropout(x))
        return x
class Positionwise_Conv_Forward_mul(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, width, dropout):
        super(Positionwise_Conv_Forward_mul, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.conv = nn.Conv1d(d_ff, d_model, width, padding=int((width-1)/2))
        #self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.w_1 (x).transpose(1,2))
        x = self.dropout(self.conv(x))
        return x

    def get_weight(self):
        weights = [self.w_1.weight, self.w_1.bias, self.w_2.weight, self.w_2.bias,self.conv.weight,self.conv.bias]
        return weights



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
    def get_weight(self):

        weights = [self.w_1.weight,self.w_1.bias,self.w_2.weight,self.w_2.bias]
        return weights

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class FFN(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        return self.dropout(self.activation(self.w_1(x)))
