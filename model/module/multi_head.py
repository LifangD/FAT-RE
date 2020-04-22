import torch.nn as nn
from model.module.single import  Scale_dot_Attention,Guassian_Attention,Scale_dot_Attention_v2
import torch
import torch.nn.init as init
from utils.Linear import Linear
from utils.norm import GroupBatchnorm2d
from model.module.caps import CapsuleLayer
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout):
        super().__init__()
        #assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h # number of heads.


        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Scale_dot_Attention(dropout,self.d_k)
        self.dropout_head = nn.Dropout2d(p=0.3) #head-mask
        #self.dropout_head = nn.Dropout(p=0.5)
        self.BN = nn.BatchNorm2d(self.h)
    def get_weight(self):
        weights = []
        for lay in self.linear_layers:
            weights +=[lay.weight,lay.bias]
        weights +=[self.output_linear.weight,self.output_linear.bias]
        return weights

    def forward(self, query, key, value, mask,adj,layer):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # multi-head number
        # query = query.unsqueeze(1).repeat(1,self.h,1,1) ## use the original copy mechanism, not the split or the linear layer;  and note the d_k value changed
        # key = key.unsqueeze(1).repeat(1,self.h,1,1)
        # value = value.unsqueeze(1).repeat(1,self.h,1,1)

        query, key, value = [l(x).contiguous().view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                              for l, x in zip(self.linear_layers, (query, key, value))]

        #query,key,value = [l(x) for l,x in zip(self.linear_layers,(query,key,value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask,adj,layer)

        # 3) "Concat" using a view and apply a final linear.
        x = self.dropout_head(x)
        #x = self.BN(x)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        out = self.output_linear(x)
        # if layer == self.trans_layers-1:
        #     out = self.dropout_head (out)
        # overfitting happens here!; wish the first head capture the relation meaning

            # x = x[:,0,:,:]
            # out = self.output_linear2(x)

        return out,attn

class CapAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, opt):
        super().__init__()
        #assert d_model % h == 0
        self.d_model = opt['hidden_dim']
        # We assume d_v always equals d_k

        #self.d_k = d_model
        self.h = opt['multi_heads'] # number of heads.
        self.d_k = self.d_model // self.h

        self.linear_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(3)])
        self.output_linear = nn.Linear(self.d_model, self.d_model)
        self.attention = Scale_dot_Attention(opt['attn_dropout'],self.d_k)
        #self.pri_cap = CapsuleLayer(num_capsules=8,num_route_nodes=-1,in_channels=self.d_model,out_channels=3,kernel_size=5,stride=1)
        self.digit_cap = CapsuleLayer(num_capsules=1,num_route_nodes=10,in_channels=self.d_k,out_channels=self.d_model)

    def get_weight(self):
        weights = []
        for lay in self.linear_layers:
            weights +=[lay.weight,lay.bias]
        weights +=[self.output_linear.weight,self.output_linear.bias]
        return weights

    def forward(self, query, key, value, mask,adj,layer):
        batch_size = query.size(0)
        query, key, value = [l(x).contiguous().view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                              for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask,adj,layer)
        # x : n*l*h*v
        x = x.transpose (1, 2).contiguous ().view(-1,self.h,self.d_k)
        x = self.digit_cap(x).view(batch_size,-1,self.d_model)

        return x


class Dual_Attention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout,trans_layers):
        super().__init__()
        #assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        #self.d_k = d_model
        self.h = h # number of heads.
        self.trans_layers = trans_layers

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        #self.output_linear2 = nn.Linear(self.d_k,d_model)
        self.sem_attention = Scale_dot_Attention(dropout)
        self.channel_attention = Scale_dot_Attention_v2(dropout)
        #self.attention = Guassian_Attention(dropout,self.d_k)
        self.dropout_head = nn.Dropout2d(p=0.1)
        self.alpha = self.w_kp = nn.Parameter(torch.Tensor([0]))
        self.beta= self.w_kp = nn.Parameter (torch.Tensor([0]))
    def get_weight(self):
        weights = []
        for lay in self.linear_layers:
            weights +=[lay.weight,lay.bias]
        weights +=[self.output_linear.weight,self.output_linear.bias]
        return weights

    def forward(self, query, key, value, mask,adj):
        batch_size = query.size(0)

        query, key, value = [l(x).contiguous().view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                               for l, x in zip(self.linear_layers, (query, key, value))]

        # x, attn = self.sem_attention(query, key, value,mask,adj)
        # sem_res = self.alpha*x

        q,k,v = [x.contiguous().view(batch_size,self.h,-1).transpose(1,2) for x in (query,key,value)]
        x2,attn = self.channel_attention(q,k,v)
        channel_res = self.beta*x2.view(batch_size,self.h,-1,self.d_k)
        x= value+channel_res

        #x = self.dropout_head(x)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        out = self.output_linear(x)
        return out,attn




