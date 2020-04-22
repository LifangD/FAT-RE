import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy

class Scale_dot_Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self,dropout,d_k):
        super(Scale_dot_Attention,self).__init__()
        self.dropout = nn.Dropout (dropout)


        self.LN = nn.Linear(2*d_k,d_k)
    def forward(self, query, key, value, mask,adj,layer):
        scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(query.size(-1))
        #scores = torch.max(scores,-1,keepdim=True)[0].expand_as(scores)-scores
        mask = mask.unsqueeze (1).repeat (1, query.size (1), 1, 1)
        if adj is not None:
            adj = adj.unsqueeze(1).repeat(1,query.size(1),1,1)
            padding = nn.ConstantPad2d ((0, 1, 0, 1), 1)
            adj = padding (adj)
            adj_mask = adj>0
            mask = mask*adj_mask
        # if layer+1 > int(self.trans_layers/2):
        #     scores = self.activation(scores)
        #     scores = torch.max(scores,-1,keepdim=True)[0].expand_as(scores)-scores

        # mask_l = batch_tril(mask,layer)
        # mask_u = batch_triu(mask,layer)
        # scores_l = scores.masked_fill(mask_l == 0, -1e9)
        # l_attn = self.dropout(F.softmax(scores_l, dim=-1))
        # scores_u = scores.masked_fill (mask_u == 0, -1e9)
        # u_attn = self.dropout(F.softmax (scores_u, dim=-1))
        # res = self.LN(torch.cat([torch.matmul(l_attn,value),torch.matmul(u_attn,value)],-1))
        # p_attn = l_attn+u_attn

        scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        res = torch.matmul(p_attn, value)
        return res, p_attn

def batch_tril(A,layer):
    B = A.clone()
    if layer ==0:
        k =5
    else: k=1
    ii,jj = numpy.triu_indices(B.size(-2), k, m=B.size(-1))
    B[...,ii,jj] = 0
    return B

def batch_triu(A,layer):
    B = A.clone()
    if layer ==0:
        k =5
    else: k=1
    ii,jj = numpy.triu_indices(B.size(-2), k, m=B.size(-1))
    B[...,ii,jj] = 0
    return B

class Scale_dot_Attention_v2(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self,dropout):
        super(Scale_dot_Attention_v2,self).__init__()
        self.dropout = nn.Dropout (dropout)
    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(query.size(-1))

        #mask = mask.unsqueeze (1).repeat (1, query.size (1), 1, 1)
        # if adj is not None:
        #     adj = adj.unsqueeze(1).repeat(1,query.size(1),1,1)
        #     padding = nn.ConstantPad2d ((0, 1, 0, 1), 1)
        #     adj = padding (adj)
        #     adj_mask = adj>0
        #     mask = mask*adj_mask
        #scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        #p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class Guassian_Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self,dropout,hidden):
        super(Guassian_Attention,self).__init__()
        self.dropout = nn.Dropout (dropout)
        layers = [nn.Linear (hidden, hidden), nn.Tanh (), nn.Linear (hidden, 1)]
        self.predict = nn.Sequential(*layers)
        self.D = 6

    def forward(self, query, key, value, mask,adj):
        scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(query.size(-1))
        length = query.size(2)
        Pi = F.sigmoid(self.predict(query))*length
        Pi = Pi.repeat(1,1,1,length)
        J = torch.FloatTensor(list(range(length))).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(query.size(0),query.size(1),length,1).cuda()
        G =-torch.pow(J-Pi,2)/math.pow(self.D/2,2)
        scores+=G

        mask = mask.unsqueeze (1).repeat (1, query.size (1), 1, 1)
        if adj is not None:
            adj = adj.unsqueeze(1).repeat(1,query.size(1),1,1)
            padding = nn.ConstantPad2d((0, 1, 0, 1),1)
            adj = padding(adj)
            adj_mask = adj>0
            mask = mask*adj_mask
        scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class Relative_Attention_inner(nn.Module):
    """
    Compute the Scaled Dot Product Attention with position-aware
    referenced on: https://github.com/tensorflow/tensor2tensor/af42d543c2f24a0143b2483db93ac931c54146b9/tensor2tensor/layers/common_attention.py
    to calculate with plus key or value, things like xy+xz
    """

    def forward(self, x, y, z, adj, transpose):
        """

        :param x:Tensor with shape [batch_size, heads, length, length or depth].
        :param y:Tensor with shape [batch_size, heads, length, depth].
        :param z:Tensor with shape [length, length, depth].
        #:param adj: the adjacency matrix
        :return: A Tensor with shape [batch_size, heads, length, length or depth].

        # different from the tensorflow, the matmul operation didn't have transpose
        """
        batch_size = x.size (0)
        heads = x.size (1)
        length = x.size (2)

        # xy_matmul is [batch_size, heads, length, length or depth]

        if transpose:
            xy_matmul = torch.matmul (x, y.transpose (-2, -1))
        else:
            xy_matmul = torch.matmul (x, y)
        # x_t is [length, batch_size, heads, length or depth]
        x_t = x.permute (2, 0, 1, 3)
        # x_t_r is [length, batch_size * heads, length or depth]
        x_t_r = x_t.contiguous ().view (length, heads * batch_size, -1)
        # x_tz_matmul is [length, batch_size * heads, length or depth]

        # for the pos relative embedding
        if transpose:
            x_tz_matmul = torch.matmul (x_t_r, z.transpose(-2,-1))
        else:
            x_tz_matmul = torch.matmul (x_t_r, z)
        # x_tz_matmul_r is [length, batch_size, heads, length or depth]
        x_tz_matmul_r =  x_tz_matmul.contiguous().view(length, batch_size, heads, -1)
        # x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
        x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)

        return xy_matmul + x_tz_matmul_r_t











