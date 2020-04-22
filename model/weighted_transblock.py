
from utils.norm import LayerNorm


import torch
import torch.nn as nn
import torch.nn.init as init
from model.module.multi_head import MultiHeadedAttention
from model.module.feed_forward import Positionwise_Conv_Forward,Positionwise_Conv_Forward_mul,PositionwiseFeedForward
from utils.Linear import Linear
from model.module.single import Scale_dot_Attention
from model.module.highway_layer import Highway
class Weighted_TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def  __init__(self, opt,hidden, n_branches, feed_forward_hidden, dropout,layer):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_v = hidden // n_branches
        self.n_branches = n_branches

        self.attention = MultiHeadedAttention (n_branches, hidden, dropout, opt['trans_layers'])

        # additional parameters for BranchedAttention
        self.w_o = nn.ModuleList([Linear(self.d_v, hidden) for _ in range(n_branches)])
        self.w_kp = torch.rand(n_branches)
        self.w_kp = nn.Parameter(self.w_kp/self.w_kp.sum())
        self.w_a = torch.rand(n_branches)
        self.w_a = nn.Parameter(self.w_a/self.w_a.sum())

        self.pos_ffn = nn.ModuleList([Positionwise_Conv_Forward(hidden, feed_forward_hidden//n_branches, 5,dropout) for _ in range(n_branches)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden)
        self.input_connect = Highway (input_size=hidden)
        self.output_connect = Highway (input_size=hidden)



    def forward(self, x, mask,adj,layer):
        context, attn = self.attention(x, x, x, mask, adj,layer)
        att_Hx = self.layer_norm (context)
        res_0= self.input_connect (x, att_Hx)
        # context: a list of tensors of shape [b_size x len_q x d_v] len: n_branches
        context = res_0.split (self.d_v, dim=-1)

        # outputs: a list of tensors of shape [b_size x len_q x d_model] len: n_branches
        outputs = [self.w_o[i] (context[i]) for i in range (self.n_branches)]
        outputs = [kappa * output for kappa, output in zip (self.w_kp, outputs)]
        outputs = [pos_ffn (output) for pos_ffn, output in zip (self.pos_ffn, outputs)]
        outputs = [alpha * output for alpha, output in zip (self.w_a, outputs)]

        # output: [b_size x len_q x d_model]
        res_1 = self.dropout (torch.stack (outputs).sum (dim=0))
        out = self.output_connect (res_0, res_1)

        return out,attn











