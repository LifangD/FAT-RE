import torch.nn as nn
import torch
from model.module.multi_head import MultiHeadedAttention
from model.module.feed_forward import Positionwise_Conv_Forward,Positionwise_Conv_Forward_mul,PositionwiseFeedForward
from utils.norm import LayerNorm
from model.module.highway_layer import Highway
from model.module.sublayer import Residual
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def  __init__(self, opt,hidden, attn_heads, feed_forward_hidden, dropout,width):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(attn_heads,hidden,dropout)

        if opt['ffn'] =="ffn":
            self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden,dropout=dropout)
        elif opt['ffn'] == "ffn_conv":
            self.feed_forward = Positionwise_Conv_Forward(d_model=hidden,d_ff=feed_forward_hidden,width=width,dropout = dropout)

        elif opt['ffn'] == "ffn_conv_m":
            self.feed_forward = Positionwise_Conv_Forward_mul(d_model=hidden, d_ff=feed_forward_hidden, width=width,dropout=dropout)

        if opt['connect'] == "residual":
            self.input_connect = Residual()
            self.output_connect = Residual()
        elif opt['connect'] == "highway":
            self.input_connect = Highway (input_size=hidden)
            self.output_connect = Highway (input_size=hidden)

        self.norm = LayerNorm(hidden)

    def get_weight(self):
        att_weight = self.attention.get_weight()
        # ffn_weight = self.feed_forward.get_weight()
        # weights= att_weight+ffn_weight
        return att_weight


    def forward(self, x, mask,adj,layer):

        att_Hx,attn = self.attention(x, x, x, mask,adj,layer)
        att_Hx = self.norm(att_Hx)
        sub1_connect= self.input_connect(x,att_Hx)


        ffn_Hx = self.feed_forward (sub1_connect)
        sub2_connect = self.output_connect(sub1_connect,ffn_Hx)
        sub2_connect = self.norm(sub2_connect)


        return sub2_connect,attn






