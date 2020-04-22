import torch.nn as nn
import torch

class cal_context(nn.Module):
    def __init__(self,da,d,dp):
        super ().__init__ ()
        self.h_linear = nn.Linear(d,da,bias=False)
        self.q_linear = nn.Linear(d,da,bias=False)
        self.pf1_linear = nn.Linear(dp,dp,bias=False)
        self.pf2_linear = nn.Linear(dp,dp,bias=False)
        self.context_linear = nn.Linear(da+da+dp+dp,1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self,h,q,pf1,pf2):
        encoder_out = torch.unbind (h, 1)
        pf1 = torch.unbind(pf1,1)
        pf2 = torch.unbind(pf2,1)
        values = []
        for state,p1,p2 in zip(encoder_out,pf1,pf2):
            Ui = self.dropout(self.activation(self.context_linear(torch.cat([self.h_linear(state),self.q_linear(q),self.pf1_linear(p1),self.pf2_linear(p2)],-1))))
            values.append (Ui)
        values = torch.stack (values).squeeze (-1)  # stack: [length,batch_size,1]
        values = self.softmax (values.transpose (0, 1))
        att_values = torch.unbind (values, 1)
        all = []
        for att, state in zip (att_values, encoder_out):
            att=att.unsqueeze(1)
            all.append (att * state)
        context_vec = torch.mean (torch.stack(all), 0)

        return context_vec
