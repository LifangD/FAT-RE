import torch.nn as nn
import torch
from torch.autograd import Variable
from model.transblock import TransformerBlock
from model.weighted_transblock import Weighted_TransformerBlock
from utils import constant
from model.module.multi_head import Dual_Attention
from utils.tree import *
from utils.position import PositionalEmbedding
from model.module.feed_forward import FFN
from utils.norm import LayerNorm
from model.module.feed_forward import Positionwise_Conv_Forward_mul
from model.module.multi_head import CapAttention
from model.bert.modeling import BertEmbeddings,BertConfig

class Trans(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, embeddings, opt,):

        super().__init__()
        self.opt = opt
        self.n_layers = opt['trans_layers']
        self.attn_heads = opt['multi_heads']
        self.ffn_ex_size =opt['ffn_ex_size']



        self.emb, self.pos_emb, self.ner_emb,self.pos1_emb,self.pos2_emb = embeddings
        self.in_dim = opt['emb_dim'] +opt['pos_dim']+ self.opt['ner_dim']

        self.transformer_blocks = nn.ModuleList ([TransformerBlock (opt, opt['hidden_dim'],self.attn_heads,
                                                                     opt['hidden_dim'] * self.ffn_ex_size,
                                                                self.opt['attn_dropout'],width=3) # width [1,3,5]
                                                                 for layer in range (self.n_layers)])


        if self.opt['rnn']:
            self.rnn_hidden = int((self.opt['hidden_dim']-opt['relative_pos_dim']*2)/2)
            self.rnn_layers = 1
            self.rnn = nn.LSTM (self.in_dim, self.rnn_hidden, self.rnn_layers, batch_first=True,
                                dropout=self.opt['compress_dropout'], bidirectional=True,)

            self.rnn_drop = nn.Dropout (self.opt['compress_dropout'])  # use on last layer output
            self.linear = nn.Linear (self.in_dim, opt['hidden_dim'])
        else:
            self.in_dim+=self.opt['relative_pos_dim']*2
            self.fc_hidden = int(self.opt['hidden_dim'] )
            self.fc = nn.Linear (self.in_dim,self.fc_hidden)
            self.fc_drop = nn.Dropout (self.opt['compress_dropout'])

        rel_linear = [nn.Linear (self.opt['hidden_dim']* 2, self.opt['hidden_dim']), nn.Sigmoid(), nn.Dropout (p=0.3)]
        self.to_rel = nn.Sequential (*rel_linear)
        self.in_drop = nn.Dropout(opt['compress_dropout'])
        #
        #
        self.sp = nn.Parameter(torch.randn(self.opt['hidden_dim']))
        self.rel_GRU = nn.GRU (self.opt['hidden_dim'], self.opt['hidden_dim'], num_layers=2, batch_first=True, dropout=0.1)
        self.GRU = nn.GRU(self.opt['hidden_dim'],self.opt['hidden_dim'],num_layers=2,batch_first=True,dropout=0.3)
        #

        self.norm = LayerNorm (opt['hidden_dim'])
        out_dim = opt['hidden_dim']
        layers = [nn.Linear (out_dim, opt['hidden_dim']), nn.ReLU ()]
        for _ in range (self.opt['mlp_layers'] - 1):
            layers += [nn.Linear (opt['hidden_dim'], opt['hidden_dim']), nn.ReLU ()]
        self.out_mlp = nn.Sequential (*layers)
        self.dropout = nn.Dropout(p=0.1)

        if opt["use_bert_embedding"]:
            bert_config = BertConfig.from_json_file(opt["pretrained_model"]+"/bert_config.json")
            self.bert_embedding= BertEmbeddings(bert_config)

            ckpt = torch.load(opt["pretrained_model"]+"/pytorch_model.bin")

            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            for key in list(state_dict.keys()):
                if 'embedding' in key:
                    new_key = key.replace("bert.embeddings.","") # delete 'bert.'
                    state_dict[new_key] = state_dict.pop(key)
            try:
                self.bert_embedding.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print(e)
            self.reduce_bert=nn.Linear(bert_config.hidden_size,opt["emb_dim"],bias=False)
            nn.init.xavier_normal_(self.reduce_bert.weight,gain=1.414)
            self.emb = nn.Sequential(self.bert_embedding,self.reduce_bert)








    def forward(self, inputs):
        words, mask_vec, pos, ner, deprel, head, subj_pos, obj_pos,subj_pos2, obj_pos2,subj_type, obj_type = inputs  # unpack
        x = words
        word_embs = self.emb(x)


        embs = [word_embs]
        embs += [self.pos_emb (pos)]
        embs += [self.ner_emb (ner)]

        if self.opt['rnn']:
            embs = self.in_drop (torch.cat (embs, dim=2))
            x0 = self.linear(embs)
            zero_padding = torch.zeros([x0.size(0),1,x0.size(2)]).cuda()
            x0 = torch.cat([x0,zero_padding],1)
            embs =self.encode_with_rnn(embs,mask_vec,words.size(0))
            #adj = inputs_to_tree_reps (head.data, words.data, l, self.opt['prune_k'], subj_pos2.data, obj_pos2.data)

            embs =  self.rnn_drop(torch.cat ([embs, self.pos1_emb (subj_pos), self.pos2_emb (obj_pos)], dim=2))
        else:
            embs+=[self.pos1_emb (subj_pos)]
            embs+=[self.pos2_emb (obj_pos)]
            embs = self.in_drop (torch.cat (embs, dim=2))
            embs = self.fc_drop(self.fc(embs))



        adj = None
        ep = self.opt['relative_pos_size']
        subj_mask, obj_mask = subj_pos.eq (ep).eq (0).unsqueeze (2), obj_pos.eq (ep).eq (0).unsqueeze (2)  # invert mask
        pool_type = self.opt['pooling']
        subj_out =pool (embs, subj_mask, type=pool_type)  # （b,h）
        obj_out = pool (embs, obj_mask, type=pool_type)  # （b,h）
        if self.opt['initial'] == "fc":
            entity = torch.cat ([subj_out, obj_out], dim=-1)
            rel = self.to_rel (entity).unsqueeze (1)
        elif self.opt['initial'] =="gru":
            sp = self.sp.expand (embs.size (0), self.opt['hidden_dim']).unsqueeze (1)
            s = torch.cat ([subj_out.unsqueeze (1)], -1)
            o = torch.cat ([obj_out.unsqueeze (1)], -1)
            entity_pair = torch.cat ([s,sp,o], dim=1)
            gru_out, gru_h = self.rel_GRU (entity_pair)
            rel = gru_h[-1,:,:].unsqueeze(1)
        else:
            rel = None

        rel_mask = torch.ones (mask_vec.size (0), 1) == 1
        mask_vec = torch.cat ([mask_vec,rel_mask.cuda ()], -1)
        mask = mask_vec.unsqueeze (1).repeat (1, x.size (1)+1, 1)

        batch_size = embs.size(0)
        x = torch.cat([embs,rel],1)


        pool_mask = None
        attn_list = []
        weights = []

        for i, transformer in enumerate(self.transformer_blocks):
            x,attn = transformer.forward(x, mask,adj,i)
            attn_list.append(attn)
        rel_hidden = x[:,-1,:]




        use_pooling=False
        if use_pooling:
            ep = self.opt['relative_pos_size']
            subj_mask, obj_mask = subj_pos.eq(ep).eq(0).unsqueeze(2), obj_pos.eq(ep).eq(0).unsqueeze(2)  # invert mask
            pool_type = self.opt['pooling']
            subj = pool(x[:,:-1,:],subj_mask,pool_type)
            obj = pool(x[:,:-1,:],obj_mask,pool_type)
            mask = mask_vec.eq (0).unsqueeze (2)
            rel_hidden = pool (x, mask, pool_type) # new rel hidden
            f_vec= self.out_mlp(torch.cat([subj,obj,rel_hidden],-1))
        else:
            f_vec = self.out_mlp(rel_hidden)
        triple = torch.cat([subj_out.unsqueeze(1), rel_hidden.unsqueeze(1), obj_out.unsqueeze(1)], 1)
        # original subj,rel and obj
        g_out, g_hidden = self.GRU(triple)
        none = g_hidden[-1, :, :]


        return f_vec,none,pool_mask,attn_list,weights,

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list (masks.data.eq (1).long ().sum (1).squeeze ())
        h0, c0 = rnn_zero_state (batch_size, self.rnn_hidden, self.rnn_layers)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence (rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn (rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence (rnn_outputs, batch_first=True)
        return rnn_outputs

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable (torch.zeros (*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda (), c0.cuda ()
    else:
        return h0, c0




def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)