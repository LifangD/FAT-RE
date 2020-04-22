
import torch
import torch.nn as nn

from torch.autograd import Variable
import numpy as np
from utils.tree import head_to_tree, tree_to_adj
from model.trans import Trans
from utils import constant,torch_utils
from model.module.context_att import cal_context
from model.module.highway_layer import HighwayCNN
from utils.position import PositionalEmbedding
import torch.nn.functional as F
from model.module.caps import CapsuleLayer

class TransClassifier (nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super ().__init__ ()
        self.opt = opt
        self.model = TransRelationModel(opt, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(200,opt['num_class'])
        self.none_classifier = nn.Linear(200,1)


    def forward(self, inputs):
        vec,none,weights= self.model(inputs)

        rel_logit = self.classifier(vec)
        none_logit = self.none_classifier(none)
        logits = torch.cat([none_logit,rel_logit],-1) # artificial class

        return logits,weights



class TransRelationModel (nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super ().__init__ ()
        self.opt = opt
        self.emb_matrix = emb_matrix
        # create embedding layers
        self.emb = nn.Embedding (opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding (len (constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        # this is the part of speech embedding rather than the relative embedding
        self.ner_emb = nn.Embedding (len (constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.pos1_emb = nn.Embedding (2*opt['relative_pos_size']+2, opt['relative_pos_dim'])  # the distance to the entity1
        self.pos2_emb = nn.Embedding (2*opt['relative_pos_size']+2, opt['relative_pos_dim'])  # the distance to the entity2

        self.init_embeddings ()

        embeddings = (self.emb, self.pos_emb, self.ner_emb,self.pos1_emb,self.pos1_emb)

        self.transformer = Trans(embeddings, opt =self.opt)
        self.GRU = nn.GRU (self.opt['hidden_dim'], self.opt['hidden_dim'], num_layers=2, batch_first=True)




    def init_embeddings(self):

        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_ (-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy (self.emb_matrix)
            self.emb.weight.data.copy_ (self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print ("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print ("Finetune top {} word embeddings.".format (self.opt['topn']))
            self.emb.weight.register_hook (lambda x: torch_utils.keep_partial_grad (x, self.opt['topn']))
        else:
            print ("Finetune all embeddings.")

    def forward(self, inputs):
        f_vec,none,pool_mask,attn_list,weights,= self.transformer(inputs) #b*l*h
        return f_vec,none,attn_list

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