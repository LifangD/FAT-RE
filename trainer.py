"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.reclassfier import TransClassifier
from utils import constant, torch_utils
from utils.loss import FocalLoss,multiClassHingeLoss
import math
import utils.constant
class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = []
        for b in batch[:12]:
            if b is None:
                inputs+=[b]
            else:
                inputs+= [Variable(b.cuda())]
        labels = Variable(batch[12].cuda())

    else:
        inputs = [Variable(b) for b in batch[:12]]
        labels = Variable(batch[12])

    return inputs, labels


class TransTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = TransClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'],opt['l2'])

    def update(self, batch,):
        inputs, labels= unpack_batch(batch, self.opt['cuda'])
        self.model.train()
        self.optimizer.zero_grad()
        logits,weights= self.model(inputs)
        loss = self.criterion(logits, labels)
        loss_val = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val,weights

    def predict(self, batch, unsort=True):
        inputs, labels= unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[13]
        self.model.eval()
        logits,weights= self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort: # if use rnn=>sort sentence length to sort=>also sort to predit
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,predictions, probs)))]
        return predictions,probs, loss.item(),weights

