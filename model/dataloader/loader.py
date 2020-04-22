"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
from ..bert.tokenization import BertTokenizer
from utils import constant, helper, vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False,over_sampling=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.over_sampling = over_sampling
        if "processed_semeval" in opt['data_dir']:
            self.label2id = constant.sem_LABEL_TO_ID
            self.op_label2id = constant.op_sem_LABEL_TO_ID
        else:
            self.label2id = constant.LABEL_TO_ID
            self.op_label2id = constant.OP_LABEL_TO_ID
        self.evaluation = evaluation

        with open(filename) as infile:
            data = json.load(infile)

        self.raw_data = data
        if opt["use_bert_embedding"]:
            data=self.preprocess_under_bert(data,opt)
        else:
            data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        #self.op_id2label = dict ([(v, k) for k, v in self.op_label2id.items ()])
        #self.op_labels = [self.op_id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess_under_bert(self, data, opt,):
        """ Preprocess the data and convert to ids. """
        bert_tokenizer = BertTokenizer(vocab_file=opt["pretrained_model"]+"/vocab.txt",max_len=opt["max_length"])
        processed = []
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens] #

            wp_tokens = []
            pos_list = []
            ner_list = []
            for old_idx,t in enumerate(tokens):
                split_tokens = bert_tokenizer.wordpiece_tokenizer.tokenize(t)
                wp_tokens+=split_tokens
                if d['subj_start'] == old_idx:
                    new_subj_start = len(wp_tokens)-len(split_tokens)
                if d['subj_end']==old_idx:
                    new_subj_end = len(wp_tokens)-1
                if d['obj_start']==old_idx:
                    new_obj_start = len(wp_tokens)-len(split_tokens)
                if d['obj_end']==old_idx:
                    new_obj_end=len(wp_tokens)-1


                pos_list+=[d['stanford_pos'][old_idx]]*len(split_tokens)
                ner_list+=[d['stanford_ner'][old_idx]]*len(split_tokens)

            # anonymize tokens,refresh the pos
            ss, se = new_subj_start, new_subj_end
            os, oe = new_obj_start, new_obj_end
            tokens=wp_tokens

            tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)

            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID) #not use
            head = [int(x) for x in d['stanford_head']] # not use
            #assert any([x == 0 for x in head])

            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            tokens = bert_tokenizer.convert_tokens_to_ids(tokens)

            pos = map_to_ids (pos_list, constant.POS_TO_ID)
            ner = map_to_ids (ner_list, constant.NER_TO_ID)

            l = len(tokens)
            subj_positions2 = get_positions(d['subj_start'], d['subj_end'],l)
            obj_positions2 = get_positions(d['obj_start'], d['obj_end'],l)
            subj_positions = get_clip_positions(d['subj_start'], d['subj_end'], l,opt['relative_pos_size'])
            obj_positions = get_clip_positions(d['obj_start'], d['obj_end'],l, opt['relative_pos_size'])

            relation = self.label2id[d['relation']]


            processed += [(tokens, pos, ner, deprel,head,subj_positions,obj_positions,subj_positions2,obj_positions2,subj_type, obj_type,  relation)]


        return processed

    def preprocess(self, data, vocab, opt, ):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = list (d['token'])
            if opt['lower']:
                tokens = [t.lower () for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']]* (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + d['obj_type']]* (oe - os + 1)
            deprel = map_to_ids (d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int (x) for x in d['stanford_head']]
            assert any ([x == 0 for x in head])
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            tokens = map_to_ids (tokens, vocab.word2id)
            # tokens= map_to_ids_padding(tokens,vocab.word2id) # After padding stop words
            pos = map_to_ids (d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids (d['stanford_ner'], constant.NER_TO_ID)

            l = len (tokens)
            subj_positions2 = get_positions (d['subj_start'], d['subj_end'], l)
            obj_positions2 = get_positions (d['obj_start'], d['obj_end'], l)

            subj_positions = get_clip_positions(d['subj_start'], d['subj_end'], l, opt['relative_pos_size'])
            obj_positions = get_clip_positions(d['obj_start'], d['obj_end'], l, opt['relative_pos_size'])
            relation = self.label2id[d['relation']]




            elements = [tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_positions2,obj_positions2]
            res = []

            for element in elements:
                if ss > os:
                    element[os:oe + 1] = (element[os],)
                    off = oe - os
                    element[ss-off:se-off + 1] = (element[ss-off],)
                else:
                    element[ss:se + 1] = (element[ss],)
                    off = se - ss
                    element[os-off:oe-off + 1] = (element[os-off],)
                res.append(element)


            # s= np.array(elements[5])
            # o = np.array(elements[6])
            # ep = self.opt['relative_pos_size']
            # sub = np.where(s == ep)[0][0]
            # obj = np.where(o == ep)[0][0]
            #
            # left = min (sub, obj)
            # right= max (sub, obj)
            #
            # pcmask = np.zeros (self.opt['max_length'])
            #
            # pcmask[0:left+1] = 0
            # pcmask[left+1:right+1] = 1
            # pcmask[right+1:l] = 2
            # pcmask[l:]=3


            res+=[subj_type,obj_type,relation]

            if self.over_sampling and relation != 0:  # over-sampling in training
                processed += 2 * [tuple(res)]
            else:
                processed += [tuple(res)]



        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 12

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        #orig_idx = None

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]
        words = self.get_long_tensor(words, batch_size)
        masks = words>0
        pos = self.get_long_tensor(batch[1], batch_size)
        ner = self.get_long_tensor(batch[2], batch_size)
        deprel = self.get_long_tensor(batch[3], batch_size)
        head = self.get_long_tensor(batch[4], batch_size)
        subj_positions = self.get_long_tensor(batch[5], batch_size)
        obj_positions = self.get_long_tensor(batch[6], batch_size)
        subj_positions2 = self.get_long_tensor (batch[7], batch_size)
        obj_positions2 = self.get_long_tensor (batch[8], batch_size)
        subj_type = self.get_long_tensor (batch[9], batch_size)
        obj_type = self.get_long_tensor (batch[10], batch_size)
        rels = torch.LongTensor(batch[11])



        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions,subj_positions2, obj_positions2,  subj_type, obj_type, rels,orig_idx,)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def get_long_tensor(self,tokens_list, batch_size):
        """ Convert list of list of tokens to a padded LongTensor. """

        token_len = max(len(x) for x in tokens_list )
        #token_len = self.opt['max_length']

        tokens = torch.LongTensor (batch_size, token_len).fill_ (constant.PAD_ID)

        for i, s in enumerate (tokens_list):
            tokens[i, :len (s)] = torch.LongTensor (s)

        return tokens


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def map_to_ids_padding(tokens, vocab):
    stop_list = constant.stopwords
    ids = []
    for i,t in enumerate(tokens):
        if t not in stop_list:
            if t in vocab:
                ids.append(vocab[t])
            else:
                ids.append(constant.UNK_ID)
        else:
            ids.append(constant.PAD_ID)
    return ids

def get_positions(start_idx, end_idx, length):  # very gracefully to generate the relative position information # whether need to clip,maybe mask is enough
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_clip_positions(start_idx, end_idx, length,max_len):  # very gracefully to generate the relative position information
    """ Get subj/obj position sequence. """
    res = list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))
    res = list(map(lambda x: max(0,min(x+max_len,max_len+max_len+1)),res))
    return res


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

