# import json
#
#
# train_file = 'train.json'
# dev_file = 'dev.json'
# test_file = 'test.json'
#
# def load_tokens(filename,relation):
#     with open(filename) as infile:
#         data = json.load(infile)
#         sens = []
#         for d in data:
#             if d['relation'] == relation:
#                 ts = d['token']
#                 ss, se, os, oe = d['subj_start'], d['subj_end'], d['obj_start'], d['obj_end']
#                 # do not create vocab for entity words
#                 # ts[ss:se+1] = ['<PAD>']*(se-ss+1)
#                 # ts[os:oe+1] = ['<PAD>']*(oe-os+1)
#                 sub = ts[ss:se+1]
#                 obj = ts[os:oe+1]
#                 tokens = list(filter(lambda t: t!='<PAD>', ts))
#                 tokens = ' '.join(tokens)
#                 sens.append([tokens,sub,obj])
#     #print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
#     return sens
#
# relation = "org:member_of"
# sens = load_tokens(test_file,relation)
# for s in sens:
#     print(s)
# from utils.vocab import Vocab
# vocab_file = '../dataset/vocab/vocab.pkl'
# vocab = Vocab(vocab_file, load=True)
# with open("tacred-relation.txt","r") as file:
#     lines = file.readlines()
#     for line in lines:
#         sub,obj = line.split()
#         sub = vocab.word2id['SUBJ-'+sub]
#         obj = vocab.word2id['OBJ-'+obj]
#         print([sub,obj],",")


from utils import constant
import  numpy as np
data = np.array(constant.TAC_REL_DES)

print(data[:,0])