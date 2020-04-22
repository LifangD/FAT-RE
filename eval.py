"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from model.dataloader.loader import DataLoader
from trainer import TransTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from utils.helper import ensure_dir
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model_dir2', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--out', type=str, default='./new_trans.pkl', help="Save model predictions to this dir.")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = TransTrainer(opt)
trainer.load(model_file)

# model_file = args.model_dir2 + '/' + args.model
# print("Loading model from {}".format(model_file))
# opt = torch_utils.load_config(model_file)
# trainer2 = TransTrainer2(opt)
# trainer2.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
id2word = vocab.id2word

# load data
# opt['batch_size'] =1
#opt['data_dir'] = "dataset/tacred-sample"
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
#data_file = "case.json"
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
batch_iter = tqdm(batch)

def viz_att(words,attn,name,label):
    sns.set()
    f, ax = plt.subplots(figsize=(20,20))
    df = pd.DataFrame(attn,index=words,columns=words)
    sns.heatmap(df,xticklabels =words,yticklabels=words,cmap="YlGnBu",ax=ax)
    ax.set_title(name)
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=90, horizontalalignment='right')
    fig_path = "svgs/"+str(label)
    ensure_dir(fig_path)
    f.savefig(fig_path+"/"+name+'.svg', format='svg', bbox_inches='tight')

def viz_token(ids,id2word):

    token = []
    for id in ids:
        token.append(id2word[id])
    token.append("cls")
    return token

for i, b in enumerate(batch_iter):

    preds, probs, _,attn_list= trainer.predict(b)


    # for i in range(len(attn_list)):
    #     layer =i
    #     for j in range(attn_list[layer].size(0)):
    #         bat = j
    #         for k in range(attn_list[layer][bat,:,:,:].size(0)):
    #             head =k
    #             attn_mat = attn_list[layer][bat,:,:,:][head,:,:]
    #             token_id = b[0][bat,:].data.cpu().numpy()
    #             token = viz_token(token_id,id2word)
    #             name = "layer"+str(layer)+"_"+"head"+str(head)
    #             #print(token)
    #             label =b[13][bat]
    #             viz_att(token,attn_mat.data.cpu().numpy(),name,label.data.cpu().numpy())
    #             print(name+".svg saved")

    predictions += preds
    all_probs += probs


predictions = [id2label[p] for p in predictions]
acc,p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")


# args.out = "./new_trans.pkl"
# with open(args.out, 'wb') as outfile:
#     pickle.dump(all_probs, outfile)
# print("Prediction scores saved to {}.".format(args.out))


# predictions = []
# all_probs = []
# for i, b in enumerate (batch_iter):
#     preds, probs, _ = trainer2.predict (b)
#
#     # for i in range(len(attn_list)):
#     #     layer =i
#     #     for j in range(attn_list[layer].size(0)):
#     #         bat = j
#     #         for k in range(attn_list[layer][bat,:,:,:].size(0)):
#     #             head =k
#     #             attn_mat = attn_list[layer][bat,:,:,:][head,:,:]
#     #             token_id = b[0][bat,:].data.cpu().numpy()
#     #             token = viz_token(token_id,id2word)
#     #             name = "layer"+str(layer)+"_"+ "batch"+str(bat)+"_"+"head"+str(head)
#     #             #print(token)
#     #             label =b[12][bat]
#     #             viz_att(token,attn_mat.data.cpu().numpy(),name,label.data.cpu().numpy())
#     #             #print(name+".jpg saved")
#
#     predictions += preds
#     all_probs += probs
#
# predictions = [id2label[p] for p in predictions]
# acc, p, r, f1 = scorer.score (batch.gold (), predictions, verbose=True)
# print ("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format (args.dataset, p, r, f1))
#
# print ("Evaluation ended.")
#
# predictions = []
# all_probs = []

# for i, b in enumerate (batch_iter):
#     logits1, _ = trainer.predict (b,False)
#     logits2 ,_ = trainer2.predict (b,False)
#     lamda = 0.5
#     logits = logits1*lamda+logits2*(1-lamda)
#     probs = F.softmax (logits2, 1).data.cpu ().numpy ().tolist ()
#     preds = np.argmax (logits2.data.cpu ().numpy (), axis=1).tolist ()
#
#     # for i in range(len(attn_list)):
#     #     layer =i
#     #     for j in range(attn_list[layer].size(0)):
#     #         bat = j
#     #         for k in range(attn_list[layer][bat,:,:,:].size(0)):
#     #             head =k
#     #             attn_mat = attn_list[layer][bat,:,:,:][head,:,:]
#     #             token_id = b[0][bat,:].data.cpu().numpy()
#     #             token = viz_token(token_id,id2word)
#     #             name = "layer"+str(layer)+"_"+ "batch"+str(bat)+"_"+"head"+str(head)
#     #             #print(token)
#     #             label =b[12][bat]
#     #             viz_att(token,attn_mat.data.cpu().numpy(),name,label.data.cpu().numpy())
#     #             #print(name+".jpg saved")
#
#     predictions += preds
#     all_probs += probs
#
# predictions = [id2label[p] for p in predictions]
# acc, p, r, f1 = scorer.score (batch.gold (), predictions, verbose=True)
# print ("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format (args.dataset, p, r, f1))
#
# print ("Evaluation ended.")


