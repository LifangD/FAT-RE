from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch


from model.dataloader.loader import DataLoader
from trainer import TransTrainer
from utils import scorer, constant, helper,torch_utils
from utils.vocab import Vocab
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='tacred')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')

parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')

parser.add_argument('--ner_dim', type=int, default=10, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=10, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=100, help=' hidden state size.')
parser.add_argument('--rnn_hidden', type=int, default=200, help=' hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of rnn layers.')

parser.add_argument('--trans_layers',type=int, default=4)
parser.add_argument('--multi_heads',type=int, default=4)
parser.add_argument('--ffn_ex_size',type=int, default=2)

parser.add_argument('--word_dropout', type=float, default=0.15, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--compress_dropout',type=float, default=0.3, help='Input dropout rate.') # 0.1:overfitting 0.5: underfitting
parser.add_argument('--attn_dropout',type = float,default=0.1,help ='MultiHeadAttention dropout rate')
parser.add_argument('--mlp_dropout',type=float,default=0.1,help = 'mlp dropout rate')
parser.add_argument('--layer_dropout',type=float,default=0.1,help = ' dropout rate between trans layers')

parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)


parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='avg', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', dest='pooling_l2',type=float, default=0)
parser.add_argument('--conv_l2', dest='conv_l2',type=float, default=0)
parser.add_argument('--l2', type=float, default=0, help='L2-regluarize for optimizer.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

parser.add_argument('--prune_k',type=int,default=1)


parser.add_argument('--ffn', choices=['ffn','ffn_conv','ffn_conv_d'],  default= 'ffn_conv',help='whether use dilation conv')
parser.add_argument('--connect', choices=['residual','highway'],  default='highway',help='how to connect sublayer')
parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')

parser.add_argument('--lr', type=float, default=0.03, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.8, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 weight decay.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')

parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax','adadelta'], default='sgd', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./output', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='0', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--model', type = str, help= "use gcn or gat or bert")

parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

parser.add_argument('--relative_pos_dim',type=int,default=10,help = 'the relative position feature dimention to sub and obj')
parser.add_argument('--relative_pos_size',type=int,default=10,help = 'the window size of relative position')
parser.add_argument('--memo',type=str,default="_",help="add description about the model setting")
parser.add_argument('--max_length',type=int,default=120,help="the max length of sentences")
parser.add_argument('--use_bert_embedding',action='store_true')
parser.add_argument('--pretrained_model',type=str)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False

elif args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.deterministic = True

init_time = time.time()

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id) # the relation num

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']
# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False,over_sampling=False)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True,over_sampling=False)



def transre_search(ffn,connect,hidden_dim,trans_layers,multi_heads,ffn_ex_size,initial,final):
    opt['weighted'] = False
    opt['rnn'] = False
    opt['ffn'] = ffn
    opt['connect'] = connect
    opt['hidden_dim'] = hidden_dim
    opt['trans_layers'] = trans_layers
    opt['multi_heads'] = multi_heads
    opt['ffn_ex_size'] = ffn_ex_size
    opt['initial'] = initial
    opt['final'] = final


    id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_name =str (opt['optim']) + '_' + str (opt['lr']) + str (ffn) + '_' +str(connect)+"_"\
                + str (hidden_dim) + '_' + str (trans_layers) + '_' + str (multi_heads) + '_' + \
                str (ffn_ex_size)+'_'+str(initial)+'_'+str(final)
    model_name = model_name+''+str(opt['memo'])

    model_name = str(id)+"_"+model_name


    model_save_dir = opt['save_dir'] + '/' + model_name
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)

    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    vocab.save(model_save_dir + '/vocab.pkl')
    file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")
    helper.print_config(opt)


    if not opt['load']:
        trainer = TransTrainer(opt, emb_matrix=emb_matrix)
    else:
        # load pre-train model
        model_file = opt['model_file']
        print("Loading model from {}".format(model_file))
        model_opt = torch_utils.load_config(model_file)
        model_opt['optim'] = opt['optim']
        trainer = TransTrainer(model_opt)
        trainer.load(model_file)


    id2label = dict([(v,k) for k,v in label2id.items()]) # the classification result
    dev_score_history = []
    dev_loss_history = []
    current_lr = opt['lr']

    global_step = 0
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
    max_steps = len(train_batch) * opt['num_epoch']

    best_result = "unknown"
    file_logger.log(str(opt['memo']))
    for epoch in range(1, opt['num_epoch']+1):
        train_loss = 0
        epoch_start_time = time.time()
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss,norm = trainer.update(batch)
            train_loss += loss
            if global_step % opt['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now(), global_step, max_steps, epoch,opt['num_epoch'], loss, duration, current_lr))

        print("Evaluating on dev set...")
        predictions = []
        dev_loss = 0
        for i, batch in enumerate(dev_batch):
            preds, _, loss,_ = trainer.predict(batch)
            predictions += preds
            dev_loss += loss
        predictions = [id2label[p] for p in predictions]
        train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
        dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']


        acc,dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
        print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,train_loss, dev_loss, dev_f1))
        dev_score = dev_f1
        file_logger.log("{}\t{:.3f}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, acc,train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))

        # save
        model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        trainer.save(model_file, epoch)

        if epoch == 1 or dev_score > max(dev_score_history):
            copyfile(model_file, model_save_dir + '/best_model.pt')
            best_result = (model_name,dev_score)
            print("new best model saved.")
            file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}".format(epoch, dev_p*100, dev_r*100, dev_score*100))
        if epoch % opt['save_epoch'] != 0:
            os.remove(model_file)

        # lr schedule
        if len(dev_score_history) > opt['decay_epoch'] and  dev_score<=dev_score_history[-1] and opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
            current_lr *= opt['lr_decay']
            trainer.update_lr(current_lr)

        dev_score_history += [dev_score]
        dev_loss_history +=[dev_loss]
        epoch_end_time = time.time()
        print("epoch time {:.3f}".format(epoch_end_time-epoch_start_time))
    return best_result

if __name__ == "__main__":


    ffn = 'ffn_conv' #[ffn,ffn_conv,ffn_conv_m]
    connect = 'highway'

    hidden_dim = 200
    trans_layers = 3
    multi_heads = 4
    ffn_ex_size = 2
    initial = "fc" #[fc,gru]
    final = "only_rel" #[only_rel,rel_none]

    model_best = transre_search(ffn,connect,hidden_dim,trans_layers,multi_heads,ffn_ex_size,initial,final)




