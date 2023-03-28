# encoding=utf-8
import sys
sys.path.append('../')
import torch
import random
import argparse
import numpy as np
from vocab import Vocab
from utils import helper

from sklearn import metrics
from bert_loader import ABSADataLoader
from bert_trainer import ABSATrainer
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("../common/bert-base-uncased/vocab.txt")
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset/Restaurants")
parser.add_argument("--vocab_dir", type=str, default="dataset/Restaurants")
parser.add_argument("--pretrained_model_path", type=str, default="")
parser.add_argument("--lower", default=True, help="Lowercase all words.")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
parser.add_argument("--max_len", type=int, default=90)

args = parser.parse_args()

# load vocab
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_tok.vocab")  # token
post_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_post.vocab")  # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pos.vocab")  # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_dep.vocab")  # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pol.vocab")  # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print(
    "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)
    )
)
args.tok_size = len(token_vocab)
args.post_size = len(post_vocab)
args.pos_size = len(pos_vocab)
args.dep_size = len(dep_vocab)

# load data
print("Loading data from {} with batch size {}...".format(args.data_dir, args.batch_size))
test_batch = ABSADataLoader(
    args.data_dir + "/test.json", args.batch_size, args, vocab, shuffle=False
)
import sys
import seaborn as sns
import matplotlib.pyplot as plt
torch.set_printoptions(profile="full")
# torch.set_printoptions()

def evaluate(model, data_loader):
    predictions, labels = [], []
    val_loss, val_acc, val_step = 0.0, 0.0, 0
    sum_0_acc = 0
    sum_2_acc = 0
    sum_1_acc = 0
    sum_1 = 0
    sum_2 = 0
    sum_0 = 0
    for i, batch in enumerate(data_loader):
        loss, acc, pred, label, predprob , sem_adj, adj_pos, adj = model.predict(batch)
        val_loss += loss
        val_acc += acc
        predictions += pred
        labels += label
        val_step += 1
        # print(label)
        # y = batch[9]
        # for j in range(len(batch[9])):
        #     sum = sum+1
        #     # print(sum)
        #     # neu = neu + 1
        #     if label[j]==pred[j]:
        #         if label[j]==0:
        #             neu=neu+1
        #         elif label[j]==1:
        #             pos = pos+1
        #         else:
        #             neg=neg+1
        # y = batch[10]
        # for j in range(len(batch[9])):
        #      token = tokenizer.convert_ids_to_tokens(y[j])
        #
        #      for m in range(len(token)):
        #          if token[m]=='[SEP]':
        #             temp = m
        #             break
        #      x = sem_adj[j][1:temp,1:temp]
        #      x_label = token[1:temp]
        #      y_label = token[1:temp]
        #      data = x.cpu().detach().numpy()
        #      # data2 =
        #      # sns.heatmap(data=data)
        #      print(tokenizer.convert_ids_to_tokens(y[j]))
        #      sns.heatmap(data=data,xticklabels =x_label,yticklabels=y_label)
        #      plt.show()

        #     print(label[j],pred[j],pred[j],predprob[j])

        #     data = sns.load_dataset(sem_adj[j]). \
        #         pivot(index='month', columns='years')


    # print(sum,neu,pos,neg)
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average="macro")
    return val_loss / val_step, val_acc / val_step, f1_score


def _totally_parameters(model):  #
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


# build model
best_path = args.pretrained_model_path
print("Test Set: {}".format(len(test_batch)))

print("Loading best checkpoint from ", best_path)
trainer = torch.load(best_path)
print(trainer.model)
print('# parameters:', _totally_parameters(trainer.model))
test_loss, test_acc, test_f1 = evaluate(trainer, test_batch)
print("Evaluation Results: test_loss:{}, test_acc:{}, test_f1:{}".format(test_loss, test_acc, test_f1))