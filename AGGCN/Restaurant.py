# coding:utf-8
import sys
sys.path.append('../')
import torch
import math
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from common.tree import head_to_adj
from common.transformer_encoder import TransformerEncoder
from common.RGAT import RGATEncoder
from transformers import BertModel, BertConfig

bert_config = BertConfig.from_pretrained("../common/bert-base-uncased/config.json")
bert_config.output_hidden_states = True
bert_config.num_labels = 3
bert = BertModel.from_pretrained("../common/bert-base-uncased/pytroch_model.bin", config=bert_config)

import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, verbose=None):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j, batch_size):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j, batch_size):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * batch_size,)).to(emb_i.device)
            one_for_not_i = one_for_not_i.scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            if self.verbose: print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N, N) + l_ij(k + N, k, N)
        return 1.0 / (2 * N) * loss

class RGATABSA(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.bert_out_dim * 2
        self.args = args
        self.temperature = 1.0
        self.enc = ABSAEncoder(args)
        self.linear = nn.Linear(args.bert_out_dim * 3, args.bert_out_dim)
        self.classifier = nn.Linear(in_dim, args.num_class)
        self.mlp = nn.Linear(args.bert_out_dim, args.bert_out_dim)
        self.dropouts = nn.ModuleList(nn.Dropout(p) for p in np.linspace(0.1,0.5,5))
        self.dropout = nn.Dropout()

    def multi_dropout(self, pooling, y, loss_fn=None):

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = dropout(pooling)
                out = self.classifier(out)
                if loss_fn is not None:
                    loss = F.cross_entropy(out, y)
            else:
                temp_out = dropout(pooling)
                out2 = self.classifier(temp_out)
                out = out + out2
                if loss_fn is not None:
                    loss = loss + F.cross_entropy(out2, y, reduction="mean")
        if loss_fn is not None:
            return out/len(self.dropouts), loss / len(self.dropouts)
        return out/len(self.dropouts), None

    def forward(self, inputs):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            src_mask,
            mask,
            l,
            text_raw_bert_indices,
            bert_sequence,
            bert_segments_ids,
            label,
        ) = inputs
        criterion = nn.CrossEntropyLoss()
        outputs, loss_similar_syn, loss_similar_sem = self.enc(inputs)

        # zi = self.W2(F.relu(self.W1(output_syn)))
        # zj = self.W2(F.relu(self.W1(output_sem)))

        logits, loss_com = self.multi_dropout(outputs, label, criterion)
        # contrast_loss = self.contrastive_loss(zi, zj, zi.size(0))

        return logits,loss_com, loss_similar_syn, loss_similar_sem


class ABSAEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.contrast_loss_syn = ContrastiveLoss()
        self.contrast_loss_sem = ContrastiveLoss()
        self.mlp = nn.Linear(args.bert_out_dim * 3, args.bert_out_dim)
        self.pos_emb = (
            nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None
        )  # pos tag emb
        self.post_emb = (
            nn.Embedding(args.post_size, args.post_dim, padding_idx=0)
            if args.post_dim > 0
            else None
        )  # position emb
        if self.args.model.lower() in ["std", "gat"]:
            embs = (self.pos_emb, self.post_emb)
            self.encoder = DoubleEncoder(args, embeddings=embs, use_dep=True)
        elif self.args.model.lower() == "rgat":
            self.dep_emb = (
                nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0)
                if args.dep_dim > 0
                else None
            )  # position emb
            embs = (self.pos_emb, self.post_emb, self.dep_emb)
            self.encoder = DoubleEncoder(args, embeddings=embs, use_dep=True)

        if self.args.output_merge.lower() == "gate":
            self.gate_map = nn.Linear(args.bert_out_dim * 2, args.bert_out_dim)
        elif self.args.output_merge.lower() == "none":
            pass
        else:
            print('Invalid output_merge type !!!')
            exit()

    def different_loss(self, Z, ZC):
        diff_loss = torch.mean(torch.matmul(Z.permute(0, 2, 1), ZC) ** 2)
        return diff_loss

    def similarity_loss(self, ZCSY, ZCSE):
        ZCSY = F.normalize(ZCSY, p=2, dim=1)
        ZCSE = F.normalize(ZCSE, p=2, dim=1)
        similar_loss = torch.mean((ZCSY - ZCSE) ** 2)
        return similar_loss

    def attention_score(self, x, y, a_mask):

        x = x * a_mask.unsqueeze(-1).repeat(1,1,768)
        alpha_mat = torch.matmul(x, y.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, y).squeeze(1)
        return x

    def forward(self, inputs):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            src_mask,
            mask,
            l,
            text_raw_bert_indices,
            bert_sequence,
            bert_segments_ids,
            label,
        ) = inputs  # unpack inputs
        maxlen = max(l.data)
        """
        print('tok', tok, tok.size())
        print('asp', asp, asp.size())
        print('pos-tag', pos, pos.size())
        print('head', head, head.size())
        print('deprel', deprel, deprel.size())
        print('postition', post, post.size())
        print('mask', mask, mask.size())
        print('l', l, l.size())
        """

        adj_lst, label_lst, pos_lst = [], [], []
        for idx in range(len(l)):
            adj_i, label_i, pos_i = head_to_adj(
                maxlen,
                head[idx],
                tok[idx],
                deprel[idx],
                l[idx],
                mask[idx],
                directed=self.args.direct,
                self_loop=self.args.loop,
            )
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))
            pos_lst.append(pos_i.reshape(1, maxlen, maxlen))

        adj = np.concatenate(adj_lst, axis=0)  # [B, maxlen, maxlen]
        adj = torch.from_numpy(adj).cuda()
        labels = np.concatenate(label_lst, axis=0)  # [B, maxlen, maxlen]
        label_all = torch.from_numpy(labels).cuda()
        pos = np.concatenate(pos_lst, axis=0)  # [B, maxlen, maxlen]
        adj_pos = torch.from_numpy(pos).cuda()
        if self.args.model.lower() == "std":
            h = self.encoder(adj=None, inputs=inputs, lengths=l)
        elif self.args.model.lower() == "gat":
            h = self.encoder(adj=adj, inputs=inputs, lengths=l)
        elif self.args.model.lower() == "rgat":
            h = self.encoder(
                adj=adj, relation_matrix=label_all, adj_pos=adj_pos, inputs=inputs, lengths=l
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        out_com, out_syn, out_sem, bert_out, pool_out = h[0], h[1], h[2], h[3], h[4]
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                          # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.bert_out_dim)  # mask for h
        syn_enc_outputs = (out_syn * mask).sum(dim=1) / asp_wn        # mask h
        sem_enc_outputs = (out_sem * mask).sum(dim=1) / asp_wn
        com_syn_outputs = (out_com * mask).sum(dim=1) / asp_wn
        # com_sem_outputs = (com_sem * mask).sum(dim=1) / asp_wn
        loss_similar_syn = self.similarity_loss(out_com, out_syn)
        loss_similar_sem = self.similarity_loss(out_com, out_sem)
        #
        output = F.relu(self.mlp(torch.cat([syn_enc_outputs, sem_enc_outputs, com_syn_outputs], 1)))
        outputs = torch.cat([output, pool_out], 1)
        return outputs, loss_similar_syn, loss_similar_sem



class DoubleEncoder(nn.Module):
    def __init__(self, args, embeddings=None, use_dep=False):
        super(DoubleEncoder, self).__init__()
        self.args = args
        self.Sent_encoder = bert
        self.in_drop = nn.Dropout(args.input_dropout)
        # self.dense = nn.Linear(args.hidden_dim, args.bert_out_dim)  # dimension reduction
        self.num_layers = 2
        self.attention_heads = self.args.head_num
        self.attns = MultiHeadAttention(self.attention_heads, args.bert_out_dim)
        self.gc1 = GCN(args, args.bert_out_dim, args.bert_out_dim, self.num_layers)
        self.gc2 = GCN(args, args.bert_out_dim, args.bert_out_dim, self.num_layers)
        self.gc3 = GCN(args, args.bert_out_dim, args.bert_out_dim, self.num_layers)

        if use_dep:
            self.pos_emb, self.post_emb, self.dep_emb = embeddings
            self.Graph_encoder = RGATEncoder(
                num_layers=args.num_layer,
                d_model=args.bert_out_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dep_dim=self.args.dep_dim,
                att_drop=self.args.att_dropout,
                dropout=0.0,
                use_structure=True
            )
        else:
            self.pos_emb, self.post_emb = embeddings
            self.Graph_encoder = TransformerEncoder(
                num_layers=args.num_layer,
                d_model=args.bert_out_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dropout=0.0
            )
        if args.reset_pooling:
            self.reset_params(bert.pooler.dense)

    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def inputs_to_att_adj(self, input, score_mask, aspect=None):

        attn_tensor = self.attns(input, input, aspect_embedding=aspect, mask=score_mask)
        attn_tensor = torch.sum(attn_tensor, dim=1)
        attn_tensor = select(attn_tensor, self.args.top_k) * attn_tensor

        return attn_tensor



    def forward(self, adj, inputs, lengths, adj_pos, relation_matrix=None):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            src_mask,
            a_mask,
            l,
            text_raw_bert_indices,
            bert_sequence,
            bert_segments_ids,
            label,
        ) = inputs  # unpack inputs
        src_mask = src_mask.unsqueeze(-2)
        bert_sequence = bert_sequence[:, 0:bert_segments_ids.size(1)]
        # input()
        res = self.Sent_encoder(
            bert_sequence, token_type_ids=bert_segments_ids
        )
        bert_out = res['last_hidden_state']
        bert_pool_output = res['pooler_output']
        bert_out = self.in_drop(bert_out)
        bert_out = bert_out[:, 0:max(l), :]
        # if adj is not None:
        #     graph_mask = adj.eq(0)
        # else:
        #     graph_mask = None
        # # print('adj mask', mask, mask.size())
        # if lengths is not None:
        #     key_padding_mask = sequence_mask(lengths)  # [B, seq_len]
        #
        # if relation_matrix is not None:
        #     dep_relation_embs = self.dep_emb(relation_matrix)
        # else:
        #     dep_relation_embs = None
        #
        # inp = bert_out  # [bsz, seq_len, H]

        # bert_out = self.dense(bert_out)
        seq_len = bert_out.size(1)
        asp_wn = a_mask.sum(dim=1).unsqueeze(-1)                          # aspect words num
        mask = a_mask.unsqueeze(-1).repeat(1, 1, self.args.bert_out_dim)  # mask for h
        aspect_pool = (bert_out * mask).sum(dim=1) / asp_wn
        aspect_embedding_output = torch.stack(seq_len * [aspect_pool], dim=1)

        att_adj = self.inputs_to_att_adj(bert_out, src_mask, aspect_embedding_output)

        # adj_com = self.h_weight[0]*adj_pos + adj_ag*self.h_weight[1]
        # syn_out = F.relu(self.gc1(bert_out, adj_pos))
        # sem_out = F.relu(self.gc2(bert_out, att_adj))
        # graph_out = self.Graph_encoder(
        #     inp, mask=graph_mask, src_key_padding_mask=key_padding_mask, structure=dep_relation_embs
        # )  # [bsz, seq_len, H]

        outputs_syn = self.gc1(bert_out, adj_pos, att_adj, type="syn")
        outputs_sem = self.gc2(bert_out, adj_pos, att_adj, type="sem")
        outputs_com = self.gc3(bert_out, adj_pos, att_adj, type="common")

        # outputs = F.relu(outputs)
        # com_syn = F.relu(com_syn)
        # com_sem = F.relu(com_sem)
        return outputs_com, outputs_syn, outputs_sem, bert_out, bert_pool_output

class GCN(nn.Module):
    def __init__(self, args, in_dim, out_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.attention_heads = args.head_num
        self.layers = num_layers
        self.dense = nn.Linear(args.bert_out_dim*2, args.bert_out_dim)
        # drop out
        self.dropout = 0.4
        self.gcn_drop = nn.Dropout(self.dropout)

        # gcn layer
        self.W = nn.ModuleList()
        self.Z = nn.ModuleList()
        for layer in range(self.layers):
            self.W.append(nn.Linear(in_dim, out_dim))

        for layer in range(self.layers):
            self.Z.append(nn.Linear(in_dim, out_dim))
        #merger
        self.h_weight = nn.Parameter(torch.FloatTensor(2).normal_(0.5, 0.5))
        # self.h_weight1 = nn.Parameter(torch.FloatTensor(2).normal_(0.5, 0.5))
        # self.h_weight = nn.Parameter(torch.FloatTensor(2).normal_(0.5, 0.5))
        self.affine1 = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.affine2 = nn.Parameter(torch.Tensor(in_dim, out_dim))
    def SynGCN_layer(self, adj, gcn_inputs, denom, l):
        Ax = adj.bmm(gcn_inputs)
        AxW = self.W[l](Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs

    def SemGCN_layer(self, adj, gcn_inputs, denom, l):
        Ax = adj.bmm(gcn_inputs)
        AxW = self.Z[l](Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs
    def forward(self, inputs, adj, att_adj, type=None):
        # gcn
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj

        if type == 'sem':
            # for layer in self.transformers:
            for i in range(self.layers):
                outputs_sem = self.SemGCN_layer(att_adj, inputs, denom, i)
            return outputs_sem

        elif type == 'syn':
            for i in range(self.layers):
                outputs_syn = self.SynGCN_layer(adj, inputs, denom, i)
            return outputs_syn
        elif type == 'common':

            inter_adj = self.h_weight[0] * att_adj + self.h_weight[1] * adj / 2
            # inter_adj = att_adj + adj / 2
            for i in range(self.layers):
                outputs_com = self.SynGCN_layer(inter_adj, inputs, denom, i)
            return outputs_com





def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.q_aspect = nn.Linear(d_model, d_model)
        self.k_aspect = nn.Linear(d_model, d_model)
        self.lambda_q_context_layer = nn.Linear(self.d_k, 1)
        self.lambda_q_query_layer = nn.Linear(self.d_k, 1)
        self.lambda_k_context_layer = nn.Linear(self.d_k, 1)
        self.lambda_k_key_layer = nn.Linear(self.d_k, 1)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, query_a=None, key_a=None, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_probs = F.softmax(scores, dim=-1)

        score_aspect = torch.matmul(query_a, key_a.transpose(2, -1)) / math.sqrt(d_k)
        score_aspect = score_aspect.masked_fill(mask == 0, -1e9)
        quasi_scalar = 1.0
        quasi_attention_scores = 1.0 * quasi_scalar * torch.sigmoid(score_aspect)

        lambda_q_context = self.lambda_q_context_layer(query_a)
        lambda_q_query = self.lambda_q_query_layer(query)
        lambda_q = torch.sigmoid(lambda_q_context + lambda_q_query)
        lambda_k_context = self.lambda_k_context_layer(key_a)
        lambda_k_key = self.lambda_k_key_layer(key)
        lambda_k = torch.sigmoid(lambda_k_context + lambda_k_key)
        lambda_q_scalar = 1.0
        lambda_k_scalar = 1.0
        lambda_context = lambda_q_scalar * lambda_q + lambda_k_scalar * lambda_k
        lambda_context = (1 - lambda_context)
        quasi_attention_prob = lambda_context * quasi_attention_scores
        new_attention_probs = attention_probs + quasi_attention_prob

        if dropout is not None:
            new_attention_probs = dropout(new_attention_probs)


        return new_attention_probs

    def forward(self, query, key, aspect_embedding=None, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        seq_len = query.size(1)
        nbatches = query.size(0)
        # print(aspect.size())
        if aspect_embedding is not None:
            query_a = self.q_aspect(aspect_embedding).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            key_a = self.k_aspect(aspect_embedding).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        # print(query_a.size())
        attn = self.attention(query, key, query_a, key_a, mask=mask, dropout=self.dropout)
        return attn

def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))

def select(matrix, top_num):
    batch = matrix.size(0)
    len = matrix.size(1)
    matrix = matrix.reshape(batch, -1)
    maxk, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= maxk[i][-1])
    matrix = matrix.reshape(batch, len, len)
    matrix = matrix + matrix.transpose(-2, -1)

    # selfloop
    for i in range(batch):
        matrix[i].fill_diagonal_(1)

    return matrix

