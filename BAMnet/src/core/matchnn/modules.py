'''
Created on Sep, 2017

@author: hugo

'''
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from ..bamnet.utils import to_cuda
from ..bamnet.modules import *
from ..bow.modules import BOWAnsEncoder


INF = 1e20
VERY_SMALL_NUMBER = 1e-10
class MatchNN(nn.Module):
    def __init__(self, vocab_size, vocab_embed_size, o_embed_size, \
        hidden_size, num_ent_types, num_relations, num_query_words, \
        constraint_mark_emb=None, \
        word_emb_dropout=None,\
        que_enc_dropout=None,\
        ans_enc_dropout=None, \
        pre_w2v=None, \
        num_hops=1, \
        att='add', \
        use_cuda=True):
        super(MatchNN, self).__init__()
        self.use_cuda = use_cuda
        self.constraint_mark_emb = constraint_mark_emb
        self.word_emb_dropout = word_emb_dropout
        self.que_enc_dropout = que_enc_dropout
        self.ans_enc_dropout = ans_enc_dropout
        self.num_hops = num_hops
        self.hidden_size = hidden_size

        self.word_emb = nn.Embedding(vocab_size, vocab_embed_size, padding_idx=0)
        self.init_word_emb(pre_w2v)
        if self.constraint_mark_emb is not None:
            mark_embed_size = self.constraint_mark_emb
            self.mark_emb = nn.Embedding(4, mark_embed_size)
            print('[ Using constraint modelling ]')
        else:
            mark_embed_size = 0
            self.mark_emb = None

        self.que_enc = SeqEncoder(vocab_size, vocab_embed_size + mark_embed_size, hidden_size, \
                        seq_enc_type='lstm', \
                        word_emb_dropout=word_emb_dropout, \
                        bidirectional=True, \
                        use_cuda=use_cuda).que_enc

        self.ans_enc = AnsSeqEncoder(o_embed_size, hidden_size, \
                        num_ent_types, num_relations, \
                        vocab_size=vocab_size, \
                        vocab_embed_size=vocab_embed_size, \
                        shared_embed=self.word_emb, \
                        word_emb_dropout=word_emb_dropout, \
                        ans_enc_dropout=ans_enc_dropout, \
                        mark_emb_size=mark_embed_size, \
                        mark_emb=self.mark_emb, \
                        use_cuda=use_cuda)


    def init_word_emb(self, init_word_embed):
        if init_word_embed is not None:
            print('[ Using pretrained word embeddings ]')
            self.word_emb.weight.data.copy_(torch.from_numpy(init_word_embed))
        else:
            self.word_emb.weight.data.uniform_(-0.08, 0.08)

    def query_kb_enc(self, memories, queries, query_marks, query_lengths, ans_mask, ctx_mask=None):
        # Question encoder
        query_emb = self.word_emb(queries)
        if self.word_emb_dropout:
            query_emb = F.dropout(query_emb, p=self.word_emb_dropout, training=self.training)

        if self.constraint_mark_emb is not None:
            query_mark_vec = self.mark_emb(query_marks)
            query_vec = torch.cat([query_emb, query_mark_vec], -1)
        else:
            query_vec = query_emb

        q_r = self.que_enc(query_vec, query_lengths)[1]
        if self.que_enc_dropout:
            q_r = F.dropout(q_r, p=self.que_enc_dropout, training=self.training)



        # Answer encoder
        _, x_bow, x_bow_len, _, x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ent, x_ctx_ent_marks, x_ctx_ent_len, x_ctx_ent_num, _, _, _, _ = memories
        ans_comp = self.ans_enc(x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ent, x_ctx_ent_marks, x_ctx_ent_len, x_ctx_ent_num)
        if self.ans_enc_dropout:
            for _ in range(len(ans_comp)):
                ans_comp[_] = F.dropout(ans_comp[_], p=self.ans_enc_dropout, training=self.training)

        ans_comp = torch.cat([each.unsqueeze(2) for each in ans_comp], 2).sum(2)
        return ans_comp, q_r

    def forward(self, memories, queries, query_marks, query_lengths, query_words, ctx_mask=None):
        ctx_mask = None
        mem_hop_scores = []
        ans_mask = create_mask(memories[0], memories[3].size(1), self.use_cuda)

        # Kb-aware question attention module
        ans_vec, q_r = self.query_kb_enc(memories, queries, query_marks, query_lengths, ans_mask, ctx_mask=ctx_mask)
        mid_score = self.scoring(ans_vec, q_r, mask=ans_mask)
        mem_hop_scores.append(mid_score)

        return mem_hop_scores

    def scoring(self, ans_r, q_r, mask=None):
        score = torch.bmm(ans_r, q_r.unsqueeze(2)).squeeze(2)
        if mask is not None:
            score = mask * score - (1 - mask) * INF # Make dummy candidates have large negative scores
        return score

class AnsSeqEncoder(nn.Module):
    """Answer Encoder"""
    def __init__(self, o_embed_size, hidden_size, num_ent_types, num_relations, vocab_size=None, \
                    vocab_embed_size=None, shared_embed=None, word_emb_dropout=None, \
                    ans_enc_dropout=None, mark_emb_size=None, mark_emb=None, use_cuda=True):
        super(AnsSeqEncoder, self).__init__()
        # Cannot have embed and vocab_size set as None at the same time.
        self.use_cuda = use_cuda
        self.word_emb_dropout = word_emb_dropout
        self.ans_enc_dropout = ans_enc_dropout
        self.hidden_size = hidden_size
        self.ent_type_embed = nn.Embedding(num_ent_types, o_embed_size // 8, padding_idx=0)
        self.relation_embed = nn.Embedding(num_relations, o_embed_size, padding_idx=0)
        self.embed = shared_embed if shared_embed is not None else nn.Embedding(vocab_size, vocab_embed_size, padding_idx=0)
        self.vocab_embed_size = self.embed.weight.data.size(1)
        self.mark_emb = mark_emb

        self.linear_type_bow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_paths = nn.Linear(hidden_size + o_embed_size, hidden_size, bias=False)
        self.linear_ctx = nn.Linear(hidden_size, hidden_size, bias=False)

        # lstm for ans encoder
        self.lstm_enc_type = EncoderRNN(vocab_size, self.vocab_embed_size, hidden_size, \
                        dropout=word_emb_dropout, \
                        bidirectional=True, \
                        rnn_type='lstm', \
                        use_cuda=use_cuda)
        self.lstm_enc_path = EncoderRNN(vocab_size, self.vocab_embed_size, hidden_size, \
                        dropout=word_emb_dropout, \
                        bidirectional=True, \
                        rnn_type='lstm', \
                        use_cuda=use_cuda)
        self.lstm_enc_ctx = EncoderRNN(vocab_size, self.vocab_embed_size + mark_emb_size, hidden_size, \
                        dropout=word_emb_dropout, \
                        bidirectional=True, \
                        rnn_type='lstm', \
                        use_cuda=use_cuda)

    def forward(self, x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ents, x_ctx_ent_marks, x_ctx_ent_len, x_ctx_ent_num):
        ans_type_bow, _, ans_path_bow, ans_paths, ans_ctx_ent = self.enc_ans_features(x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ents, x_ctx_ent_marks, x_ctx_ent_len, x_ctx_ent_num)

        ans_type_bow = self.linear_type_bow(ans_type_bow)
        ans_paths = self.linear_paths(torch.cat([ans_path_bow, ans_paths], -1))
        ans_ctx = self.linear_ctx(ans_ctx_ent)

        ans_comp = [ans_type_bow, ans_paths, ans_ctx]
        return ans_comp

    def enc_ans_features(self, x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ents, x_ctx_ent_marks, x_ctx_ent_len, x_ctx_ent_num):
        '''
        x_types: answer type
        x_paths: answer path, i.e., bow of relation
        x_ctx_ents: answer context, i.e., bow of entity words, (batch_size, num_cands, num_ctx, L)
        '''
        # ans_types = torch.mean(self.ent_type_embed(x_types.view(-1, x_types.size(-1))), 1).view(x_types.size(0), x_types.size(1), -1)
        x_type_bow_emb = self.embed(x_type_bow.view(-1, x_type_bow.size(-1)))
        x_type_bow_emb = F.dropout(x_type_bow_emb, p=self.word_emb_dropout, training=self.training)
        ans_type_bow = (self.lstm_enc_type(x_type_bow_emb, x_type_bow_len.view(-1))[1]).view(x_type_bow.size(0), x_type_bow.size(1), -1)

        x_path_bow_emb = self.embed(x_path_bow.view(-1, x_path_bow.size(-1)))
        x_path_bow_emb = F.dropout(x_path_bow_emb, p=self.word_emb_dropout, training=self.training)
        ans_path_bow = (self.lstm_enc_path(x_path_bow_emb, x_path_bow_len.view(-1))[1]).view(x_path_bow.size(0), x_path_bow.size(1), -1)
        ans_paths = torch.mean(self.relation_embed(x_paths.view(-1, x_paths.size(-1))), 1).view(x_paths.size(0), x_paths.size(1), -1)

        # Avg over ctx
        x_ctx_ents_emb = self.embed(x_ctx_ents.view(-1, x_ctx_ents.size(-1)))
        x_ctx_ents_emb = F.dropout(x_ctx_ents_emb, p=self.word_emb_dropout, training=self.training)
        ctx_num_mask = create_mask(x_ctx_ent_num.view(-1), x_ctx_ents.size(2), self.use_cuda).view(x_ctx_ent_num.shape + (-1,))

        if self.mark_emb is not None:
            ctx_ent_mark_vec = self.mark_emb(x_ctx_ent_marks.view(-1, x_ctx_ent_marks.size(-1)))
            x_ctx_ents_emb = torch.cat([x_ctx_ents_emb, ctx_ent_mark_vec], -1)
        else:
            x_ctx_ents_emb = x_ctx_ents_emb

        ans_ctx_ent = (self.lstm_enc_ctx(x_ctx_ents_emb, x_ctx_ent_len.view(-1))[1]).view(x_ctx_ents.size(0), x_ctx_ents.size(1), x_ctx_ents.size(2), -1)
        ans_ctx_ent = ctx_num_mask.unsqueeze(-1) * ans_ctx_ent
        ans_ctx_ent = torch.sum(ans_ctx_ent, dim=2) / torch.clamp(x_ctx_ent_num.float().unsqueeze(-1), min=VERY_SMALL_NUMBER)

        if self.ans_enc_dropout:
            # ans_types = F.dropout(ans_types, p=self.ans_enc_dropout, training=self.training)
            ans_type_bow = F.dropout(ans_type_bow, p=self.ans_enc_dropout, training=self.training)
            ans_path_bow = F.dropout(ans_path_bow, p=self.ans_enc_dropout, training=self.training)
            ans_paths = F.dropout(ans_paths, p=self.ans_enc_dropout, training=self.training)
            ans_ctx_ent = F.dropout(ans_ctx_ent, p=self.ans_enc_dropout, training=self.training)
        return ans_type_bow, None, ans_path_bow, ans_paths, ans_ctx_ent


# class SimpleRomHop(nn.Module):
#     def __init__(self, query_embed_size, in_memory_embed_size, hidden_size, atten_type='add'):
#         super(SimpleRomHop, self).__init__()
#         self.hidden_size = hidden_size
#         self.gru_linear_z = nn.Linear(in_memory_embed_size + hidden_size, hidden_size, bias=False)
#         self.gru_linear_r = nn.Linear(in_memory_embed_size + hidden_size, hidden_size, bias=False)
#         self.gru_linear_t = nn.Linear(in_memory_embed_size + hidden_size, hidden_size, bias=False)
#         self.gru_atten = Attention(hidden_size, query_embed_size, in_memory_embed_size, atten_type=atten_type)

#     def forward(self, h_state, in_memory_embed, out_memory_embed, atten_mask=None):
#         attention = self.gru_atten(h_state, in_memory_embed, atten_mask=atten_mask)
#         probs = torch.softmax(attention, dim=-1)

#         memory_output = torch.bmm(probs.unsqueeze(1), out_memory_embed).squeeze(1)
#         # GRU-like memory update
#         z = torch.sigmoid(self.gru_linear_z(torch.cat([h_state, memory_output], -1)))
#         r = torch.sigmoid(self.gru_linear_r(torch.cat([h_state, memory_output], -1)))
#         t = torch.tanh(self.gru_linear_t(torch.cat([r * h_state, memory_output], -1)))
#         output = (1 - z) * h_state + z * t
#         return output
