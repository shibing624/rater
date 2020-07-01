import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import inner_product_attention_signal


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.attention_signal = AttentionSignal(dim, 'inner-product', True)

    def forward(self, query, keys, values):
        # query: N * emb_dim
        # keys: N * num_keys * emb_dim
        # values: N * num_keys * emb_dim
        att_signal = self.attention_signal(query, keys)  # N * num_keys
        att_signal = att_signal.unsqueeze(dim=2)  # N * num_keys * 1
        weighted_values = torch.mul(att_signal, values)
        return weighted_values  # N * num_keys * emb_dim


class AttentionSignal(nn.Module):
    def __init__(self, dim, similarity='inner-product', scale=False, activation='relu'):
        super(AttentionSignal, self).__init__()
        self.dim = dim
        self.similarity = similarity
        self.scale = scale
        self.activation = activation
        if similarity == 'inner-product':  # a_i = query^T * keys_i
            pass

        elif self.similarity == 'concat':  # a_i = v^T * ReLU(W_q * query + W_k * keys_i)
            # v
            self.v_a = nn.Parameter(torch.zeros(dim))
            nn.init.xavier_uniform_(self.v_a.data)
            # W_q
            self.weights_q = nn.Parameter(torch.zeros((dim, dim)))
            nn.init.xavier_uniform_(self.weights_q.data)
            # W_k
            self.weights_k = nn.Parameter(torch.zeros((dim, dim)))
            nn.init.xavier_uniform_(self.weights_k.data)

        else:  # general, a_i = query^T * W * keys_i
            self.weights_a = nn.Parameter(torch.zeros((dim, dim)))
            nn.init.xavier_uniform_(self.weights_a.data)

    def forward(self, query, keys):
        # query: N * emb_dim
        # keys: N * num_keys * emb_dim

        if self.similarity == 'inner-product':
            att = inner_product_attention_signal(query, keys, None)

        elif self.similarity == 'concat':
            query = query.unsqueeze(dim=1)  # N * 1 * emb_dim
            weighted_q = torch.matmul(query, self.weights_q)  # N * 1 * emb_dim
            weighted_k = torch.matmul(keys, self.weights_k)  # N * num_keys * emb_dim
            weighted_kq = torch.add(weighted_q, weighted_k)  # N * num_keys * emb_dim
            if not self.activation:
                pass
            elif self.activation == 'relu':
                weighted_kq = F.relu(weighted_kq)
            elif self.activation == 'tanh':
                weighted_kq = F.tanh(weighted_kq)
            elif self.activation == 'sigmoid':
                weighted_kq = F.sigmoid(weighted_kq)
            att = torch.mul(weighted_kq, self.v_a)  # N * num_keys * emb_dim
            att = torch.sum(att, dim=2)  # N * num_keys

        else:
            query = query.unsqueeze(dim=1)  # N * 1 * emb_dim
            qw = torch.matmul(query, self.weights_a)  # (N * 1 * emb_dim) * (emb_dim * emb_dim) = N * 1 * emb_dim
            qw = qw.transpose(1, 2)  # N * emb_dim * 1
            att = torch.bmm(keys, qw)  # (N * num_keys * emb_dim) * (N * emb_dim * 1) = N * num_keys * 1
            att = att.squeeze()  # N * num_keys
        if self.scale:
            att = att / torch.sqrt(self.dim)
        return F.softmax(att)
