import torch
import torch.nn.functional as F


def build_cross(num_fields, feat_emb):
    # num_pairs = num_fields * (num_fields-1) / 2
    row = []
    col = []
    for i in range(num_fields - 1):
        for j in range(i + 1, num_fields):
            row.append(i)
            col.append(j)
    p = feat_emb[:, row]  # N * num_pairs * emb_dim
    q = feat_emb[:, col]  # N * num_pairs * emb_dim
    return p, q


def bi_interaction(input_tensor):
    # tensor: N * F * emb_dim
    square_of_sum = torch.sum(input_tensor, dim=1)  # N * emb_dim
    square_of_sum = torch.mul(square_of_sum, square_of_sum)  # N * emb_dim

    sum_of_square = torch.mul(input_tensor, input_tensor)  # N * F * emb_dim
    sum_of_square = torch.sum(sum_of_square, dim=1)  # N * emb_dim

    bi_out = torch.sub(square_of_sum, sum_of_square)
    return bi_out  # N * emb_dim


def inner_product_attention_signal(query, keys, norm='softmax'):
    # query: N * emb_dim
    # keys: N * num_keys * emb_dim

    query = query.unsqueeze(dim=1)  # N * 1 * emb_dim
    kq = torch.mul(query, keys)  # N * num_keys * emb_dim (broadcast)
    kq = torch.sum(kq, dim=2)  # N * num_keys

    if norm == 'softmax':
        kq = F.softmax(kq)  # N * num_keys
    return kq
