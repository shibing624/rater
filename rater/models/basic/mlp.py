import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, feat_dim, fc_dims, dropout=None, batch_norm=None):
        super(MLP, self).__init__()
        self.fc_dims = fc_dims
        layer_dims = [feat_dim]
        layer_dims.extend(fc_dims)
        layers = []
        for i in range(len(layer_dims) - 1):
            fc_layer = nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1])
            nn.init.xavier_uniform_(fc_layer.weight)
            layers.append(fc_layer)
            if batch_norm:
                batch_norm_layer = nn.BatchNorm1d(num_features=layer_dims[i + 1])
                layers.append(batch_norm_layer)
            layers.append(nn.ReLU())
            if dropout:
                dropout_layer = nn.Dropout(dropout)
                layers.append(dropout_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature):
        y = self.mlp(feature)
        return y
