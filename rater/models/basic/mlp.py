import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, fc_dims=[], dropout=0.0, is_batch_norm=False):
        super(MLP, self).__init__()
        self.fc_dims = fc_dims
        layer_dims = [input_size]
        if fc_dims:
            layer_dims.extend(fc_dims)
        layers = []
        for i in range(len(layer_dims) - 1):
            fc_layer = nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1])
            nn.init.xavier_uniform_(fc_layer.weight)
            layers.append(fc_layer)
            if is_batch_norm:
                batch_norm_layer = nn.BatchNorm1d(num_features=layer_dims[i + 1])
                layers.append(batch_norm_layer)
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                dropout_layer = nn.Dropout(dropout)
                layers.append(dropout_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature):
        y = self.mlp(feature)
        return y
