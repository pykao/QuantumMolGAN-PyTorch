import torch
import torch.nn as nn

# Used in MultiGraphConvolutionalLayers
# MolGAN/utils/layers.py graph_convolution_layer
# Implementation-MolGAN-PyTorch/layers.py GraphConvolutionLayer
# Checked by Ken in September 14, 2021
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, units, activation, edge_type_num, dropout_rate=0.):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.units = units
        self.edge_type_num = edge_type_num
        self.adj_list = nn.ModuleList()
        self.dropout_rate = dropout_rate
        for _ in range(edge_type_num):
            self.adj_list.append(nn.Linear(in_features, units))
        self.linear = nn.Linear(in_features, units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node_tensor, adjancy_tensor, hidden_tensor=None):
        if hidden_tensor is not None:
            # TF MolGAN
            annotations = torch.cat((hidden_tensor, node_tensor), -1)
            # Implementation-MolGAN-PyTorch
            #annotations = torch.cat((node_tensor, hidden_tensor), -1)
        else:
            annotations = node_tensor
        output = torch.stack([self.adj_list[i](annotations) for i in range(self.edge_type_num)], 1)
        output = torch.matmul(adjancy_tensor, output)
        output = torch.sum(output, dim=1) + self.linear(annotations)
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output)
        return output

# Used in MolGAN/models/__init__.py encoder_rgcn
# MolGAN/utils/layers.py multi_graph_convolution_layers
# Implementation-MolGAN-PyTorch/layers.py MultiGraphConvolutionLayers
# Checked by Ken in September 14, 2021
class MultiGraphConvolutionLayers(nn.Module):
    def __init__(self, in_features, units, activation, edge_type_num, with_features=False, f_dim=0, dropout_rate=0.):
        super(MultiGraphConvolutionLayers, self).__init__()
        self.in_features = in_features
        self.units = units
        self.activation = activation
        self.edge_type_num = edge_type_num
        self.with_features = with_features
        self.f_dim = f_dim
        self.dropout_rate = dropout_rate
        self.conv_nets = nn.ModuleList()
        in_units = list()
        if with_features:
            for i in range(len(units)):
                in_units = list([x + in_features for x in units])
            for u0, u1 in zip([in_features + f_dim] + in_units[:-1], units):
                self.conv_nets.append(GraphConvolutionLayer(u0, u1, activation, edge_type_num, dropout_rate))
        else:
            for i in range(len(units)):
                in_units = list([x + in_features for x in units])
            for u0, u1 in zip([in_features] + in_units[:-1], units):
                self.conv_nets.append(GraphConvolutionLayer(u0, u1, activation, edge_type_num, dropout_rate))

    def forward(self, node_tensor, adjacency_tensor, hidden_tensor=None):
        for conv_idx in range(len(self.units)):
            hidden_tensor = self.conv_nets[conv_idx](node_tensor, adjacency_tensor, hidden_tensor)
        return hidden_tensor


# Used in MolGAN/models/__init__.py encoder_rgcn.graph_aggregation
# MolGAN/utils/layers.py graph_aggregation_layer
# Implementation-MolGAN-PyTorch GraphAggregation
# Checked by Ken in September 14, 2021
class GraphAggregation(nn.Module):
    def __init__(self, in_features, aux_units, activation, with_features=False, f_dim=0, dropout_rate=0.):
        super(GraphAggregation, self).__init__()
        self.in_features = in_features
        self.aux_units = aux_units
        self.activation = activation
        self.with_features = with_features
        self.f_dim = f_dim
        self.dropout_rate = dropout_rate
        if with_features:
            self.i = nn.Sequential(nn.Linear(in_features + f_dim, aux_units), nn.Sigmoid())
            j_layers = [nn.Linear(in_features + f_dim, aux_units)]
            if self.activation is not None:
                j_layers.append(self.activation)
            self.j = nn.Sequential(*j_layers)
        else:
            self.i = nn.Sequential(nn.Linear(in_features, aux_units), nn.Sigmoid())
            j_layers = [nn.Linear(in_features, aux_units)]
            if self.activation is not None:
                j_layers.append(self.activation)
            self.j = nn.Sequential(*j_layers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node_tensor, output_tensor, hidden_tensor=None):
        if hidden_tensor is not None:
            annotations = torch.cat((output_tensor, hidden_tensor, node_tensor), -1)
        else:
            annotations = torch.cat((output_tensor, node_tensor), -1)
        # The i seems to be an attention
        i = self.i(annotations)
        j = self.j(annotations)
        output = torch.sum(torch.mul(i, j), 1)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)

        return output

# Used in MolGAN/models/__init__.py decoder_adj
# Used in MolGAN/models/gan.py GraphGANModel.D_x and GraphGANModel.V_x
# MolGAN/units/layers.py multi_dense_layers
# Implementation-MolGAN-PyTorch MultiDenseLayer
# Check by Ken in September 14, 2021
class MultiDenseLayers(nn.Module):
    def __init__(self, aux_unit, linear_units, activation=None, dropout_rate=0.):
        super(MultiDenseLayers, self).__init__()
        self.aux_unit = aux_unit
        self.linear_units = linear_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        layers = list()
        for c0, c1 in zip([aux_unit] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            # Kind of different than Implementation-MolGAN-PyTorch
            if activation is not None:
                layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
        self.linear_layers = nn.Sequential(*layers)

    def forward(self, inputs):
        hidden_tensor = self.linear_layers(inputs)
        return hidden_tensor



# MolGAN/models/__init__.py encoder_rgcn.graph_convolutions
# Implementation-MolGAN-PyTorch/layers.py GraphConvolution
class GraphConvolution(nn.Module):
    def __init__(self, in_features, graph_conv_units, edge_type_num, with_features=False, f_dim=0, dropout_rate=0.):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.graph_conv_units = graph_conv_units
        self.edge_type_num =  edge_type_num
        self.with_features = with_features
        self.f_dim = f_dim
        self.dropout_rate = dropout_rate
        # line #9 in MolGAN/models/__init__.py
        self.activation_f = torch.nn.Tanh()
        self.multi_graph_convolution_layers = \
            MultiGraphConvolutionLayers(in_features, graph_conv_units, self.activation_f, edge_type_num, with_features, f_dim, dropout_rate)
    def forward(self, node_tensor, adjacency_tensor, hidden_tensor=None):
        output = self.multi_graph_convolution_layers(node_tensor, adjacency_tensor, hidden_tensor)
        return output
