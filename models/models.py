import torch
import torch.nn as nn
import numpy as np
from models.layers import GraphConvolution, GraphAggregation, MultiGraphConvolutionLayers, MultiDenseLayers

# decoder_adj in MolGAN/models/__init__.py
# Implementation-MolGAN-PyTorch/models_gan.py Generator
class Generator(nn.Module):
    """Generator network of MolGAN"""

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator, self).__init__()
        self.conv_dims = conv_dims
        self.z_dim = z_dim
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.dropout_rate = dropout_rate

        self.activation_f = nn.Tanh()
        self.multi_dense_layers = MultiDenseLayers(z_dim, conv_dims, self.activation_f)
        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layers(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


# encoder_rgcn in MolGAN/model/__init__.py
# MolGAN/models/gan.py GraphGANModel.D_x
# Implementation-MolGAN-PyTorch/models_gan.py Discriminator
class Discriminator(nn.Module):
    """Discriminator network of MolGAN"""

    def __init__(self, conv_dims, m_dim, b_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.conv_dims = conv_dims
        self.m_dim = m_dim
        self.b_dim = b_dim
        self.with_features = with_features
        self.f_dim = f_dim
        self.dropout_rate = dropout_rate

        self.activation_f = nn.Tanh()
        # line #6
        graph_conv_dim, aux_dim, linear_dim = conv_dims
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f, with_features, f_dim, dropout_rate)
        self.multi_dense_layers = MultiDenseLayers(aux_dim, linear_dim, self.activation_f, dropout_rate)
        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adjacency_tensor, hidden, node, activation=None):
        adj = adjacency_tensor[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(node, h, hidden)
        #h = self.agg_layer(h, node, hidden)
        h = self.multi_dense_layers(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h
