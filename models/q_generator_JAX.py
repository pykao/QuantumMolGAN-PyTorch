import torch
import pennylane as qml
import torch.nn as nn

# Quantum variables
n_qubits = 4  # Total number of qubits / N #( 5 for patch_multiplier = 3; 4 for patch_multiplier = 1)
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 45#90  # Number of subgenerators for the patch method / N_G# -*- coding: utf-8 -*-
patch_multiplier = 1
bond_matrix_size = 9
upper_triangle_number = (bond_matrix_size * bond_matrix_size - bond_matrix_size)//2
output_size_subGen = 5
# Quantum simulator
# dev = qml.device("lightning.qubit", wires=n_qubits)
dev = qml.device("default.qubit", wires=n_qubits)
# Enable CUDA device if available
# device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(noise, weights):

    weights = weights.reshape(q_depth, n_qubits)

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    for i in range(q_depth):
        # Parameterised layer
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)

        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))

# For further info on how the non-linear transform is implemented in Pennylane
# https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
def partial_measure(noise, weights, p_size):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights)
    # probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    # probsgiven0 /= torch.sum(probs)

    # Post-Processing
    # probsgiven = probsgiven0 / torch.max(probsgiven0)
    # return probsgiven
    

    probsgiven5 = probs[:p_size]
    #a = 1/torch.sum(probs)
    probsgiven5 /= torch.sum(probs)
    return torch.nn.functional.softmax(probsgiven5, -1).float().unsqueeze(0)


def partial_measure_3(noise, weights, p_size):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights)
    # probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    # probsgiven0 /= torch.sum(probs)

    # Post-Processing
    # probsgiven = probsgiven0 / torch.max(probsgiven0)
    # return probsgiven
    
    probsgiven15 = probs[:p_size*3]
    probsgiven15 /= torch.sum(probs)
    q = torch.nn.functional.softmax(probsgiven15, -1)
    return torch.cat((q[:p_size].float().unsqueeze(0)*patch_multiplier, q[p_size:p_size*2].float().unsqueeze(0)*patch_multiplier, q[p_size*2:p_size*3].float().unsqueeze(0)*patch_multiplier),0)
    
    
class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()
        tensor_mean = torch.full(size=(q_depth,n_qubits), fill_value=0, dtype=(torch.float32))
        tensor_std = torch.full(size=(q_depth, n_qubits), fill_value=1.05, dtype=(torch.float32))
        self.q_params = nn.ParameterList(
            [
                # nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                nn.Parameter(q_delta * torch.normal(mean=tensor_mean,std=tensor_std), requires_grad=True)
                for _ in range(n_generators//patch_multiplier)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        # Size of each sub-generator output
        patch_size = output_size_subGen#2 ** (n_qubits - n_a_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        # images = torch.Tensor(x.size(0), 0).to(device)

        '''
        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)
        
        return images
        '''
        ''' # 9 by 9 square matrix
        edges = torch.Tensor(x.size(0), 0).to(device)
        nodes = torch.Tensor(x.size(0), 0).to(device)
        for jj, elem in enumerate(x):
            patches_edges = torch.Tensor(0, patch_size).to(device)
            patches_nodes = torch.Tensor(0, patch_size).to(device)
            for ii, params in enumerate(self.q_params):
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                if ii < 81:
                    patches_edges = torch.cat((patches_edges, q_out))
                else:
                    patches_nodes = torch.cat((patches_nodes, q_out))
            edge =  torch.reshape(patches_edges, (9,9,5))
            node =  torch.reshape(patches_nodes, (9,5))
            if jj == 0: 
                edges =  edge
                nodes =  node
            elif jj == 1:
                edges = torch.stack((edges, edge), 0) 
                nodes = torch.stack((nodes, node), 0)
            else: 
                edges = torch.cat((edges, torch.unsqueeze( edge,0)), 0) 
                nodes = torch.cat((nodes, torch.unsqueeze( node,0)), 0)
        return edges, nodes
        '''
        ''' # 45 nodes and edges in a row
        edges_nodes = torch.Tensor(x.size(0), 0).to(device)
        for jj, elem in enumerate(x):
            # patches_edges_nodes = torch.Tensor(0, patch_size).to(device)
            q_out_list = []
            for ii, params in enumerate(self.q_params):
                q_out = partial_measure(elem, params)
                q_out_list.append(q_out)
            edge_node = torch.cat(tuple(q_out_list))
            if jj == 0: 
                edges_nodes = edge_node
            elif jj == 1:
                edges_nodes = torch.stack((edges_nodes, edge_node), 0 ) 
            else: 
                edges_nodes = torch.cat((edges_nodes, torch.unsqueeze( edge_node, 0 ) ), 0) 
        return edges_nodes
        '''
        edges = torch.Tensor(x.size(0), 0)#.to(device)
        nodes = torch.Tensor(x.size(0), 0)#.to(device)
        
        edges_GT = torch.Tensor(x.size(0), 0)#.to(device)
        nodes_GT = torch.Tensor(x.size(0), 0)#.to(device)
        a = torch.triu_indices(bond_matrix_size, bond_matrix_size, offset=1)
        q_t = None
        if patch_multiplier not in [1, 3]:
            print("patch measurement undefined!!")
        for jj, elem in enumerate(x):
            patches_edges_list = []
            # patches_edges_list_GT = []
            patches_nodes = torch.Tensor(0, patch_size)#.to(device)
            patches_nodes_GT = torch.Tensor(0, patch_size)#.to(device)
            for ii, params in enumerate(self.q_params):
                q_out=partial_measure(elem, params, patch_size) if patch_multiplier==1 else partial_measure_3(elem, params, patch_size)
                if q_t is None: 
                    q_t = torch.zeros_like(q_out)
                    q_t[0][-1] = 1
                    # q_t = torch.nn.functional.softmax(q_t[0], -1).float().unsqueeze(0)
                if ii < (upper_triangle_number)//patch_multiplier:
                    patches_edges_list.append(q_out)
                    # patches_edges_list_GT.append(q_t)
                else:
                    patches_nodes = torch.cat((patches_nodes, q_out))
                    patches_nodes_GT = torch.cat((patches_nodes_GT, q_t))
            edge = torch.zeros(bond_matrix_size,bond_matrix_size,patch_size)#.to(device)
            edge_GT = torch.zeros(bond_matrix_size,bond_matrix_size,patch_size)#.to(device)
            for ii, q in enumerate(torch.cat(tuple(patches_edges_list))):
                row = a[0][ii]; col = a [1][ii]
                edge[row][col][:] = q; edge[col][row][:] = q
                edge_GT[row][col][:] = q_t; edge_GT[col][row][:] = q_t
            node =  torch.reshape(patches_nodes, (bond_matrix_size,patch_size))
            node_GT =  torch.reshape(patches_nodes_GT, (bond_matrix_size,patch_size))
            if jj == 0: 
                edges =  edge
                nodes =  node
                edges_GT =  edge_GT
                nodes_GT =  node_GT
            elif jj == 1:
                edges = torch.stack((edges, edge), 0) 
                nodes = torch.stack((nodes, node), 0)
                edges_GT = torch.stack((edges_GT, edge_GT), 0) 
                nodes_GT = torch.stack((nodes_GT, node_GT), 0)
            else: 
                edges = torch.cat((edges, torch.unsqueeze( edge,0)), 0) 
                nodes = torch.cat((nodes, torch.unsqueeze( node,0)), 0)
                edges_GT = torch.cat((edges_GT, torch.unsqueeze( edge_GT,0)), 0) 
                nodes_GT = torch.cat((nodes_GT, torch.unsqueeze( node_GT,0)), 0)
        return edges, nodes, edges_GT, nodes_GT
                    
