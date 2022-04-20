import jax#import torch
import pennylane as qml
#import jax.nn#import torch.nn as nn
import jax.numpy as jnp
import flax.linen as nn
import torch
import numpy as np
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
dev = qml.device('default.qubit.jax', wires=n_qubits)#dev = qml.device("default.qubit", wires=n_qubits)
# Enable CUDA device if available
# device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

@qml.qnode(dev, interface="jax", diff_method="backprop")
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
    a = 1/jnp.sum(probs)
    probsgiven5 *= a
    return jnp.expand_dims(jax.nn.softmax(probsgiven5, -1),0)#torch.nn.functional.softmax(probsgiven5, -1).float().unsqueeze(0)


def partial_measure_3(noise, weights, p_size):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights)
    # probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    # probsgiven0 /= torch.sum(probs)

    # Post-Processing
    # probsgiven = probsgiven0 / torch.max(probsgiven0)
    # return probsgiven
    
    probsgiven15 = probs[:p_size*3]
    probsgiven15 /= jnp.sum(probs)
    q = jax.nn.softmax(probsgiven15, -1)#torch.nn.functional.softmax(probsgiven15, -1)
    return jnp.concatenate((q[:p_size].float().unsqueeze(0)*patch_multiplier, q[p_size:p_size*2].float().unsqueeze(0)*patch_multiplier, q[p_size*2:p_size*3].float().unsqueeze(0)*patch_multiplier),0)
    
    
class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

   # def __init__(self, n_generators, key, q_delta=1):
    """
    Args:
        n_generators (int): Number of sub-generators to be used in the patch method.
        q_delta (float, optional): Spread of the random distribution for parameter initialisation.
    """

    #    super().__init__()
    #    self.q_params =[
                # nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
    #            1.05 * jax.random.normal(key,(q_depth,n_qubits))
    #            for _ in range(n_generators//patch_multiplier)
    #            ]
    #    self.n_generators = n_generators
    @nn.compact
    def __call__(self, x):
        q_params = [ 1.05 * jax.random.normal(jax.random.PRNGKey(0),(q_depth,n_qubits)) for _ in range(n_generators//patch_multiplier) ]
        # Size of each sub-generator output
        patch_size = output_size_subGen#2 ** (n_qubits - n_a_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        # images = torch.Tensor(x.size(0), 0).to(device)
        edges = jnp.zeros(x.shape[0])#torch.Tensor(x.size(0), 0)#.to(device)
        nodes = jnp.zeros(x.shape[0])#torch.Tensor(x.size(0), 0)#.to(device)
        
        edges_GT = jnp.zeros(x.shape[0])#torch.Tensor(x.size(0), 0)#.to(device)
        nodes_GT = jnp.zeros(x.shape[0])#torch.Tensor(x.size(0), 0)#.to(device)
        a = jnp.triu_indices(bond_matrix_size, k=1)#torch.triu_indices(bond_matrix_size, bond_matrix_size, offset=1)
        q_t = None
        if patch_multiplier not in [1, 3]:
            print("patch measurement undefined!!")
        for jj, elem in enumerate(x):
            patches_edges_list = []
            # patches_edges_list_GT = []
            patches_nodes = None#jnp.zeros((1,patch_size))#torch.Tensor(0, patch_size)#.to(device)
            patches_nodes_GT = None#jnp.zeros((1,patch_size))#torch.Tensor(0, patch_size)#.to(device)
            for ii, params in enumerate(q_params):
                q_out=partial_measure(elem, params, patch_size) if patch_multiplier==1 else partial_measure_3(elem, params, patch_size)
                if q_t is None: 
                    q_t_np = np.zeros_like(q_out)#torch.zeros_like(q_out)
                    q_t_np[0][-1] = (1)
                    q_t = jnp.array(q_t_np)
                    # q_t = torch.nn.functional.softmax(q_t[0], -1).float().unsqueeze(0)
                if ii < (upper_triangle_number)//patch_multiplier:
                    patches_edges_list.append(q_out)
                    # patches_edges_list_GT.append(q_t)
                else:
                    if patches_nodes is None:
                        patches_nodes = q_out
                    else:
                        patches_nodes = jnp.concatenate((patches_nodes, q_out))#torch.cat((patches_nodes, q_out))
                    if patches_nodes_GT is None:
                        patches_nodes_GT = q_t
                    else:
                        patches_nodes_GT = jnp.concatenate((patches_nodes_GT, q_t))#torch.cat((patches_nodes_GT, q_t))
            edge_np = np.zeros((bond_matrix_size,bond_matrix_size,patch_size))#torch.zeros(bond_matrix_size,bond_matrix_size,patch_size)#.to(device)
            edge_GT_np = np.zeros((bond_matrix_size,bond_matrix_size,patch_size))#torch.zeros(bond_matrix_size,bond_matrix_size,patch_size)#.to(device)
            for ii, q in enumerate(jnp.concatenate(tuple(patches_edges_list))):
                row = a[0][ii]; col = a [1][ii]
                edge_np[row][col][:] = q; edge_np[col][row][:] = q
                edge_GT_np[row][col][:] = q_t; edge_GT_np[col][row][:] = q_t
                edge = jnp.array(edge_np); edge_GT = jnp.array(edge_GT_np)
            node =  jnp.reshape(patches_nodes, (bond_matrix_size,patch_size))
            node_GT =  jnp.reshape(patches_nodes_GT, (bond_matrix_size,patch_size))
            if jj == 0: 
                edges =  edge
                nodes =  node
                edges_GT =  edge_GT
                nodes_GT =  node_GT
            elif jj == 1:
                edges = jnp.stack((edges, edge), 0) 
                nodes = jnp.stack((nodes, node), 0)
                edges_GT = jnp.stack((edges_GT, edge_GT), 0) 
                nodes_GT = jnp.stack((nodes_GT, node_GT), 0)
            else: 
                edges = jnp.concatenate((edges, jnp.expand_dims( edge,0)), 0) 
                nodes = jnp.concatenate((nodes, jnp.expand_dims( node,0)), 0)
                edges_GT = jnp.concatenate((edges_GT, jnp.expand_dims( edge_GT,0)), 0) 
                nodes_GT = jnp.concatenate((nodes_GT, jnp.expand_dims( node_GT,0)), 0)
        return edges, nodes, edges_GT, nodes_GT
                    
