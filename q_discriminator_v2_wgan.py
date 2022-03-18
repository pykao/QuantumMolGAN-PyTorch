import pennylane as qml
import sys
import torch

ATOM_NUM = 45
n_circuites = ATOM_NUM
n_qubits = 9

n_measured_wire = 1


dev = qml.device("default.qubit", wires=n_qubits)

MEASURED_QUBIT_IDX = 4#int(sys.argv[1])

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.001, normalize=(True))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(MEASURED_QUBIT_IDX, MEASURED_QUBIT_IDX+1)]

n_layers = 3
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

class HybridModel(torch.nn.Module):
    def __init__(self, LAYER3 = False):
        super().__init__()
        self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        out = self.qlayer_1(x)
        return out

