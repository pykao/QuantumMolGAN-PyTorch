import pennylane as qml
import sys
import torch

ATOM_NUM = 45
n_circuites = ATOM_NUM
n_qubits = 5

n_measured_wire = 1
n_2nd_qubits = 9#int(n_circuites * n_measured_wire)#int(np.ceil(np.sqrt(n_qubits * n_circuites)))
n_2nd_circuits = int(n_circuites/n_2nd_qubits)

n_3rd_circuits = 1
n_3rd_qubits = int( n_measured_wire * n_2nd_circuits / n_3rd_circuits)

print((n_circuites, n_qubits), (n_2nd_circuits, n_2nd_qubits), (n_3rd_circuits, n_3rd_qubits))
dev = qml.device("default.qubit", wires=n_qubits)
dev1 = qml.device("default.qubit", wires=n_2nd_qubits)
dev2 = qml.device("default.qubit", wires=n_3rd_qubits)

MEASURED_QUBIT_IDX = 2#int(sys.argv[1])
MEASURED_QUBIT_2ND_IDX = 7#int(sys.argv[2])

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=MEASURED_QUBIT_IDX))]

@qml.qnode(dev1, interface="torch", diff_method="backprop")
def qnode_(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_2nd_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_2nd_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(MEASURED_QUBIT_2ND_IDX, MEASURED_QUBIT_2ND_IDX+1)]

@qml.qnode(dev2, interface="torch", diff_method="backprop")
def qnode__(inputs, weights):

    qml.templates.AngleEmbedding(inputs, wires=range(n_3rd_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_3rd_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in [2, 3]]


n_layers = 1
weight_shapes = {"weights": (n_layers, n_qubits, 3)}
n_2nd_layers = 3
n_3rd_layers = 1
weight_shapes_2nd = {"weights": (n_2nd_layers, n_2nd_qubits, 3)}
weight_shapes_3rd = {"weights": (n_3rd_layers, n_3rd_qubits, 3)}


class HybridModel(torch.nn.Module):
    def __init__(self, LAYER3 = False):
        super().__init__()
        self.qlayer_1 = torch.nn.ModuleList([qml.qnn.TorchLayer(qnode, weight_shapes) for i in range(ATOM_NUM)])
        self.qlayer_21 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
        self.LAYER3 = LAYER3
        if self.LAYER3:
            self.qlayer_22 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
            self.qlayer_23 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
            self.qlayer_24 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
            self.qlayer_25 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
            self.qlayer_31 = qml.qnn.TorchLayer(qnode__, weight_shapes_3rd)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.split(x, 1, dim=1)

        for i, l in enumerate(self.qlayer_1):
            tmp = self.qlayer_1[i](x[i])
            if i > 0:
                out = torch.cat([out, tmp], axis = 2)
            else:
                out = tmp

        x = torch.squeeze(out, 1) #4 x 20
        if self.LAYER3:
            x = torch.split(x, 9, dim=1)
            out1 = self.qlayer_21(x[0])
            out2 = self.qlayer_22(x[1])
            out3 = self.qlayer_23(x[2])
            out4 = self.qlayer_24(x[3])
            out5 = self.qlayer_25(x[4])
            out = torch.cat([out1, out2, out3, out4, out5], axis = 1)
            out = self.qlayer_31(out)
        else:
            out = self.qlayer_21(x)
        return self.softmax(out)
