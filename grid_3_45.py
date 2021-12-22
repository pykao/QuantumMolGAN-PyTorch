import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import torch
from scipy.signal import savgol_filter
import os

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
ATOM_NUM = 45
def read_data():
    r = np.load('real_fake_a_x/real.npy')
    f = np.load('real_fake_a_x/fake.npy')
    r_y = np.ones((r.shape[0]), dtype=np.int64)
    f_y = np.zeros((f.shape[0]), dtype=np.int64)
    d = np.abs(np.concatenate([r, f]))
    d = d.tolist()
    d_y = np.concatenate([r_y, f_y])
    d_y = d_y.tolist()
    return d, d_y


X, y = read_data()
X = np.array(X)
X = X[:, :ATOM_NUM, :]

y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
y_hot = torch.scatter(torch.zeros((len(y), 2)), 1, y_, 1)

import pennylane as qml
import sys
'''
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

MEASURED_QUBIT_IDX = int(sys.argv[1])
MEASURED_QUBIT_2ND_IDX = int(sys.argv[2])

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
'''
from q_discriminator import HybridModel
loss = torch.nn.L1Loss()
device='cuda'
model = HybridModel(LAYER3=True).to(device)
X = torch.tensor(X).float().to(device)#, requires_grad=True).float()
y_hot = y_hot.float().to(device)

batch_size = 4
batches = 256 // batch_size


data_loader = torch.utils.data.DataLoader(
    list(zip(X, y_hot)), batch_size=batch_size, shuffle=True, drop_last=True
)

opt = torch.optim.SGD(model.parameters(), lr=0.2)
epochs = 10
loss_curve = []
accuracy_curve = []
import datetime
for epoch in range(epochs):

    running_loss = 0
    print('========= epoch=======') 
    print(datetime.datetime.now())
    for xs, ys in data_loader:
        opt.zero_grad()
        out = model(xs)
        out = out.to(device)
        loss_evaluated = loss(out, ys)
        loss_evaluated.backward()
        opt.step()
        for p in model.parameters():
            print(p.grad.norm(), p.name())            
        running_loss += loss_evaluated
        loss_curve += [loss_evaluated]

    print(datetime.datetime.now())
    print('========================')
    avg_loss = running_loss / batches
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

    y_pred = model(X)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()


    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)

    accuracy_curve += [accuracy] * int(batches)

### ================================= ###
#   draw locss curve and accuracy curve 
### ================================= ###

loss_curve = [loss.cpu().detach().numpy() for loss in loss_curve]
loss_curve = np.array(loss_curve).tolist()
accuracy_curve_ = np.array(accuracy_curve).tolist()

print(f"Accuracy: {accuracy * 100}%")

title = 'mesaured: ({}), ({}, {}), accuracy: {:.2f} %'.format(MEASURED_QUBIT_IDX, MEASURED_QUBIT_2ND_IDX, MEASURED_QUBIT_2ND_IDX + 1, accuracy_curve[-1] * 100)

img_dir = 'res_3_45'
if os.path.exists(img_dir):
    os.makedirs(img_dir)
f_name = '{}/{}_{}.png'.format(img_dir, MEASURED_QUBIT_IDX, MEASURED_QUBIT_2ND_IDX)
yhat = savgol_filter(loss_curve, 7, 1)
plt.plot(range(len(loss_curve)), loss_curve, label='loss')
plt.plot(range(len(accuracy_curve_)), accuracy_curve, label='accuracy')
plt.plot(range(len(accuracy_curve_)), yhat, label='trend')
plt.legend()
plt.title(title)
plt.savefig(f_name)


