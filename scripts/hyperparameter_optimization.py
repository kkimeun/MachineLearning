import os, sys; sys.path.append("./libPython")
import argparse
import ROOT

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Sequential, ReLU, Linear, Dropout
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.utils import dropout_edge
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import shuffle
from DataManager import rtfileToDataList, GraphDataset

import matplotlib.pyplot as plt
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument("--dropout_p", required=True, type=float, help="edge dropout_p")
parser.add_argument("--pilot", action="store_true", default=False, help="only use 10K events for signal and background")
args = parser.parse_args()

maxSize = 10000 if args.pilot else -1
SIGNAL = "MHc-130_MA-90"
BACKGROUND = "ttZ"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"@@@@ Using DEVICE {DEVICE}")

## load dataset
rtSig = ROOT.TFile.Open(f"./DATA/{SIGNAL}.root")
rtBkg = ROOT.TFile.Open(f"./DATA/{BACKGROUND}.root")

sigDataList = shuffle(rtfileToDataList(rtSig, isSignal=True,  maxSize=maxSize), random_state=42)
bkgDataList = shuffle(rtfileToDataList(rtBkg, isSignal=False, maxSize=maxSize), random_state=42)
dataList = shuffle(sigDataList+bkgDataList, random_state=42)

trainset = GraphDataset(dataList[:int(len(dataList)*0.6)])
validset = GraphDataset(dataList[int(len(dataList)*0.6):int(len(dataList)*0.7)])
testset  = GraphDataset(dataList[int(len(dataList)*0.7):])

trainLoader = DataLoader(trainset, batch_size=512, pin_memory=True, shuffle=True)
validLoader = DataLoader(validset, batch_size=512, pin_memory=True, shuffle=False)
testLoader = DataLoader(testset, batch_size=512, pin_memory=True, shuffle=False)

## reload dataLists for visualization
sigDataList = shuffle(rtfileToDataList(rtSig, isSignal=True, returnParticleTypes=True, maxSize=10000), random_state=42)
bkgDataList = shuffle(rtfileToDataList(rtBkg, isSignal=False, returnParticleTypes=True, maxSize=10000), random_state=42)

rtSig.Close()
rtBkg.Close()

## construct ParticleNet and define helper functions
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(aggr="mean")
        self.mlp = Sequential(
                Linear(2*in_channels, hidden_channels), ReLU(), BatchNorm1d(hidden_channels), Dropout(0.4),
                Linear(out_channels, hidden_channels), ReLU(), BatchNorm1d(hidden_channels), Dropout(0.4),
                Linear(out_channels, hidden_channels), ReLU(), BatchNorm1d(hidden_channels), Dropout(0.4)
                )

    def forward(self, x, edge_index, batch=None):
        return self.propagate(edge_index, x=x, batch=batch)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_p, training, k=4):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.shortcut = Sequential(Linear(in_channels, out_channels), ReLU())
        self.training = training
        self.dropout_p = dropout_p
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index is None: edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = dropout_edge(edge_index, p=self.dropout_p, training=self.training)
        out = super().forward(x, edge_index, batch=batch)
        out += self.shortcut(x)
        return out


class ParticleNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, dropout_p=0.2):
        super(ParticleNet, self).__init__()
        self.gn0 = GraphNorm(in_channels)
        self.bn0 = BatchNorm1d(hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        #self.bn2 = BatchNorm1d(hidden_channels)
        self.conv1 = TransformerConv(in_channels, hidden_channels, dropout_p=dropout_p, edge_dim=edge_dim, training=self.training)
        self.conv2 = DynamicEdgeConv(hidden_channels, hidden_channels, hidden_channels, dropout_p=dropout_p, training=self.training, k=4)
        self.conv3 = DynamicEdgeConv(hidden_channels, hidden_channels, hidden_channels, dropout_p=dropout_p, training=self.training, k=4)
        self.dense1 = Linear(hidden_channels, hidden_channels)
        self.dense2 = Linear(hidden_channels, hidden_channels)
        self.output = Linear(hidden_channels, out_channels)
        self.dropout_p = dropout_p
    
    def forward(self, x, edge_index, edge_attribute=None, batch=None, return_attention_weights=False):
        # convolution layers
        x = self.gn0(x, batch=batch)
        conv1, attention_weights = self.conv1(x, edge_index, edge_attr=edge_attribute, return_attention_weights=True)
        conv1 = F.elu(conv1)
        conv2 = self.conv2(conv1, edge_index, batch=batch)
        conv3 = self.conv3(conv2, edge_index, batch=batch)
        x = conv1+conv2+conv3
        
        # readout layers
        x = global_mean_pool(x, batch=batch)
        
        # dense layers
        x = self.bn0(x)
        x = F.alpha_dropout(x, p=0.4)
        x = F.selu(self.dense1(x))
        x = F.alpha_dropout(x, p=0.4)
        x = F.selu(self.dense2(x))
        x = self.output(x)

        if return_attention_weights:
            return F.log_softmax(x, dim=1), attention_weights
        else:
            return F.log_softmax(x, dim=1)
        

def train(model, optimizer, scheduler):
    model.train()
    
    train_loss = 0.
    for data in trainLoader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, batch=data.batch)
        optimizer.zero_grad()
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.sum().item()
    scheduler.step()
    return train_loss / len(trainLoader.dataset)

def test(model, loader):
    model.eval()
    loss = 0.
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, batch=data.batch)
            pred = out.argmax(dim=1)
            loss += F.cross_entropy(out, data.y).sum().item()
            correct += pred.eq(data.y).sum().item()
    loss /= len(loader.dataset)
    correct /= len(loader.dataset)
    return (loss, correct)


## train model
model = ParticleNet(in_channels=8, hidden_channels=64, out_channels=2, dropout_p=args.dropout_p).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.01,
                                                         step_size_up=3, step_size_down=5,
                                                         cycle_momentum=False)
summaryPath = f"./runs/dropout_p_{str(args.dropout_p).replace('.', 'p')}"
os.makedirs(os.path.dirname(summaryPath), exist_ok=True)
writer = SummaryWriter(summaryPath)
print(model)

for epoch in range(30):
    loss = train(model, optimizer, scheduler)
    trainLoss, trainAcc = test(model, trainLoader)
    validLoss, validAcc = test(model, validLoader)
    writer.add_scalar("Loss/train", trainLoss, epoch)
    writer.add_scalar("Loss/valid", validLoss, epoch)
    writer.add_scalar("Acc/train", trainAcc, epoch)
    writer.add_scalar("Acc/valid", validAcc, epoch)
    
    print(f"[EPOCH {epoch}]\tTrain Acc: {trainAcc*100:.2f}\tTrain Loss: {trainLoss:.4e}")
    print(f"[EPOCH {epoch}]\tValid Acc: {validAcc*100:.2f}\tValid Loss: {validLoss:.4e}\n")
    
writer.flush()
writer.close()

## Make histograms and save as root files
outPath = f"outputs/ex5/ParticleNet_{str(args.dropout_p).replace('.', 'p')}.root"
f = ROOT.TFile.Open(outPath, "recreate")
hSig = ROOT.TH1D(f"mask_sig_{str(args.dropout_p).replace('.', 'p')}", "", 100, 0., 1.)
hBkg = ROOT.TH1D(f"mask_bkg_{str(args.dropout_p).replace('.', 'p')}", "", 100, 0., 1.)

model.to("cpu")
model.eval()
for data in sigDataList:
    with torch.no_grad():
        out, weights = model(data.x, data.edge_index, return_attention_weights=True)
    hSig.Fill(weights[1][0].numpy())
for data in bkgDataList:
    with torch.no_grad():
        out, weights = model(data.x, data.edge_index, return_attention_weights=True)
    hBkg.Fill(weights[1][0].numpy())
f.cd()
hSig.Write()
hBkg.Write()
f.Close()

# draw 5 inputs for both signal and background
def visualize_graph(data, outPath):
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        out, weights = model(data.x, data.edge_index, return_attention_weights=True)

    # edge_mask dict: {(u, v): weights}
    edge_mask_dict = {}
    for (u, v, weight) in zip(weights[0][0].numpy(), weights[0][1].numpy(), weights[1].numpy()):
        if (u == v): continue
        if u > v: u, v = v, u
        edge_mask_dict[(u, v)] = weight.mean()
    #print(edge_mask_dict)

    G = nx.Graph()
    positions = {}
    muons = []
    electrons = []
    jets = []
    bjets = []
    METv = []
    for i, features in enumerate(data.x):
        thisParticle = ROOT.TLorentzVector()
        thisParticle.SetPxPyPzE(features[1], features[2], features[3], features[0])
        positions[i] = (thisParticle.Eta(), thisParticle.Phi())
    for i, particleType in enumerate(data.particleTypes):
        if particleType == "muon": muons.append(i)
        if particleType == "electron": electrons.append(i)
        if particleType == "light-jet": jets.append(i)
        if particleType == "heavy-jet": bjets.append(i)
        if particleType == "METv": METv.append(i)
    G.add_nodes_from(positions.keys())
    G.add_edges_from(edge_mask_dict.keys())
    edge_color = [edge_mask_dict[(u, v)] for u, v, in G.edges()]
    widths = [x*10 for x in edge_color]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, positions, nodelist=muons, node_color="red", label="muon")
    nx.draw_networkx_nodes(G, positions, nodelist=electrons, node_color="blue", label="electron")
    nx.draw_networkx_nodes(G, positions, nodelist=jets, node_color="yellowgreen", label="light-jet")
    nx.draw_networkx_nodes(G, positions, nodelist=bjets, node_color="green", label="heavy-jet")
    nx.draw_networkx_nodes(G, positions, nodelist=METv, node_color="gray", label="MET")
    nx.draw_networkx_edges(G, positions, edgelist=edge_mask_dict.keys(), edge_color=edge_color, alpha=0.5, width=widths, edge_cmap=plt.cm.Blues)
    plt.legend(loc="upper right")
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\phi$")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks([-3., -2., -1., 0., 1., 2., 3])
    plt.yticks([-3.14, -2., -1., 0., 1., 2., 3.14])
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.tight_layout()
    plt.savefig(outPath)
    
for idx, data in enumerate(sigDataList[:10]):
    outPath = f"outputs/ex5/plots/ParticleNet-{str(args.dropout_p).replace('.', 'p')}/signal-{idx}.png"
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    visualize_graph(data, outPath)
    
for idx, data in enumerate(bkgDataList[:10]):
    outPath = f"outputs/ex5/plots/ParticleNet-{str(args.dropout_p).replace('.', 'p')}/background-{idx}.png"
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    visualize_graph(data, outPath)