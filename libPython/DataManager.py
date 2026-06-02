import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from Particles import Particle, Muon, Electron, Jet
from itertools import permutations

def getMuons(evt):
    muons = []
    muons_zip = zip(evt.MuonPtColl,
                    evt.MuonEtaColl,
                    evt.MuonPhiColl,
                    evt.MuonMassColl,
                    evt.MuonChargeColl,
                    evt.MuonLabelColl)
    for pt, eta, phi, mass, charge, label in muons_zip:
        thisMuon = Muon(pt, eta, phi, mass)
        thisMuon.SetCharge(charge)
        thisMuon.SetParticleLabel(label)
        muons.append(thisMuon)
    return muons

def getElectrons(evt):
    electrons = []
    electrons_zip = zip(evt.ElectronPtColl,
                        evt.ElectronEtaColl,
                        evt.ElectronPhiColl,
                        evt.ElectronMassColl,
                        evt.ElectronChargeColl,
                        evt.ElectronLabelColl)
    for pt, eta, phi, mass, charge, label in electrons_zip:
        thisElectron = Electron(pt, eta, phi, mass)
        thisElectron.SetCharge(charge)
        thisElectron.SetParticleLabel(label)
        electrons.append(thisElectron)
    return electrons

def getJets(evt):
    jets = []
    jets_zip = zip(evt.JetPtColl,
                   evt.JetEtaColl,
                   evt.JetPhiColl,
                   evt.JetMassColl,
                   evt.JetChargeColl,
                   evt.JetBtagScoreColl,
                   evt.JetIsBtaggedColl,
                   evt.JetLabelColl)
    for pt, eta, phi, mass, charge, btagScore, isBtagged, label in jets_zip:
        thisJet = Jet(pt, eta, phi, mass)
        thisJet.SetCharge(charge)
        thisJet.SetBtagScore(btagScore)
        thisJet.SetBtagging(isBtagged)
        thisJet.SetParticleLabel(label)
        jets.append(thisJet)

    bjets = list(filter(lambda jet: jet.IsBtagged(), jets))
    return jets, bjets

def getMETv(evt):
    out = Particle(evt.METvPt, 0., evt.METvPhi, 0.)
    out.SetParticleTypeToMETv()
    return out

def getNodeTensors(objects):
    nodeList = []
    for obj in objects:
        nodeList.append([obj.E(), obj.Px(), obj.Py(), obj.Pz(), 
                         obj.MuonCharge(), obj.ElectronCharge(), obj.JetCharge(), 
                         obj.BtagScore()])
    return torch.tensor(nodeList, dtype=torch.float)

def getParticleTypes(objects):
    particleTypes = []
    for obj in objects:
        particleTypes.append(obj.GetParticleType())
    return particleTypes

def getParticleLabels(objects):
    particleLabels = []
    for obj in objects:
        particleLabels.append(obj.GetParticleLabel())
    return torch.tensor(particleLabels, dtype=torch.long)

def getEdgeTensors(objects, edgeAttributeList):
    edgeIndex = []
    edgeAttributes = []
    for (i, j) in permutations(range(len(objects)), 2):
        obj_i = objects[i]
        obj_j = objects[j]
        edgeIndex.append([i, j])
        thisAttributes = []
        if edgeAttributeList is not None:
            if "deltaR" in edgeAttributeList: thisAttributes.append(obj_i.DeltaR(obj_j))
            if "invMass" in edgeAttributeList: thisAttributes.append((obj_i+obj_j).M())
        edgeAttributes.append(thisAttributes)
        
    return (torch.tensor(edgeIndex, dtype=torch.long), torch.tensor(edgeAttributes, dtype=torch.float))

def evtToGraph(objects, labels, edgeAttributeList, returnParticleTypes):
    x = getNodeTensors(objects)
    edgeIndex, edgeAttributes = getEdgeTensors(objects, edgeAttributeList)
    if edgeAttributeList is None: edgeAttributes = None
    if returnParticleTypes:
        particleTypes = getParticleTypes(objects)
        return Data(x=x, y=labels,
                    edge_index=edgeIndex.t().contiguous(),
                    edge_attribute=edgeAttributes,
                    particleTypes=particleTypes)
    else:
        return Data(x=x, y=labels,
                    edge_index=edgeIndex.t().contiguous(),
                    edge_attribute=edgeAttributes)

def rtfileToDataList(rtfile, edgeAttributeList=None, returnParticleTypes=False, isSignal=None, maxSize=-1):
    dataList = []
    for evt in rtfile.Events:
        muons = getMuons(evt)
        electrons = getElectrons(evt)
        jets, _ = getJets(evt)
        METv = getMETv(evt)
        
        objects = muons+electrons+jets+[METv]
        data = evtToGraph(objects, labels=int(isSignal), edgeAttributeList=edgeAttributeList, returnParticleTypes=returnParticleTypes)
        dataList.append(data)
    
        if len(dataList) == maxSize: break
    print(f"@@@@ no. of dataList ends with {len(dataList)}")
    
    return dataList 


class ArrayDataset(Dataset):
    def __init__(self, sample):
        super(ArrayDataset, self).__init__()
        self.features = sample.iloc[:, :-1].to_numpy()
        self.labels = sample.iloc[:, -1:].to_numpy()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.features[idx])
        y = torch.LongTensor(self.labels[idx])
        return (X, y)


class GraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(GraphDataset, self).__init__("./tmp/data")
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)
        

