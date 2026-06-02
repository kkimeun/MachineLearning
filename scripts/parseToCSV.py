import os
import numpy as np
import pandas as pd
import ROOT
from itertools import combinations
from DataFormats import Jet

ROOT.EnableImplicitMT()

# global variables
maxEvents = 100000
SAMPLE = "QCD"
features = ["j1_energy", "j1_px", "j1_py", "j1_pz", "j1_chEmEF", "j1_chHEF", "j1_neEmEF", "j1_neHEF", "j1_muEF", "j1_btagDeepFlavB", "j1_btagDeepFlavQG",
            "j2_energy", "j2_px", "j2_py", "j2_pz", "j2_chEmEF", "j2_chHEF", "j2_neEmEF", "j2_neHEF", "j2_muEF", "j2_btagDeepFlavB", "j2_btagDeepFlavQG",
            "j3_energy", "j3_px", "j3_py", "j3_pz", "j3_chEmEF", "j3_chHEF", "j3_neEmEF", "j3_neHEF", "j3_muEF", "j3_btagDeepFlavB", "j3_btagDeepFlavQG",
            "j4_energy", "j4_px", "j4_py", "j4_pz", "j4_chEmEF", "j4_chHEF", "j4_neEmEF", "j4_neHEF", "j4_muEF", "j4_btagDeepFlavB", "j4_btagDeepFlavQG",
            "avg_deltaR", "Nj", "HT"]


chain = ROOT.TChain("Events")
chain.Add(f"DATA/{SAMPLE}/PrunedNano_*.root")

# book features to data
data = {}
for feature in features: 
    data[feature] = []

def getJets(event):
    jets = []
    jets_zip = zip(evt.Jet_pt, evt.Jet_eta, evt.Jet_phi, evt.Jet_mass, evt.Jet_chEmEF, evt.Jet_chHEF, evt.Jet_neEmEF, evt.Jet_neHEF, evt.Jet_muEF, evt.Jet_btagDeepFlavB, evt.Jet_btagDeepFlavQG)
    for pt, eta, phi, mass, chEmEF, chHEF, neEmEF, neHEF, muEF, btagB, btagQG in jets_zip:
        thisJet = Jet(pt, eta, phi, mass)
        thisJet.SetEnergyFractions(chEmEF, chHEF, neEmEF, neHEF, muEF)
        thisJet.SetBtagDeepFlavScores(btagB, btagQG)
        jets.append(thisJet)
    return jets


for idx, evt in enumerate(chain, start=1):
    if idx % 1000 == 0: print(f"processing {idx}th event")

    jets = getJets(evt)
    dRjets = []
    for idx1, idx2 in combinations(range(len(jets)), 2):
        dRjets.append(jets[idx1].DeltaR(jets[idx2]))
    avg_deltaR = np.mean(dRjets)
    HT = 0.
    for j in jets: HT += j.Pt()
    
    j1, j2, j3, j4 = jets[0], jets[1], jets[2], jets[3]

    data["j1_energy"].append(j1.E()); data["j1_px"].append(j1.Px()); data["j1_py"].append(j1.Py()); data["j1_pz"].append(j1.Pz());
    data["j1_chEmEF"].append(j1.GetChargedEMFraction())
    data["j1_chHEF"].append(j1.GetChargedHadronicFraction())
    data["j1_neEmEF"].append(j1.GetNeutralEMFraction())
    data["j1_neHEF"].append(j1.GetNeutralHadronicFraction())
    data["j1_muEF"].append(j1.GetMuonEnergyFraction())
    data["j1_btagDeepFlavB"].append(j1.GetBtagDeepFlavB())
    data["j1_btagDeepFlavQG"].append(j1.GetBtagDeepFlavQG())
    data["j2_energy"].append(j2.E()); data["j2_px"].append(j2.Px()); data["j2_py"].append(j2.Py()); data["j2_pz"].append(j2.Pz());
    data["j2_chEmEF"].append(j2.GetChargedEMFraction())
    data["j2_chHEF"].append(j2.GetChargedHadronicFraction())
    data["j2_neEmEF"].append(j2.GetNeutralEMFraction())
    data["j2_neHEF"].append(j2.GetNeutralHadronicFraction())
    data["j2_muEF"].append(j2.GetMuonEnergyFraction())
    data["j2_btagDeepFlavB"].append(j2.GetBtagDeepFlavB())
    data["j2_btagDeepFlavQG"].append(j2.GetBtagDeepFlavQG())
    data["j3_energy"].append(j3.E()); data["j3_px"].append(j3.Px()); data["j3_py"].append(j3.Py()); data["j3_pz"].append(j3.Pz());
    data["j3_chEmEF"].append(j3.GetChargedEMFraction())
    data["j3_chHEF"].append(j3.GetChargedHadronicFraction())
    data["j3_neEmEF"].append(j3.GetNeutralEMFraction())
    data["j3_neHEF"].append(j3.GetNeutralHadronicFraction())
    data["j3_muEF"].append(j3.GetMuonEnergyFraction())
    data["j3_btagDeepFlavB"].append(j3.GetBtagDeepFlavB())
    data["j3_btagDeepFlavQG"].append(j3.GetBtagDeepFlavQG())
    data["j4_energy"].append(j4.E()); data["j4_px"].append(j4.Px()); data["j4_py"].append(j4.Py()); data["j4_pz"].append(j4.Pz());
    data["j4_chEmEF"].append(j4.GetChargedEMFraction())
    data["j4_chHEF"].append(j4.GetChargedHadronicFraction())
    data["j4_neEmEF"].append(j4.GetNeutralEMFraction())
    data["j4_neHEF"].append(j4.GetNeutralHadronicFraction())
    data["j4_muEF"].append(j4.GetMuonEnergyFraction())
    data["j4_btagDeepFlavB"].append(j4.GetBtagDeepFlavB())
    data["j4_btagDeepFlavQG"].append(j4.GetBtagDeepFlavQG())
    data["avg_deltaR"].append(avg_deltaR)
    data["Nj"].append(len(jets))
    data["HT"].append(HT)

    if idx == maxEvents: break

df = pd.DataFrame(data, columns=features)
df.to_csv(f"DATA/{SAMPLE}/sample.csv")


