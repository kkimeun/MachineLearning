from ROOT import TLorentzVector

class Jet(TLorentzVector):
    def __init__(self, pt, eta, phi, mass):
        TLorentzVector.__init__(self)
        self.SetPtEtaPhiM(pt, eta, phi, mass)
        self.chEmEF = -1.
        self.chHEF = -1.
        self.neEmEF = -1.
        self.neHEF = -1.
        self.muEF = -1.
        self.btagDeepFlavB = -1.
        self.btagDeepFlavQG = -1.

    def SetEnergyFractions(self, chEmEF, chHEF, neEmEF, neHEF, muEF):
        self.chEmEF = chEmEF
        self.chHEF = chHEF
        self.neEmEF = neEmEF
        self.neHEF = neHEF
        self.muEF = muEF

    def SetBtagDeepFlavScores(self, btagDeepFlavB, btagDeepFlavQG):
        self.btagDeepFlavB = btagDeepFlavB
        self.btagDeepFlavQG = btagDeepFlavQG

    def GetChargedEMFraction(self):
        return self.chEmEF

    def GetChargedHadronicFraction(self):
        return self.chHEF

    def GetNeutralEMFraction(self):
        return self.neEmEF

    def GetNeutralHadronicFraction(self):
        return self.neHEF

    def GetMuonEnergyFraction(self):
        return self.muEF

    def GetBtagDeepFlavB(self):
        return self.btagDeepFlavB

    def GetBtagDeepFlavQG(self):
        return self.btagDeepFlavQG
