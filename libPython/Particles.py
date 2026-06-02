from ROOT import TLorentzVector
from ROOT import TMath

# Base class for all objects
class Particle(TLorentzVector):
    def __init__(self, pt, eta, phi, mass):
        super().__init__()
        self.SetPtEtaPhiM(pt, eta, phi, mass)
        self.muonCharge = 0
        self.electronCharge = 0
        self.jetCharge = 0.
        self.btagScore = 0.
        self.isBtagged = False
        self.isFromCH = False
        self.isMETv = False
        
    def MuonCharge(self):
        return self.muonCharge

    def ElectronCharge(self):
        return self.electronCharge

    def JetCharge(self):
        return self.jetCharge

    def BtagScore(self):
        return self.btagScore

    def IsBtagged(self):
        return self.isBtagged

    def MT(self, part):
        dPhi = self.DeltaPhi(part)
        return TMath.Sqrt(2*self.Pt()*part.Pt()*(1.-TMath.Cos(dPhi)))

    def SetParticleLabel(self, label):
        self.isFromCH = label
    
    def SetParticleTypeToMETv(self):
        self.isMETv = True
        
    def GetParticleLabel(self):
        return self.isFromCH
    
    def GetParticleType(self):
        if self.isMETv:
            return "METv"
        else:
            return None

class Muon(Particle):
    def __init__(self, pt, eta, phi, mass):
        super().__init__(pt, eta, phi, mass)

    def SetCharge(self, charge):
        self.muonCharge = charge

    def SetLeptonType(self, lepType):
        self.lepType = lepType
        
    def GetLeptonType(self):
        return self.lepType

    def GetParticleType(self):
        return "muon"

class Electron(Particle):
    def __init__(self, pt, eta, phi, mass):
        super().__init__(pt, eta, phi, mass)
       
    def SetCharge(self, charge):
        self.electronCharge = charge

    def SetLeptonType(self, lepType):
        self.lepType = lepType
        
    def GetLeptonType(self):
        return self.lepType
    
    def GetParticleType(self):
        return "electron"

class Jet(Particle):
    def __init__(self, pt, eta, phi, mass):
        super().__init__(pt, eta, phi, mass)

    def SetCharge(self, charge):
        self.jetCharge = charge

    def SetBtagScore(self, btagScore):
        self.btagScore = btagScore

    def SetBtagging(self, isBtagged):
        self.isBtagged = isBtagged

    def GetParticleType(self):
        if self.IsBtagged():
            return "heavy-jet"
        else:
            return "light-jet"