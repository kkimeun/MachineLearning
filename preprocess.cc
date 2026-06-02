void preprocess() {
    // data locations
    //const TString dataPath = "/gv0/Users/choij/NanoAOD/NanoAOD/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv11/231110_024454/0000/";
    const TString dataPath = "/gv0/Users/choij/NanoAOD/NanoAOD/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv11/231110_024308/0000/";

    auto chain = TChain("Events");
    for (unsigned int i = 1; i < 51; i++)
        chain.Add(dataPath+"NANOAOD_"+TString::Itoa(i, 10)+".root");

    cout << chain.GetEntries() << endl;

    // declare variables
    UInt_t nJet;                       chain.SetBranchAddress("nJet", &nJet);
    Float_t Jet_pt[30];                 chain.SetBranchAddress("Jet_pt", &Jet_pt);
    Float_t Jet_eta[30];                chain.SetBranchAddress("Jet_eta", &Jet_eta);
    Float_t Jet_phi[30];                chain.SetBranchAddress("Jet_phi", &Jet_phi);
    Float_t Jet_mass[30];               chain.SetBranchAddress("Jet_mass", &Jet_mass);
    Float_t Jet_btagDeepFlavB[30];      chain.SetBranchAddress("Jet_btagDeepFlavB", &Jet_btagDeepFlavB);
    Float_t Jet_btagDeepFlavQG[30];     chain.SetBranchAddress("Jet_btagDeepFlavQG", &Jet_btagDeepFlavQG);
    Float_t Jet_chEmEF[30];             chain.SetBranchAddress("Jet_chEmEF", &Jet_chEmEF);
    Float_t Jet_chHEF[30];              chain.SetBranchAddress("Jet_chHEF", &Jet_chHEF);
    Float_t Jet_neEmEF[30];             chain.SetBranchAddress("Jet_neEmEF", &Jet_neEmEF);
    Float_t Jet_neHEF[30];              chain.SetBranchAddress("Jet_neHEF", &Jet_neHEF);
    Float_t Jet_muEF[30];               chain.SetBranchAddress("Jet_muEF", &Jet_muEF);
    Float_t LHE_HT;                     chain.SetBranchAddress("LHE_HT", &LHE_HT);
    Int_t Jet_nElectrons[30];           chain.SetBranchAddress("Jet_nElectrons", &Jet_nElectrons);
    Int_t Jet_nMuons[30];               chain.SetBranchAddress("Jet_nMuons", &Jet_nMuons);
    Int_t Jet_nSVs[30];                 chain.SetBranchAddress("Jet_nSVs", &Jet_nSVs); 
    Int_t Jet_hadronFlavour[30];        chain.SetBranchAddress("Jet_hadronFlavour", &Jet_hadronFlavour);
    Int_t Jet_partonFlavour[30];        chain.SetBranchAddress("Jet_partonFlavour", &Jet_partonFlavour);

    for (unsigned int fIdx=0; fIdx < 50; fIdx++) {
        cout << "@@@@ processing " << fIdx << "th file..." << endl; 
        TFile *f = new TFile("NANOAOD_"+TString::Itoa(fIdx, 10)+".root", "recreate");
        TTree *tr = new TTree("Events", "Events");
        tr->Branch("nJet", &nJet);
        tr->Branch("Jet_pt", Jet_pt, "Jet_pt[nJet]/F");
        tr->Branch("Jet_eta", Jet_eta, "Jet_eta[nJet]/F");
        tr->Branch("Jet_phi", Jet_phi, "Jet_phi[nJet]/F");
        tr->Branch("Jet_mass", Jet_mass, "Jet_mass[nJet]/F");
        tr->Branch("Jet_btagDeepFlavB", Jet_btagDeepFlavB, "Jet_btagDeepFlavB[nJet]/F");
        tr->Branch("Jet_btagDeepFlavQG", Jet_btagDeepFlavQG, "Jet_btagDeepFlavQG[nJet]/F");
        tr->Branch("Jet_chEmEF", Jet_chEmEF, "Jet_chEmEF[nJet]/F");
        tr->Branch("Jet_chHEF", Jet_chHEF, "Jet_chHEF[nJet]/F");
        tr->Branch("Jet_neEmEF", Jet_neEmEF, "Jet_neEmEF[nJet]/F");
        tr->Branch("Jet_neHEF", Jet_neHEF, "Jet_neHEF[nJet]/F");
        tr->Branch("Jet_muEF", Jet_muEF, "Jet_muEF[nJet]/F");
        tr->Branch("Jet_nElectrons", Jet_nElectrons, "Jet_nElectrons[nJet]/F");
        tr->Branch("Jet_nMuons", Jet_nMuons, "Jet_nMuons[nJet]/F");
        tr->Branch("Jet_nSVs", Jet_nSVs, "Jet_nSVs[nJet]/F"); 
        tr->Branch("Jet_hadronFlavour", Jet_hadronFlavour, "Jet_hadronFlavour[nJet]/F");
        tr->Branch("Jet_partonFlavour", Jet_partonFlavour, "Jet_partonFlavour[nJet]/F");

        for (unsigned int evtIdx = fIdx*25000; evtIdx < (fIdx+1)*25000; evtIdx++) {
            chain.GetEvent(evtIdx);
            if (LHE_HT < 200 || LHE_HT > 400) continue;
            tr->Fill();
        }
        f->cd();
        tr->Write();
        f->Write();
    }
}
