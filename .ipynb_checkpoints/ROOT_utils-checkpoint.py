import ROOT


def create_lorentz_vector(pt, eta, phi, mass):
    vec = ROOT.TLorentzVector()
    vec.SetPtEtaPhiM(pt, eta, phi, mass)
    return vec

def single_entry_sum_particles(tree, branches, truthM, truthID, MET_ids=[]):
    """
    Sums the lorentz vectors for each particle, excluding MET particles,

    Parameters:
    -----------
    tree : ROOT.TTree
        Contains particle information with branches for Id, Pt, Eta, and Phi
    branches : list of str
        List of branch name prefixes. Each prefix should have corresponding
        branches - prefix+'Id', prefix+'Pt', prefix+'Eta', prefix+'Phi'
    truthM : list
        List of true mass values corresponding to truthID indices
    truthID : list
        List of particle IDs that map to the truthM masses
    MET_ids : list, optional
        List of Missing Transverse Energy (MET) IDs to exclude from feature extraction

    Returns:
    --------
    TLorentzVector
        Vector sum of all particles in the current TTree entry
    """
    sum_vec = ROOT.TLorentzVector() 
    
    for name in branches:
        Id = list(getattr(tree, name + "Id"))
        Pt = list(getattr(tree, name + "Pt"))
        Eta = list(getattr(tree, name + "Eta"))
        Phi = list(getattr(tree, name + "Phi"))
        for j in range(len(Id)):
            if (Id[j] in MET_ids):
                continue
            mass = truthM[truthID.index(Id[j])]
            vec = create_lorentz_vector(Pt[j], Eta[j], Phi[j], mass)
            sum_vec += vec
    
    return sum_vec

def single_entry_extract_features(tree, branches, truthM, truthID, MET_ids=[], clip_eta=False):
    """
    Extracts particle features, excluding MET particles, and optionally filtering out events with extreme eta
    
    Parameters:
    -----------
    tree : ROOT.TTree
        Contains particle information with branches for Id, Pt, Eta, and Phi
    branches : list of str
        List of branch name prefixes. Each prefix should have corresponding
        branches - prefix+'Id', prefix+'Pt', prefix+'Eta', prefix+'Phi'
    truthM : list
        List of true mass values corresponding to truthID indices
    truthID : list
        List of particle IDs that map to the truthM masses
    MET_ids : list, optional
        List of Missing Transverse Energy (MET) IDs to exclude from feature extraction
    clip_eta : bool, optional
        If True, returns empty list for particles with |eta| > 3.4
        
    Returns:
    --------
    list
        Flattened list of features in the order [pt, eta, phi, mass] for each particle
    """

    features = []
    for name in branches:
        Id = list(getattr(tree, name + "Id"))
        Pt = list(getattr(tree, name + "Pt"))
        Eta = list(getattr(tree, name + "Eta"))
        Phi = list(getattr(tree, name + "Phi"))
        
        for j in range(len(Id)):
            if (Id[j] in MET_ids):
                continue
            if (clip_eta and abs(Eta[j]) > 3.4):
                return []   # Aggressive strategy: Drop the entire event if any particle has Eta > 3.4 
            mass = truthM[truthID.index(Id[j])]  # Find index of ID, access corresponding mass value at that index
            features += [Pt[j], Eta[j], Phi[j], mass]
    
    return features