import matplotlib.pyplot as plt
import numpy as np

import ROOT

def create_lorentz_vector(pt, eta, phi, mass):
    vec = ROOT.TLorentzVector()
    vec.SetPtEtaPhiM(pt, eta, phi, mass)
    return vec

def vector_sum_particles_list(tree, branches, truthM, truthID, MET_ids=[]):
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

def vector_sum_particles(tree, branches):
    sum_vec = ROOT.TLorentzVector() 
    
    for name in branches:
        Mass = getattr(tree, name + "M")
        Pt = getattr(tree, name + "Pt")
        Eta = getattr(tree, name + "Eta")
        Phi = getattr(tree, name + "Phi")
        vec = create_lorentz_vector(Pt, Eta, Phi, Mass)
        sum_vec += vec
    
    return sum_vec

def single_entry_extract_features(tree, branches, truthM, truthID, MET_ids=[], clip_eta=False):
    """
    Extracts particle features, excluding MET particles, and optionally filtering out extreme eta
    
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
                return []
            mass = truthM[truthID.index(Id[j])]  # Find index of ID, access corresponding mass value at that index
            features += [Pt[j], Eta[j], Phi[j], mass]
    
    return features

def create_1var_histogram_with_marker(
    data,
    data_label,
    marker,
    marker_label,
    title,
    x_label,
    filename
):
    plt.style.use('default')

    # Process data
    npData = np.array(data)
    max_val = npData.max()

    # Create figure 
    fig, ax = plt.subplots(figsize=(12, 6))
    n, bins, patches = ax.hist(npData, bins=200, range=(0, max_val), edgecolor='black')
    ax.axvline(x=marker, color='red', linestyle='--', linewidth=2)
      

    # Add labels and format axes
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_xticks(np.linspace(0, max_val, 6))
    ax.set_xticklabels([f'{x:.2e}' for x in np.linspace(0, max_val, 6)])

    ax.set_ylabel("Frequency", fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend([marker_label, data_label], fontsize=10)

    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename)
    plt.close(fig)
