import matplotlib.pyplot as plt
import numpy as np

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

def reformat_data(X):
    """
    Reformat the sample dataset for GNN input. Completes MET to a full particle (0 Eta, 0 Mass), then
    reshapes to 2D samples (4 Features x 4 Particles).
    """
    X_graphical = np.zeros((len(X), 16), dtype=np.float32)
    X_graphical[:, :12] = X[:, :12]
    X_graphical[:, 14] = X[:, 13]
    X_graphical = X_graphical.reshape(len(X), 4, 4)
    return X_graphical

def create_1var_histogram_with_marker(data, data_label, marker, marker_label, title, x_label, filename):
    """
    Creates a histogram from one dataset with a vertical marker line and saves it to a file. Uses 200 bins
    """
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

def create_2var_histogram_with_marker(data1, data_label1, data2, data_label2, marker, marker_label, 
                                      title, x_label, filename):
    """
    Creates a histogram comparing two datasets with a vertical marker line and saves it to a file. Uses 200 bins.
    Histograms are semi-transparent (alpha=0.5) to show overlap. First dataset is blue, second is green
    """
    plt.style.use('default')

    # Process data
    npData1 = np.array(data1)
    npData2 = np.array(data2)
    max_val = max(npData1.max(), npData2.max())

    # Create figure 
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot both histograms with transparency
    n1, bins1, patches1 = ax.hist(npData1, bins=200, range=(0, max_val), 
                                 alpha=0.5, edgecolor='black', color="blue", label=data_label1)
    n2, bins2, patches2 = ax.hist(npData2, bins=200, range=(0, max_val), 
                                 alpha=0.5, edgecolor='black', color="green", label=data_label2)
    ax.axvline(x=marker, color='red', linestyle='--', linewidth=2, label=marker_label)

    # Add labels and format axes
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_xticks(np.linspace(0, max_val, 6))
    ax.set_xticklabels([f"{x:.2e}" for x in np.linspace(0, max_val, 6)])

    ax.set_ylabel("Frequency", fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename)
    plt.close(fig)