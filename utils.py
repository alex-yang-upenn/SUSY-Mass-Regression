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

def extract_features_particles_list(tree, branches, truthM, truthID, MET_ids=[]):
    features = []
    for name in branches:
        Id = list(getattr(tree, name + "Id"))
        Pt = list(getattr(tree, name + "Pt"))
        Eta = list(getattr(tree, name + "Eta"))
        Phi = list(getattr(tree, name + "Phi"))
        for j in range(len(Id)):
            if (Id[j] in MET_ids):
                continue
            mass = truthM[truthID.index(Id[j])]
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
