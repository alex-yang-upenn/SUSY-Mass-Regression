"""
Module Name: ROOT_utils

Description:
    This module provides helpful functions to work with the ROOT package and extract data
    from ROOT files.

Usage:
Author:
Date:
License:
"""
import ROOT


def extract_event_features(tree, decay_chain, MET_ids, Lepton_ids):
    """
    Extracts event features. Filters out events with any |eta| > 3.4.
    
    Args:
        tree (ROOT.TTree): 
        decay_chain (str): 
            The prefix for the decay chain to extract features from. This string will be concatenated with the
            suffixes "Id", "Pt", "Eta", and "Phi" to access the corresponding branches
        MET_ids (set of int):
            Set of IDs that indicate MET particles
        Lepton_ids 
            Set of IDs that indicate Lepton particles
    
    Returns:
        list:
            Features for a single event, as described at the top of the file, with dimensions (p, 6)
    """
    features = []

    Id = list(getattr(tree, decay_chain + "Id"))
    Pt = list(getattr(tree, decay_chain + "Pt"))
    Eta = list(getattr(tree, decay_chain + "Eta"))
    Phi = list(getattr(tree, decay_chain + "Phi"))
    
    for j in range(len(Id)):
        if abs(Eta[j]) > 3.4:
            return None  # Drops entire event
        
        if Id[j] in MET_ids:
            one_hot = [1, 0, 0]
            Eta[j] = 0
        elif Id[j] in Lepton_ids:
            one_hot = [0, 1, 0]
        else:
            one_hot = [0, 0, 1]
        
        features.append([Pt[j], Eta[j], Phi[j]] + one_hot)
    
    return features


def create_lorentz_vector(pt, eta, phi, mass):
    """
    Args:
        pt (float):
        eta (float):
        phi (float):
        mass (float): GeV / c^2

    Returns:
        ROOT.TLorentzVector: Lorentz vector from pt, eta, phi, mass
    """
    vec = ROOT.TLorentzVector()
    vec.SetPtEtaPhiM(pt, eta, phi, mass)
    return vec



def single_entry_sum_particles(tree, branches, truthM, truthID, MET_ids=[]):
    """
    For a single entry in a TTree, iterates over every given branch to get particles' lorentz vectors, then 
    sums the vectors. Excludes particles with id contained in MET_ids

    Args:
        tree (ROOT.TTree):
            Contains particle information with branches for Id, Pt, Eta, and Phi
        branches (list of str): 
            List of branch name prefixes. Each prefix should have the corresponding branches: prefix+'Id',
            prefix+'Pt', prefix+'Eta', prefix+'Phi'
        truthM (list of float):
            List of true mass values corresponding to truthID indices
        truthID (list of int):
            List of particle IDs that map to the truthM masses
        MET_ids (list of int, optional):
            List of Missing Transverse Energy (MET) IDs to exclude from feature extraction

    Returns:
        TLorentzVector: Sum of all particles' lorentz vectors
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

