"""Utilities for working with ROOT files and extracting particle physics data.

This module provides functions to extract event features from ROOT TTrees,
create Lorentz vectors, and sum particle vectors for physics analysis.
"""

import ROOT


def extract_event_features(tree, decay_chains, MET_ids, Lepton_ids, Gluon_ids=None):
    """Extract particle features from ROOT TTree for a single event.

    Extracts particle features (Pt, Eta, Phi) and creates one-hot encodings for
    particle types (MET, Lepton, Gluon, Quark). Filters out events where any
    particle has |eta| > 3.4, as these are outside the detector acceptance.

    Args:
        tree: ROOT.TTree containing particle information with branches for each
            decay chain (suffixed with "Id", "Pt", "Eta", "Phi").
        decay_chains: List of str, branch name prefixes for decay chains to extract.
            Each prefix will be concatenated with "Id", "Pt", "Eta", and "Phi".
        MET_ids: Set or list of int, particle IDs that indicate Missing Transverse
            Energy (MET) particles. Eta values for MET are set to 0.
        Lepton_ids: Set or list of int, particle IDs that indicate lepton particles.
        Gluon_ids: Set or list of int, optional. Particle IDs that indicate gluon
            particles. If None, uses 3-category encoding (MET, Lepton, Quark),
            otherwise uses 4-category encoding (MET, Lepton, Gluon, Quark).

    Returns:
        list or None: List of particle features for the event, where each particle
            is represented as [Pt, Eta, Phi, one_hot_encoding]. Returns None if any
            particle has |eta| > 3.4 (entire event is dropped).
            Shape: (n_particles, 6) for 3-category or (n_particles, 7) for 4-category.

    Example:
        >>> features = extract_event_features(tree, ["P1"], {12}, {11})
        >>> # Returns features like [[pt, eta, phi, 1, 0, 0], ...] for each particle
    """
    if Gluon_ids == None:
        MET = [1, 0, 0]
        Lepton = [0, 1, 0]
        Quark = [0, 0, 1]
    else:
        MET = [1, 0, 0, 0]
        Lepton = [0, 1, 0, 0]
        Gluon = [0, 0, 1, 0]
        Quark = [0, 0, 0, 1]
        Gluon_ids = set()

    features = []

    for decay_chain in decay_chains:
        Id = list(getattr(tree, decay_chain + "Id"))
        Pt = list(getattr(tree, decay_chain + "Pt"))
        Eta = list(getattr(tree, decay_chain + "Eta"))
        Phi = list(getattr(tree, decay_chain + "Phi"))

        for j in range(len(Id)):
            if abs(Eta[j]) > 3.4:
                return None  # Drops entire event

            if Id[j] in MET_ids:
                one_hot = MET
                Eta[j] = 0
            elif Id[j] in Lepton_ids:
                one_hot = Lepton
            elif Id[j] in Gluon_ids:
                one_hot = Gluon
            else:
                one_hot = Quark

            features.append([Pt[j], Eta[j], Phi[j]] + one_hot)

    return features


def create_lorentz_vector(pt, eta, phi, mass):
    """Create a ROOT TLorentzVector from kinematic variables.

    Args:
        pt: Float, transverse momentum in GeV/c.
        eta: Float, pseudorapidity (dimensionless).
        phi: Float, azimuthal angle in radians.
        mass: Float, particle mass in GeV/c^2.

    Returns:
        ROOT.TLorentzVector: Four-vector representation of the particle.

    Example:
        >>> vec = create_lorentz_vector(50.0, 1.2, 0.5, 0.0)
        >>> print(vec.M())  # Prints the mass
    """
    vec = ROOT.TLorentzVector()
    vec.SetPtEtaPhiM(pt, eta, phi, mass)
    return vec


def single_entry_sum_particles(tree, branches, truthM, truthID, MET_ids=[]):
    """Sum Lorentz vectors of all particles in a single TTree entry.

    Iterates over specified branches to extract particle kinematics, creates
    Lorentz vectors using truth mass values, and sums them. Particles with IDs
    in MET_ids are excluded from the sum since MET cannot be directly measured.

    Args:
        tree: ROOT.TTree containing particle information with branches for Id,
            Pt, Eta, and Phi for each branch prefix.
        branches: List of str, branch name prefixes. Each prefix should have
            corresponding branches: prefix+'Id', prefix+'Pt', prefix+'Eta',
            prefix+'Phi'.
        truthM: List of float, true mass values in GeV/c^2 for each particle
            type, indexed by truthID.
        truthID: List of int, particle PDG IDs that correspond to the masses
            in truthM.
        MET_ids: List of int, optional. Particle IDs for Missing Transverse
            Energy to exclude from summation. Defaults to empty list.

    Returns:
        ROOT.TLorentzVector: Sum of all particles' four-vectors, excluding MET.
            Used to compute invariant mass of the system.

    Example:
        >>> sum_vec = single_entry_sum_particles(tree, ["P1"], [125.0], [25], [12])
        >>> print(sum_vec.M())  # Prints invariant mass of the system
    """
    sum_vec = ROOT.TLorentzVector()

    for name in branches:
        Id = list(getattr(tree, name + "Id"))
        Pt = list(getattr(tree, name + "Pt"))
        Eta = list(getattr(tree, name + "Eta"))
        Phi = list(getattr(tree, name + "Phi"))
        for j in range(len(Id)):
            if Id[j] in MET_ids:
                continue
            mass = truthM[truthID.index(Id[j])]
            vec = create_lorentz_vector(Pt[j], Eta[j], Phi[j], mass)
            sum_vec += vec

    return sum_vec
