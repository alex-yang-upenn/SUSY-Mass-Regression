import ROOT

import numpy as np

import os
import sys

import helper


DATA_DIRECTORY = "MassRegressionData"
OUTPUT_IMAGE_DIRECTORY = "DirectSummation_Naive_Graphs"

for name in os.listdir(DATA_DIRECTORY):
    # Access ROOT file
    print(f"Processing file {name}")
    filename = os.path.join(DATA_DIRECTORY, name)
    file = ROOT.TFile.Open(filename)

    # Collect "info" branch
    info_tree = file.Get("info")
    info_tree.GetEntry(0)
    truthM = list(getattr(info_tree, "truthM"))
    truthID = list(getattr(info_tree, "truthId"))

    # Iterate through "test" branch
    test_tree = file.Get("test")

    MassCalculated = []
    TrueMassCalculated = []
    MassTarget = []
    for entryNum in range(0, test_tree.GetEntries()):
        test_tree.GetEntry(entryNum)

        particle_sum_vec = helper.vector_sum_particles_list(
            tree=test_tree, 
            branches=["P1"],
            truthM=truthM,
            truthID=truthID,
            MET_ids=[12]
        )
        MassCalculated.append(particle_sum_vec.M())

        true_particle_sum_vec = helper.vector_sum_particles_list(
            tree=test_tree, 
            branches=["P1"],
            truthM=truthM,
            truthID=truthID,
        )
        TrueMassCalculated.append(true_particle_sum_vec.M())
        
        T1Mass = getattr(test_tree, "T1M")
        MassTarget.append(T1Mass)
    
    # Close file
    file.Close()

    # Mass plot
    helper.create_1var_histogram_with_marker(
        data=MassCalculated,
        data_label="X Mass Distribution (calculated without neutrino)",
        marker=np.array(MassTarget).mean(),
        marker_label="X True Mass",
        title=f"Mass Regression, {name}",
        x_label="Mass (GeV / c^2)",
        filename=f"{OUTPUT_IMAGE_DIRECTORY}/{name.replace('.root', '')}.png"
    )

    helper.create_1var_histogram_with_marker(
        data=TrueMassCalculated,
        data_label="X Mass Distribution (calculated with all particles)",
        marker=np.array(MassTarget).mean(),
        marker_label="X True Mass",
        title=f"Mass Regression, {name}",
        x_label="Mass (GeV / c^2)",
        filename=f"{OUTPUT_IMAGE_DIRECTORY}/{name.replace('.root', '')}_verify.png"
    )