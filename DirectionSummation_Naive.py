import ROOT
import numpy as np

import os
import sys
import random

import utils


# Parameters
DATA_DIRECTORY = "MassRegressionData"
OUTPUT_IMAGE_DIRECTORY = "DirectSummation_Naive_Graphs"

os.makedirs(OUTPUT_IMAGE_DIRECTORY, exist_ok=True)

random.seed(42)


def main():
    rootFiles = random.sample(os.listdir(DATA_DIRECTORY), 5)
    for name in rootFiles:
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

        NaiveMassCalculated = []
        TrueMassCalculated = []
        MassTarget = []
        for entryNum in range(0, test_tree.GetEntries()):
            test_tree.GetEntry(entryNum)

            particle_sum_vec = utils.vector_sum_particles_list(
                tree=test_tree, 
                branches=["P1"],
                truthM=truthM,
                truthID=truthID,
                MET_ids=[12]
            )
            METPt = getattr(test_tree, "METPt")
            METPhi = getattr(test_tree, "METPhi")
            particle_sum_vec += utils.create_lorentz_vector(METPt, 0, METPhi, 0)
            NaiveMassCalculated.append(particle_sum_vec.M())

            true_particle_sum_vec = utils.vector_sum_particles_list(
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

        # Plot for summation with MET masked
        utils.create_1var_histogram_with_marker(
            data=NaiveMassCalculated,
            data_label="X Mass Distribution (calculated without neutrino)",
            marker=np.array(MassTarget).mean(),
            marker_label="X True Mass",
            title=f"Mass Regression, {name}",
            x_label="Mass (GeV / c^2)",
            filename=f"{OUTPUT_IMAGE_DIRECTORY}/{name.replace('.root', '')}.png"
        )

        # Plot for summation with full information
        utils.create_1var_histogram_with_marker(
            data=TrueMassCalculated,
            data_label="X Mass Distribution (calculated with all particles)",
            marker=np.array(MassTarget).mean(),
            marker_label="X True Mass",
            title=f"Mass Regression, {name}",
            x_label="Mass (GeV / c^2)",
            filename=f"{OUTPUT_IMAGE_DIRECTORY}/{name.replace('.root', '')}_verify.png"
        )

if __name__ == "__main__":
    main()