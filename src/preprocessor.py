import numpy as np
import matplotlib.pyplot as plt
import h5py
from transformers import AutoTokenizer

def scale(trajectories , decimal_places):
    max_preys = np.max(trajectories[:, :, 0], axis=1)
    max_predators = np.max(trajectories[:, :, 1], axis=1)
    av_max_prey = np.mean(max_preys)
    av_max_predator = np.mean(max_predators)
    max_value = max(av_max_prey, av_max_predator)
    scaled_trajectories = trajectories / max_value

    # Round each scaled value to x decimal places
    scaled_trajectories = np.round(scaled_trajectories, decimal_places)
    
    return scaled_trajectories , max_value

def convert_string(trajectories, decimal_places): 
    trajectory_strings = []
    n_systems = np.shape(trajectories)[0]
    for sys in range(n_systems):

        predator_vals = trajectories[sys, :, 0]
        prey_vals = trajectories[sys, :, 1]

        paired_vals = [f"{pred:.{decimal_places}f},{prey:.{decimal_places}f}" for pred, prey in zip(predator_vals, prey_vals)]
        trajectory_string = ";".join(paired_vals)

        trajectory_strings.append(trajectory_string)

    return trajectory_strings


def load_and_preprocess(data_path):

    train_size = 800
    with h5py.File(data_path, "r") as f:
        # Access the full dataset
        trajectories = f["trajectories"][:] # Shape: (n_systems, n_time_points, n_species)
    scaled_trajectories , norm_factor = scale(trajectories, 3)
    final_trajects=convert_string(scaled_trajectories, 3)

    train_data = final_trajects[:train_size]
    val_data = final_trajects[train_size:]

    return train_data, val_data , norm_factor
   
