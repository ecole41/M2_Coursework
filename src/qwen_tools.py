import torch
from tqdm import tqdm 
import numpy as np
from preprocessor import reverse_preprocessing

def generate_predictions(test_texts, model, tokenizer, device, decimal_places=3, num_sys=6):
    """
    Function to generate predictions for multiple systems.

    Parameters:
    - test_texts (list): The test texts for each system.
    - model (transformers.Model): The model used for generation.
    - tokenizer (transformers.Tokenizer): The tokenizer used to process text.
    - device (torch.device): The device to run the model on.
    - decimal_places (int): Number of decimal places in the sequence (default: 3).
    - num_sys (int): Number of systems (default: 6).

    Returns:
    - torch.Tensor: The generated predictions stacked into a tensor.
    """
    # Tokenize the test texts
    tokenized_test = [
        tokenizer(test_texts[sys], return_tensors="pt")["input_ids"].tolist()[0] for sys in range(len(test_texts))
    ]

    # Convert tokenized text to tensors and move to device
    tokenized_tensors_test = torch.stack([torch.tensor(seq, dtype=torch.long).to(device) for seq in tokenized_test])

    tokens_per_time_point = (6+2*decimal_places)
    training_token_id = 80*tokens_per_time_point # 80 time points, 12 tokens per time point
    max_new_tokens = 20*tokens_per_time_point-1 #239

    # Initialize a list to store predicted tokens
    predicted_tokens_list = []

    model.eval()
    # Create a progress bar for tracking generation process
    with torch.no_grad():
        for sys in tqdm(range(num_sys), desc="Generating Tokens"):
            # Slice the data for the current batch and set the `training_token_id`
            data = tokenized_tensors_test[sys][:training_token_id]
            data = data.unsqueeze(0)
            predicted_trajectories = []

            # Loop for generating 10 different trajectories
            for _ in range(10):
                predicted_tokens_batch = model.generate(data, min_length=1199,max_new_tokens=max_new_tokens)
                predicted_tokens_batch = predicted_tokens_batch.squeeze(0)  
                predicted_trajectories.append(predicted_tokens_batch)

        
            predicted_tokens_list.append(predicted_trajectories)

            # Clear GPU cache (for memory management)
            # torch.mps.empty_cache()
            # torch.cuda.empty_cache()

    # Convert the list of predicted tokens into a tensor
    all_predicted_tokens = torch.stack([torch.stack(trajectories) for trajectories in predicted_tokens_list])

    return all_predicted_tokens

def convert_tokens(predicted_tokens, tokenizer):
    """
    This function convert the predicted tokens into arrays of trajectories
    """
    decoded_predictions = []
    for sys in range(len(predicted_tokens)):
        predicted_tokens_sys = predicted_tokens[sys]
        decoded_sys_predictions = [
            tokenizer.decode(tokens, skip_special_tokens=True) for tokens in predicted_tokens_sys
        ]
        decoded_predictions.append(decoded_sys_predictions)

    predicted_prey = []
    predicted_predator = []
    for sys in range(len(decoded_predictions)):
        preys = []
        predators = []
        for traj in range(len(decoded_predictions[sys])):
            prey, pred = reverse_preprocessing(decoded_predictions[sys][traj])
            preys.append(prey)
            predators.append(pred)
        predicted_prey.append(preys)
        predicted_predator.append(predators)
    return predicted_prey, predicted_predator

def mse(pred, true):
    return np.mean((np.array(pred) - np.array(true))**2)

def tot_mse(predicted_prey, predicted_predator, trajectories, norm_factor):
    """
    This function calculates the total MSE for all of the systems
    """
    mse_prey = 0
    mse_predator = 0
    num_sys = len(predicted_prey)
    for sys in range(num_sys):
        orig_sys= sys+900

        target_length = len(trajectories[orig_sys])
        for i in range(len(predicted_prey[sys])):
            # Enforce length 1199 for each prey and predator array
            predicted_prey[sys][i] = np.pad(predicted_prey[sys][i], (0, target_length - len(predicted_prey[sys][i])), 'constant', constant_values=0)[:target_length]
            predicted_predator[sys][i] = np.pad(predicted_predator[sys][i], (0, target_length - len(predicted_predator[sys][i])), 'constant', constant_values=0)[:target_length]

        prey_mean = np.mean(predicted_prey[sys], axis=0)
        predator_mean = np.mean(predicted_predator[sys], axis=0)

        mse_prey += mse(prey_mean[80:], trajectories[orig_sys][80:,0]/norm_factor)
        mse_predator += mse(predator_mean[80:], trajectories[orig_sys][80:,1]/norm_factor)
    return mse_prey, mse_predator