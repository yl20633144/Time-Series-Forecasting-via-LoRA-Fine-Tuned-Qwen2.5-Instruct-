import h5py
import numpy as np
from transformers import AutoTokenizer

# Convert the data to LLMTIME format
def convert_to_llmtime(data_array, alpha, decimal_places):
    """
    Convert a dataset of predator-prey time series into LLMTIME format.

    Parameters:
    data_array: NumPy array 
        Containing time series data of shape (N, T, 2),
                  where N is the number of systems, T is the number of time points,
                  and the last dimension represents (prey, predator).

    alpha: float
        Scaling factor computed from the training set.

    decimal_places: int
        Number of decimal places to round the scaled values.

    Returns:
    - A list of LLMTIME-formatted strings, where each string represents a time series.
    """
    text_list = []
    for i in data_array:
        #Each system has a shape of (100, 2), each row corresponds to a time point and the columns are the prey and predator populations
        time_steps = []
        for j in i:
            #Scaling the data for each (prey,predator) value
            values = [round(j[0] / alpha, decimal_places), round(j[1] * alpha, decimal_places)]
            # Join values of a timestep with commas
            step_str = ",".join(map(str, values))
            time_steps.append(step_str)
        # Use semicolon ";" to separate different timesteps in the same trajectory
        llmtime_seq = ";".join(time_steps)
        text_list.append(llmtime_seq)
    return text_list

def load_and_preprocess(file_path, decimal_places, max_target_value):
    """
    Load and preprocess the predator-prey dataset from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing the dataset.
    decimal_places : int
        Number of decimal places to round the scaled values.
    max_target_value : float
        Maximum target value for scaling.

    Returns
    -------
    train_texts : list of str
        Preprocessed training data sequences in LLMTIME format.
    val_texts : list of str
        Preprocessed validation data sequences in LLMTIME format.
    test_texts : list of str
        Preprocessed test data sequences in LLMTIME format.
    """

    with h5py.File(file_path, "r") as f:
        # Access the full dataset
        trajectories = f["trajectories"][:]
        

        # Example from the sheet of accessing a single trajectory (for the first 50 points):
        # system_id = 0  # First system
        # prey = trajectories[system_id, :50, 0]
        # predator = trajectories[system_id, :50, 1]
        # times = time_points[:50]

    total_number = trajectories.shape[0]  # 1000

    # ---Spilt dataset into train, validation, and test sets---
    # Generate a random permutation of indices for all systems
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(total_number) 

    # Compute split indices for ratio 7 : 1.5 : 1.5 (sums to 10)
    total_ratio = 10.0
    train_end = int(total_number * (7.0 / total_ratio))       # ~70%
    val_end = train_end + int(total_number * (1.5 / total_ratio))  # ~15%
    # remainder ~15% for test

    # Slice the permuted indices for each split
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Gather data for each split
    train_data = trajectories[train_indices]
    val_data = trajectories[val_indices]
    test_data = trajectories[test_indices]
    #---End of data split---

    # Compute alpha based on training set only. Set 95% of the values in the training set within 0-max_target_value
    train_q95 = np.percentile(train_data,95)
    alpha = train_q95 / max_target_value 


    train_texts = convert_to_llmtime(train_data, alpha, decimal_places)
    val_texts = convert_to_llmtime(val_data, alpha, decimal_places)
    test_texts = convert_to_llmtime(test_data, alpha, decimal_places)

    return train_texts, val_texts, test_texts

