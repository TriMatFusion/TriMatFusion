"""
utils.py (Shared Utilities)
Responsibilities:
- Host functions used by multiple modules (train, test, main).
- e.g., evaluate(), write_results(), model_summary()
"""

import torch
import numpy as np
import csv
import torch.nn.functional as F

def model_summary(model):
    """Prints a summary of the model's parameters."""
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")

    total_params = 0
    num_trainable_params = 0

    for name, param in model.named_parameters():
        p_shape = list(param.size())

        # Use numel() for clarity and directness
        p_count = param.numel()

        line_new = "{:>30}  {:>20} {:>20}".format(name, str(p_shape), f"{p_count:,}") # Use f-string for formatting numbers
        
        print(line_new)
        
        # Accumulate totals within the loop
        total_params += p_count
        if param.requires_grad:
            num_trainable_params += p_count
            
    print("--------------------------------------------------------------------------")
    # Print formatted numbers for readability
    print(f"Total params: {total_params:,}") 
    print(f"Trainable params: {num_trainable_params:,}")
    print(f"Non-trainable params: {total_params - num_trainable_params:,}")
    print("--------------------------------------------------------------------------")

def evaluate(loader, model, loss_method, device, out=False):
    """
    Evaluates the model on a given data loader.
    """
    model.eval()
    loss_all = 0
    count = 0
    
    if out:
        ids_list, predict_list, target_list = [], [], []

    if loader is None:
        if out:
            return torch.tensor(0.0), np.array([]), np.array([]), np.array([])
        else:
            return torch.tensor(0.0)
            
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            loss = getattr(F, loss_method)(output, data.y)
            loss_all += loss * output.size(0)
            
            if out:
                try:
                    ids_temp = [item for sublist in data.structure_id for item in sublist]
                    ids_temp = [item for sublist in ids_temp for item in sublist]
                    ids_list.extend(ids_temp)
                except AttributeError:
                    ids_list.extend(["unknown"] * output.size(0)) # Fallback
                    
                predict_list.append(output.data.cpu().numpy())
                target_list.append(data.y.cpu().numpy())
                
            count = count + output.size(0)

    if count == 0:
        if out:
            return torch.tensor(0.0), np.array([]), np.array([]), np.array([])
        else:
            return torch.tensor(0.0)

    loss_all = loss_all / count

    if out:
        ids = np.array(ids_list)
        targets = np.concatenate(target_list, axis=0)
        predictions = np.concatenate(predict_list, axis=0)
        return loss_all, ids, targets, predictions
    else:
        return loss_all


def write_results(filename, ids, targets, predictions):
    """
    Writes model predictions to a CSV file with smart headers.
    """
    if targets.ndim == 1:
        num_targets = 1
        targets = targets.reshape(-1, 1)
        predictions = predictions.reshape(-1, 1)
    else:
        num_targets = targets.shape[1]
        
    if ids.ndim == 0:
        ids = ids.reshape(-1)

    header = ["ids"]
    header += [f"target_{i+1}" for i in range(num_targets)]
    header += [f"prediction_{i+1}" for i in range(num_targets)]
    
    # Ensure ids is a 2D array for hstack
    if ids.ndim == 1:
        ids = ids.reshape(-1, 1)

    try:
        output_data = np.hstack((ids, targets, predictions))
    except ValueError:
        print("Warning: Could not hstack ids, targets, predictions. Skipping CSV write.")
        print(f"Shapes: ids={ids.shape}, targets={targets.shape}, predictions={predictions.shape}")
        return

    with open(filename, "w", newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)
        csvwriter.writerows(output_data)
