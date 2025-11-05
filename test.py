# test.py
"""
test.py (Test Executor)
Responsibilities:
- Receives the *best* model and data loaders.
- Runs final evaluation on train, val, and test sets.
- Writes all output files (CSV, error files).
"""
import torch
import numpy as np

# Import shared functions from utils
from utils import evaluate, write_results

def run_final_evaluation(
    device,
    model,
    train_loader,
    val_loader,
    test_loader,
    loss_method,
    job_name,
    write_output=True,
    write_error=True
):
    """
    Runs final evaluation on all data splits and writes results.
    """
    
    print("--- Performing final evaluation ---")
    
    # Evaluate on training set
    train_error, train_ids, train_targets, train_preds = evaluate(
        train_loader, model, loss_method, device, out=True
    )
    print(f"Final Train Error: {train_error:.5f}") 

    # Evaluate on validation set
    if val_loader:
        val_error, val_ids, val_targets, val_preds = evaluate(
            val_loader, model, loss_method, device, out=True
        )
        print(f"Final Val Error: {val_error:.5f}")
    else:
        val_error = float("NaN")

    # Evaluate on test set
    if test_loader:
        test_error, test_ids, test_targets, test_preds = evaluate(
            test_loader, model, loss_method, device, out=True
        )
        print(f"Final Test Error: {test_error:.5f}")
    else:
        test_error = float("NaN")

    # Write outputs to CSV
    if write_output:
        print("Writing outputs to CSV...")
        write_results(
            f"{job_name}_train_outputs.csv", 
            train_ids, train_targets, train_preds
        )
        if val_loader:
            write_results(
                f"{job_name}_val_outputs.csv",
                val_ids, val_targets, val_preds
            )
        if test_loader:
            write_results(
                f"{job_name}_test_outputs.csv",
                test_ids, test_targets, test_preds
            )

    # Write final error values
    train_err_val = train_error.item() if torch.is_tensor(train_error) else train_error
    val_err_val = val_error.item() if torch.is_tensor(val_error) else val_error
    test_err_val = test_error.item() if torch.is_tensor(test_error) else test_error
    
    error_values = np.array((train_err_val, val_err_val, test_err_val))

    if write_error:
        error_filename = f"{job_name}_errorvalues.csv"
        np.savetxt(
            error_filename,
            error_values[np.newaxis, ...],
            delimiter=",",
            header="train_error,val_error,test_error",
            comments=""
        )
        print(f"Error values saved to {error_filename}")
        
    return error_values