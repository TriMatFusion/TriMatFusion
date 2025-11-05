"""
train.py (Training Executor)
Responsibilities:
- Receives *ready-to-use* objects (model, loaders, optimizer, scheduler).
- Runs the training and validation epoch loop.
- Tracks the best model state *in memory*.
- Returns the best model state dictionary.
"""
import time
import torch
import torch.nn.functional as F

# Import shared functions from utils
from utils import evaluate 

def train(model, optimizer, loader, loss_method, device):
    """Performs a single training epoch."""
    model.train()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = getattr(F, loss_method)(output, data.y)
        loss.backward()
        loss_all += loss.detach() * output.size(0)
        optimizer.step()
        count = count + output.size(0)
    
    if count == 0:
        return torch.tensor(0.0) # Handle empty loader
        
    loss_all = loss_all / count
    return loss_all


def run_training_loop(
    device,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    loss_method,
    epochs,
    verbosity
):
    """
    The main training and validation loop.

    Returns:
        dict: The state dict of the best performing model, or None.
    """
    
    train_error = val_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    best_model_state = None
    
    print("--- Beginning epoch loop ---")
    for epoch in range(1, epochs + 1):

        lr = scheduler.optimizer.param_groups[0]["lr"]
        train_error = train(model, optimizer, train_loader, loss_method, device=device) 
        
        if val_loader:
            val_error = evaluate(val_loader, model, loss_method, device=device, out=False)
        else:
            val_error = float("NaN") # No validation set

        epoch_time = time.time() - train_start
        train_start = time.time()
    
        # Save best model state in memory
        if val_loader and (val_error < best_val_error):
            best_val_error = val_error
            best_model_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
        elif not val_loader:
            # If no validation, just save the last epoch
            best_model_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }

        # Step the scheduler on validation loss if available, else on train loss
        if val_loader:
            scheduler.step(val_error)
        else:
            scheduler.step(train_error)

        if epoch % verbosity == 0:
            print(f"Epoch: {epoch:04d}, LR: {lr:.6f}, "
                  f"Train Error: {train_error:.5f}, Val Error: {val_error:.5f}, "
                  f"Time/epoch (s): {epoch_time:.5f}")
            
    print("--- Epoch loop finished ---")
    
    return best_model_state

