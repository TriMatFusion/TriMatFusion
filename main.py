"""
main.py (The Conductor)
Responsibilities:
- Parse CLI arguments.
- Load and merge the config.yml.
- Set up global environment (seeds, device).
- Create all objects
- Call train.run_training_loop() and pass objects.
- Load best model state.
- Call test.run_final_evaluation() and pass objects.
"""
import os
import argparse
import time
import sys
import random
import numpy as np
import yaml
import random
import torch
from torch_geometric.data import DataLoader

# --- Import all local modules ---
import dataset as dataset_loader  # Aliased to avoid name conflict
import train as trainer
import test as tester
import utils
from model import ModalFusion

# Set python hash seed for reproducibility
os.environ["PYTHONHASHSEED"] = str(42)


def setup_environment(seed):
    """Sets random seeds for all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}. Deterministic CUDNN enabled.")


def parse_arguments():
    """Parses command-line arguments (simplified flat structure)."""
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config_path", default="config.yml",
                        type=str, help="Location of config file")

    # Job Arguments
    parser.add_argument("--job_name", default=None,
                        type=str, help="name of your job")
    parser.add_argument("--seed", default=None, type=int,
                        help="seed for data split, 0=random")
    parser.add_argument("--model_path", default=None,
                        type=str, help="path of the model .pth file")
    parser.add_argument("--save_model", default=None,
                        type=str, help="save model")
    parser.add_argument("--load_model", default=None,
                        type=str, help="load model")
    parser.add_argument("--write_output", default=None,
                        type=str, help="write outputs to csv")
    parser.add_argument("--reprocess", default=None,
                        type=str, help="reprocess data")

    # Processing Arguments
    parser.add_argument("--data_path", default=None,
                        type=str, help="location of data")
    parser.add_argument("--format", default=None, type=str,
                        help="format of input data")
    parser.add_argument("--target_path", default=None,
                        type=str, help="target path of processed data")

    # Training Arguments
    parser.add_argument("--train_ratio", default=None,
                        type=float, help="train ratio")
    parser.add_argument("--val_ratio", default=None,
                        type=float, help="validation ratio")
    parser.add_argument("--test_ratio", default=None,
                        type=float, help="test ratio")
    parser.add_argument("--verbosity", default=None, type=int,
                        help="prints errors every x epochs")
    parser.add_argument("--target_index", default=None,
                        type=int, help="which column to use as target")

    # Model Hyperparameters
    parser.add_argument("--epochs", default=None, type=int,
                        help="number of total epochs")
    parser.add_argument("--batch_size", default=None,
                        type=int, help="batch size")
    parser.add_argument("--lr", default=None, type=float, help="learning rate")

    return parser.parse_args(sys.argv[1:])


def load_and_merge_config(args):
    """Loads YAML and merges CLI args, handling the flat structure."""
    print(f"Loading config file from {args.config_path}")
    assert os.path.exists(
        args.config_path), f"Config file not found in {args.config_path}"
    with open(args.config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    args_dict = vars(args)

    # Define mapping from flat args_dict to nested config
    key_mapping = {
        "job_name": "Job", "seed": "Job", "model_path": "Job",
        "load_model": "Job", "save_model": "Job", "write_output": "Job",
        "reprocess": "Job", "write_error": "Job",

        "data_path": "Processing", "format": "Processing", "target_path": "Processing",
        "dataset_type": "Processing", "dictionary_source": "Processing",

        "train_ratio": "Training", "val_ratio": "Training", "test_ratio": "Training",
        "verbosity": "Training", "target_index": "Training", "loss": "Training",

        "epochs": "Models", "batch_size": "Models", "lr": "Models",
        "num_workers": "Models"
    }

    # Iterate over all provided CLI args
    for arg_key, arg_val in args_dict.items():
        if arg_val is not None and arg_key in key_mapping:
            config_section = key_mapping[arg_key]
            # Handle special cases like 'format' -> 'data_format'
            config_key = "data_format" if arg_key == "format" else arg_key
            config[config_section][config_key] = arg_val

    if config["Job"].get("seed") == 0:
        config["Job"]["seed"] = np.random.randint(1, 1e6)

    return config


def loader_setup(
    train_ratio,
    val_ratio,
    test_ratio,
    batch_size,
    dataset,
    data_path,
    num_workers=0,
    seed=42
):
    """Splits data and creates PyTorch Geometric DataLoader instances."""
    train_dataset, val_dataset, test_dataset = dataset_loader.split_data(
        dataset, train_ratio, val_ratio, test_ratio, data_path, seed=seed)

    train_loader = val_loader = test_loader = None

    # Use persistent_workers=True if num_workers > 0 for speed
    persist_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persist_workers
    )
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persist_workers
        )
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persist_workers
        )
    return (
        train_loader,
        val_loader,
        test_loader
    )


def model_setup(
    device,
    model_params,
    dataset,
    load_model=False,
    model_path=None
):
    """Initializes the hard-coded model, moves it to the device."""
    model = ModalFusion(
        data=dataset, **(model_params if model_params is not None else {})
    ).to(device)

    if str(load_model).lower() == "true":
        print(f"Loading model from {model_path}...")
        assert os.path.exists(
            model_path), f"Saved model not found at: {model_path}"
        saved = torch.load(model_path, map_location=device)
        model.load_state_dict(saved["model_state_dict"])

    utils.model_summary(model)
    return model


def main():
    """Main execution function."""
    start_time = time.time()
    print("Starting main.py...")

    # 1. Load Configs
    args = parse_arguments()
    config = load_and_merge_config(args)

    j_params = config["Job"]
    p_params = config["Processing"]
    t_params = config["Training"]
    m_params = config["Models"]

    # 2. Set up Environment
    setup_environment(j_params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Create All Objects
    print("Loading dataset...")
    dataset = dataset_loader.get_dataset(
        p_params["data_path"],
        t_params["target_index"],
        reprocess=j_params.get("reprocess", "False"),
        processing_args=p_params
    )

    print("Setting up model...")
    model = model_setup(
        device,
        m_params,
        dataset,
        j_params["load_model"],
        j_params["model_path"],
    )

    print("Setting up data loaders...")
    (
        train_loader,
        val_loader,
        test_loader
    ) = loader_setup(
        t_params["train_ratio"],
        t_params["val_ratio"],
        t_params["test_ratio"],
        m_params["batch_size"],
        dataset,
        p_params["data_path"],
        num_workers=m_params.get("num_workers", 0),
        seed=j_params.get("seed", 42)
    )

    optimizer = getattr(torch.optim, m_params["optimizer"])(
        model.parameters(),
        lr=m_params["lr"],
        **m_params.get("optimizer_args", {})
    )

    scheduler = getattr(torch.optim.lr_scheduler, m_params["scheduler"])(
        optimizer, **m_params.get("scheduler_args", {})
    )

    # 4. Run Training
    print("Handing over to trainer.run_training_loop()...")
    best_model_state = trainer.run_training_loop(
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_method=t_params["loss"],
        epochs=m_params["epochs"],
        verbosity=t_params["verbosity"]
    )

    # 5. Save Best Model and Run Final Test
    if best_model_state and str(j_params["save_model"]).lower() == "true":
        print(f"Saving best model to {j_params['model_path']}")
        torch.save(best_model_state, j_params["model_path"])
        # Load best state for final test
        model.load_state_dict(best_model_state["model_state_dict"])
    else:
        print("Warning: No best model state found or save_model=False. Using final model for testing.")

    print("Handing over to tester.run_final_evaluation()...")
    tester.run_final_evaluation(
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_method=t_params["loss"],
        job_name=j_params["job_name"],
        write_output=str(j_params["write_output"]).lower() == "true",
        write_error=str(j_params.get("write_error", "False")).lower() == "true"
    )

    print(
        f"--- {(time.time() - start_time):.4f} total seconds elapsed (main.py) ---")


if __name__ == "__main__":
    main()
