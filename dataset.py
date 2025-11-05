"""
dataset.py
Data Processing Module

Responsibilities:
- Check for processed data on disk.
- If data is not found or reprocess=True, run the process_data function.
- Load and return the processed dataset.
- Contains all logic for graph creation, featurization, and text processing.
- Contains data splitting logic.
"""
import os
import sys
import csv
import json
import numpy as np
import ase
from ase.io import read
import glob
from scipy.stats import rankdata
import pickle
from pymatgen.core.periodic_table import Element
import torch
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from transformers import AutoTokenizer
from tokenizers.normalizers import BertNormalizer

################################################################################
# Data splitting
################################################################################

def save_split_datasets(train_dataset, val_dataset, test_dataset, data_path):
    """Persist splits so subsequent runs/models reuse the exact same split."""
    split_file = os.path.join(data_path, "split_datasets.pkl")
    try:
        with open(split_file, "wb") as f:
            pickle.dump((train_dataset, val_dataset, test_dataset), f)
    except IOError as e:
        print(f"Warning: Could not save split datasets: {e}")


def load_split_datasets(data_path):
    """Load previously saved splits; return None if not found."""
    split_file = os.path.join(data_path, "split_datasets.pkl")
    try:
        with open(split_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except IOError as e:
        print(f"Warning: Could not load split datasets: {e}")
        return None

# In-memory cache for the current run
split_datasets_cache = None 

def split_data(
    dataset, 
    train_ratio,
    val_ratio,
    test_ratio,
    data_path,
    seed: int, # Seed must be provided
    save=True
):
    """
    Random split into train/val/test; cache and persist the split for reproducibility.

    """
    
    global split_datasets_cache 
    if split_datasets_cache is not None:
        print("Using in-memory split cache.")
        return split_datasets_cache
    
    split_datasets = load_split_datasets(data_path) 
    if split_datasets is not None:
        print(f"Loaded split datasets from: {data_path}")
        split_datasets_cache = split_datasets
        return split_datasets
    
    print(f"Performing new data split with seed: {seed}")
    dataset_size = len(dataset)
    
    if (train_ratio + val_ratio + test_ratio) > 1.0 + 1e-9: # Add tolerance
         raise ValueError(f"Invalid ratios: {train_ratio}+{val_ratio}+{test_ratio} sum to > 1.")
    
    train_length = int(dataset_size * train_ratio)
    val_length = int(dataset_size * val_ratio)
    test_length = int(dataset_size * test_ratio)
    
    # Ensure sum of lengths does not exceed dataset_size due to rounding
    if train_length + val_length + test_length > dataset_size:
        print("Warning: Train/Val/Test lengths rounded up. Adjusting test_length.")
        test_length = dataset_size - train_length - val_length

    unused_length = dataset_size - train_length - val_length - test_length
    (
        train_dataset,
        val_dataset,
        test_dataset,
        unused_dataset,
    ) = torch.utils.data.random_split(
        dataset,
        [train_length, val_length, test_length, unused_length],
        generator=torch.Generator().manual_seed(seed),
    )
    print(
        f"train length: {train_length}, val length: {val_length}, "
        f"test length: {test_length}, unused length: {unused_length}"
    )
    
    split_datasets = (train_dataset, val_dataset, test_dataset)
    split_datasets_cache = split_datasets
    
    if save:
        save_split_datasets(train_dataset, val_dataset, test_dataset, data_path)
        print(f"Saved split datasets to {data_path}")
    
    return split_datasets


################################################################################
# Pytorch datasets
################################################################################

def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    """
    Fetch dataset; if processed files are absent or reprocess=True, 
    run `process_data` to generate them.
    """
    
    if processing_args is None:
        print("Error: processing_args is None. Cannot process data.")
        sys.exit()
    else:
        processed_path = processing_args.get("processed_path", "processed")

    transforms = GetY(index=target_index)

    if not os.path.exists(data_path):
        print(f"Error: Data path not found in: {data_path}")
        sys.exit()

    processed_dir = os.path.join(data_path, processed_path)
    processed_file_inmemory = os.path.join(processed_dir, "data.pt")
    processed_file_sharded = os.path.join(processed_dir, "data0.pt")

    needs_processing = reprocess == "True" or \
                       (not os.path.exists(processed_file_inmemory) and \
                        not os.path.exists(processed_file_sharded))

    if needs_processing:
        if reprocess == "True":
            print("Reprocessing data as requested.")
        else:
            print(f"Processed data not found in {processed_dir}. Starting processing...")
        
        if not os.path.isdir(processed_dir):
            os.makedirs(processed_dir)
            
        process_data(data_path, processed_dir, processing_args)
    else:
        print(f"Found processed data in {processed_dir}.")
 
    if os.path.exists(processed_file_inmemory):
        print("Loading InMemoryDataset from data.pt...")
        return StructureDataset(data_path, processed_path, transforms)
    elif os.path.exists(processed_file_sharded):
        print("Loading sharded Dataset from data0.pt...")
        return StructureDataset_large(data_path, processed_path, transforms)
    else:
        raise RuntimeError(f"Processing finished but no processed files found in {processed_dir}.")


class StructureDataset(InMemoryDataset):
    """
    In-memory dataset loading 'processed/data.pt'.
    """
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform) 
        # Compatibility fix for some torch_geometric versions
        if 'storage' in dir(torch_geometric.data):
             torch_geometric.data.storage = torch_geometric.data
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=torch.device("cpu"))
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self): 
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self): 
        return ["data.pt"]

    
class StructureDataset_large(Dataset):
    """
    Sharded dataset loading 'processed/data_{i}.pt' files.
    """
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset_large, self).__init__(
            data_path, transform, pre_transform
        )
        self._files = sorted(
            [os.path.basename(p) for p in glob.glob(os.path.join(self.processed_dir, "data*.pt"))]
        )
        if not self._files:
            raise FileNotFoundError(f"No shard files found under {self.processed_dir} matching 'data*.pt'.")
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        return self._files

    def len(self):
        return len(self._files)

    def get(self, idx):
        fname = self._files[idx]
        path = os.path.join(self.processed_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shard not found: {path}")
        return torch.load(path)

################################################################################
# Text normalization
################################################################################
# Load mappings relative to this file's location
_mappings_file = os.path.join(os.path.dirname(__file__), "vocab_mappings.txt")
if os.path.exists(_mappings_file):
    with open(_mappings_file, "r", encoding="utf-8") as f:
        mappings = {m[0]: m[2:] for m in f.read().strip().split("\n")}
else:
    print("Warning: vocab_mappings.txt not found. Text normalization may be incomplete.")
    mappings = {}

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize(text):
    """
    Normalize text line-by-line using BertNormalizer and the vocab mapping.
    """
    if not text:
        return ""
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)

################################################################################
# Fingerprinting
################################################################################

def _elem_props(sym: str) -> np.ndarray:
    """Helper function to get 8 elemental properties for a symbol."""
    try:
        e = Element(sym)
    except:
        # Handle dummy species or other non-element symbols
        return np.zeros(8, dtype=np.float32)
        
    return np.array([
        e.Z,
        e.group or 0,
        e.row  or 0,
        float(e.molar_volume) if e.molar_volume is not None else 0.0,
        float(e.atomic_mass)  if e.atomic_mass  is not None else 0.0,
        float(e.X)            if e.X            is not None else 0.0,
        float(e.atomic_radius)      if e.atomic_radius      is not None else 0.0,
        float(e.thermal_conductivity) if e.thermal_conductivity is not None else 0.0,
    ], dtype=np.float32)

def comp8_stats_from_ase(atoms, stats=("mean","std","min","max","q25","q75")) -> np.ndarray:
    """
    Return a fingerprint of shape (len(stats), 8).
    """
    if len(atoms) == 0:
        return np.zeros((len(stats), 8), dtype=np.float32)

    M = np.vstack([_elem_props(a.symbol) for a in atoms])  # [N,8]

    outs = []
    for s in stats:
        if s == "mean":
            outs.append(M.mean(axis=0))
        elif s == "std":
            outs.append(M.std(axis=0))
        elif s == "min":
            outs.append(M.min(axis=0))
        elif s == "max":
            outs.append(M.max(axis=0))
        elif s == "q25":
            outs.append(np.quantile(M, 0.25, axis=0))
        elif s == "q75":
            outs.append(np.quantile(M, 0.75, axis=0))
        else:
            raise ValueError(f"unknown stat: {s}")
    return np.stack(outs, axis=0).astype(np.float32)  # (S,8)

################################################################################
#  Processing
################################################################################

def process_data(data_path, processed_dir, processing_args):
    """Process raw structures and texts into a PyG dataset."""
    print("Processing data to: " + processed_dir)
    
    #Load atom dictionary
    if processing_args["dictionary_source"] == "default":
        print("Using default dictionary.")
        dict_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "dictionary_default.json",
        )
    elif processing_args["dictionary_source"] == "provided":
        print("Using provided dictionary.")
        dict_path = os.path.join(data_path, processing_args["dictionary_path"])
    else:
        print(f"Error: Invalid dictionary_source")
        sys.exit()

    if not os.path.exists(dict_path):
        print(f"Atom dictionary not found at {dict_path}, exiting program...")
        sys.exit()
    else:
        print(f"Loading atom dictionary from {dict_path}.")
        atom_dictionary = get_dictionary(dict_path)
     
        
    #Load targets [ID,Description,Target]
    target_property_file = os.path.join(data_path, processing_args["target_path"])
    assert os.path.exists(target_property_file), (
        "targets not found in " + target_property_file
    ) 
    
    _max = sys.maxsize
    while True:
        try:
            csv.field_size_limit(_max)
            break
        except OverflowError:
            _max = int(_max / 10)
    
    with open(target_property_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        try:
            next(reader) # skip header
        except StopIteration:
            print(f"Error: Target file {target_property_file} is empty.")
            sys.exit()
        target_data = [row for row in reader]

    # --- OPTIMIZATION: Load tokenizer *before* the loop ---
    print("Loading tokenizer 'matscibert'...")
    tokenizer = AutoTokenizer.from_pretrained('matscibert')

    # --- OPTIMIZATION: Prepare edge feature featurizer *before* loop ---
    if processing_args.get("edge_features", "False") == "True":
        print("Edge features enabled. Initializing GaussianSmearing.")
        distance_gaussian = GaussianSmearing(
            0, 1, processing_args["graph_edge_length"], 0.2
        )
    else:
        distance_gaussian = None
        
    data_list = []
    all_elements = []
    all_lengths = []
    
    print(f"Processing {len(target_data)} structures...")
    for index in range(0, len(target_data)): 
        structure_id = target_data[index][0]
        data = Data() 
        
        struct_file = os.path.join(
            data_path, structure_id + "." + processing_args["data_format"]
        )
        if not os.path.exists(struct_file):
            print(f"Warning: Structure file not found: {struct_file}. Skipping.")
            continue
            
        try:
            ase_crystal = read(struct_file)
        except Exception as e:
            print(f"Error reading {struct_file}: {e}. Skipping.")
            continue
            
        data.ase = ase_crystal # Store temporarily

        all_lengths.append(len(ase_crystal))
        all_elements.extend(list(set(ase_crystal.get_chemical_symbols())))

        distance_matrix = ase_crystal.get_all_distances(mic=True)

        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )
        
        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1] 

        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
        )
        
        data.edge_index = edge_index
        data.edge_weight = edge_weight
        data.edge_descriptor = {"distance": edge_weight}
        
        target = target_data[index][2:]  
        y = torch.Tensor(np.array([target], dtype=np.float32))
        data.y = y
        
        fp8 = comp8_stats_from_ase(ase_crystal)
        data.fingerprint = torch.from_numpy(fp8).float()
    
        text = target_data[index][1]
        encoding = tokenizer(
            normalize(text), # Apply normalization
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ) 
        data.text_input_ids = encoding['input_ids'] 
        data.text_attention_mask = encoding['attention_mask'] 
    
        data.z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
        data.structure_id = [[structure_id] * len(data.y)]
        data.length = torch.LongTensor([len(ase_crystal)])
        
        if atom_dictionary:
            try:
                atom_fea = np.vstack(
                    [
                        atom_dictionary[str(data.ase.get_atomic_numbers()[i])]
                        for i in range(len(data.ase)) 
                    ]
                ).astype(float) 
                data.x = torch.Tensor(atom_fea)
            except KeyError as e:
                print(f"Error: Atom number {e} not found in dictionary for structure {structure_id}. Skipping.")
                continue
            
        data = OneHotDegree(
            data, processing_args["graph_max_neighbors"] + 1
        )
        
        if distance_gaussian:
            data.edge_attr = distance_gaussian(
                data.edge_descriptor["distance"]
            )

        if processing_args.get("verbose", "False") == "True" and (
            (index + 1) % 500 == 0 or (index + 1) == len(target_data)
        ):
            print("Data processed: ", index + 1, "out of", len(target_data))

        data_list.append(data)
        
    print(f"Successfully processed {len(data_list)} structures.")
    
    Cleanup(data_list, ["ase", "edge_descriptor"]) 

    if processing_args["dataset_type"] == "inmemory":
        print("Collating dataset and saving to data.pt...")
        data, slices = InMemoryDataset.collate(data_list) 
        torch.save((data, slices), os.path.join(processed_dir, "data.pt"))
        print("Dataset saved.")
    else:
        # This part would need to be implemented to save shards
        print(f"Warning: 'large' dataset_type saving not implemented. Saving as 'inmemory'.")
        data, slices = InMemoryDataset.collate(data_list) 
        torch.save((data, slices), os.path.join(processed_dir, "data.pt"))
        print("Dataset saved.")


################################################################################
#  Processing sub-functions
################################################################################

def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False): 
    """
    Sorts a distance matrix, applying a threshold and neighbor limit.
    """
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1 
        )
    distance_matrix_trimmed = np.nan_to_num( 
        np.where(mask, np.nan, distance_matrix_trimmed) 
    )
    # +1 to account for self-loop/diagonal
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    else:
        return (distance_matrix_trimmed != 0).astype(int)
    
    
class GaussianSmearing(torch.nn.Module): 
    """Expands edge distances with a Gaussian basis."""
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    """
    Obtain node degree in one-hot representation.
    max_degree is the maximum *number* of neighbors (e.g., 12),
    so we add +1 for the self-loop, and +1 for num_classes.
    """
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg
    return data


def get_dictionary(dictionary_file):
    """Obtain dictionary file for elemental features."""
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


def Cleanup(data_list, entries): 
    """Deletes temporary attributes from Data objects to speed up dataloader."""
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


################################################################################
#  Transforms
################################################################################
class GetY(object):
    """
    Transform to extract a specific target index from data.y.
    Assumes data.y is shape [1, num_targets].
    """
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        if self.index == -1:
            # Keep all targets
            return data
            
        try:
            if data.y.ndim == 1:
                # Handle 1D target
                data.y = data.y.view(1, -1)[0][self.index]
            else:
                # Standard case: [1, num_targets] -> [1]
                data.y = data.y[0][self.index]
        except Exception as e:
            print(f"Error extracting target index {self.index} from data.y with shape {data.y.shape}.")
            print(f"Data sample ID: {getattr(data, 'structure_id', 'unknown')}")
            raise e
        return data