import torch
from torch.utils.data import Dataset, DataLoader

class MemoryCycleGANDataset(Dataset):
    def __init__(self, tensor_A, tensor_B):
        """
        Args:
            tensor_A: Tensor for Domain A (Input)
            tensor_B: Tensor for Domain B (Target/Style)
            
            Expected Input Shape: (2, 128, 128, N) or (N, 2, 128, 128)
        """
        self.data_A = self._fix_shape(tensor_A)
        self.data_B = self._fix_shape(tensor_B)

        self.len_A = len(self.data_A)
        self.len_B = len(self.data_B)
        self.max_len = max(self.len_A, self.len_B)

    def _fix_shape(self, tensor):
        """
        Automatically permutes (C, H, W, N) -> (N, C, H, W) if needed.
        """
        # If the last dimension is largest, it's likely N. 
        # Or if the first dimension is 2 (Channels), we need to move N to front.
        if tensor.shape[0] == 2 and len(tensor.shape) == 4:
            # Assuming shape is (2, 128, 128, N) -> Permute to (N, 2, 128, 128)
            print(f"Permuting tensor from {tensor.shape} to PyTorch standard...")
            return tensor.permute(3, 0, 1, 2)
        return tensor

    def __getitem__(self, index):
        # 1. Get Domain A (Deterministic index)
        # We loop over A if the dataset is longer than A
        idx_A = index % self.len_A
        item_A = self.data_A[idx_A]

        # 2. Get Domain B (Random index)
        # CycleGAN requires UNPAIRED training. Even if your data is paired in the file,
        # we randomize B to force the network to learn the "Style" rather than memorizing pairs.
        idx_B = torch.randint(0, self.len_B, (1,)).item()
        item_B = self.data_B[idx_B]

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return self.max_len

# --- Helper Function to Load and Create Loader ---

def get_memory_loader(path_A, path_B, batch_size=4):
    """
    Loads large single-file tensors and returns a DataLoader.
    Assumes files are .pt or .npy
    """
    # 1. Load the big files into memory
    print(f"Loading {path_A}...")
    # Map location ensures we don't overload GPU memory immediately; we load to CPU first
    t_A = torch.load(path_A, map_location='cpu') 
    
    print(f"Loading {path_B}...")
    t_B = torch.load(path_B, map_location='cpu')
    
    # 2. Create Dataset
    dataset = MemoryCycleGANDataset(t_A, t_B)
    
    # 3. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffles the 'A' indices
        num_workers=0, # Use 0 for memory-loaded tensors to avoid memory duplication overhead
        pin_memory=True # Speed up transfer to GPU
    )
    
    return loader



# Assuming you have two files: 'inputs.pt' and 'targets.pt'
# Each contains a tensor of shape (2, 128, 128, 5000) for example.

loader = get_memory_loader("inputs.pt", "targets.pt", batch_size=4)

# Integration with the previous training loop
for i, batch in enumerate(loader):
    real_A = batch['A'].to(device) # Shape: [4, 2, 128, 128]
    real_B = batch['B'].to(device) # Shape: [4, 2, 128, 128]
    
    # ... call your train step ...
    
    if i == 0:
        print("Shapes verified:", real_A.shape, real_B.shape)