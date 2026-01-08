import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
from diagnostics import calculate_physics_metrics
import itertools
import random
import os
import csv
import numpy as np
# ==========================================
# 0. UTILITIES (Inception Score & Checkpoints)
# ==========================================

class CheckpointManager:
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # Initialize CSV logging
        self.log_file = os.path.join(save_dir, "training_log.csv")
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Loss_G",  "Loss_D",     
                     "Loss_G_GAN",
                    "Loss_G_Cycle",    
                    "Loss_G_ID",       
                    "PSD_Error", "Hist_Error"])

    def save_checkpoint(self, state, epoch):
        filepath = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(state, filepath)
        print(f"-> Checkpoint saved: {filepath}")
        
    def log_metrics(self, epoch, loss_g, loss_d, loss_g_gan, loss_g_cycle, loss_g_id, psd_error, hist_error):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss_g, loss_d,loss_g_gan, loss_g_cycle, loss_g_id, psd_error, hist_error])
            



# ==========================================
# 1. ARCHITECTURE DEFINITIONS
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_shape=(2, 128, 128), num_residual_blocks=6):
        super(Generator, self).__init__()
        channels = input_shape[0]

        # Initial Conv
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual Blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape=(2, 128, 128)):
        super(Discriminator, self).__init__()
        channels = input_shape[0]

        def discriminator_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# ==========================================
# 2. DATA LOADING (TENSOR BASED)
# ==========================================

class MemoryCycleGANDataset(Dataset):
    def __init__(self, tensor_A, tensor_B):
        self.data_A = self._fix_shape(tensor_A)
        self.data_B = self._fix_shape(tensor_B)
        self.len_A = len(self.data_A)
        self.len_B = len(self.data_B)
        self.max_len = max(self.len_A, self.len_B)

    def _fix_shape(self, tensor):
        # Permutes (2, 128, 128, N) -> (N, 2, 128, 128)
        if tensor.shape[0] == 2 and len(tensor.shape) == 4:
            return tensor.permute(3, 0, 1, 2)
        return tensor

    def __getitem__(self, index):
        # A is deterministic, B is random (Unpaired training)
        item_A = self.data_A[index % self.len_A]
        idx_B = random.randint(0, self.len_B - 1)
        item_B = self.data_B[idx_B]
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return self.max_len

# ==========================================
# 3. UTILITIES
# ==========================================

class ReplayBuffer:
    """Stores generated images to stabilize the discriminator."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def get_memory_loader(input_path_A , input_path_B , batch_size =4):
    """
    Loads large single-file tensors and returns a DataLoader.
    Assumes files are .pt 
    """

    print("Loading tensors...")
    # NOTE: Using dummy data for demonstration if files don't exist
    if input_path_A is None :
        print("Warning: Files not found. Generating Dummy Data (2, 128, 128, 100)...")
        tensor_A = torch.randn(100,2, 128, 128)
        tensor_B = torch.randn(100, 2, 128, 128 )
    else:
        tensor_A = torch.load(input_path_A, map_location='cpu')
        tensor_B = torch.load(input_path_B, map_location='cpu')

    dataset = MemoryCycleGANDataset(tensor_A, tensor_B)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True # Speed up transfer to GPU
    )
    
    return loader

# ==========================================
# 4. MAIN TRAINING SCRIPT
# ==========================================

def train(input_path_A=None, input_path_B=None, epochs=100,save_dir="./", 
          batch_size=4, resume_path=None ):
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    # Manager for saving stuff
    manager = CheckpointManager(save_dir)
    # 2. Load Data
    dataloader = get_memory_loader(input_path_A, input_path_B, batch_size) 
    # 3. Initialize Models
    input_shape = (2, 128, 128)
    G_AB = Generator(input_shape).to(device)
    G_BA = Generator(input_shape).to(device)
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)
    lr= 0.0002
    beta1 = 0.5
    beta2= 0.999 
    lambda_id =5.
    lambda_cyc= 10.0
    # 4. Optimizers & Losses
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, beta2))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, beta2))

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    start_epoch = 1

    # ==========================================
    #  LOAD CHECKPOINT LOGIC
    # ==========================================
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # 1. Load Model Weights
        G_AB.load_state_dict(checkpoint['G_AB'])
        G_BA.load_state_dict(checkpoint['G_BA'])
        D_A.load_state_dict(checkpoint['D_A'])
        D_B.load_state_dict(checkpoint['D_B'])
        
        # 2. Load Optimizer States (Crucial for stability)
        optimizer_G.load_state_dict(checkpoint['opt_G'])
        optimizer_D_A.load_state_dict(checkpoint['opt_D_A'])
        optimizer_D_B.load_state_dict(checkpoint['opt_D_B'])
        
        # 3. Set Start Epoch
        start_epoch = checkpoint['epoch'] + 1
        epochs+= start_epoch
        print(f"Resuming training from Epoch {start_epoch}")
    else:
        print("No checkpoint found or provided. Starting from scratch.")

    # ==========================================
    #  TRAINING LOOP
    # ==========================================

    print(f"Starting training. Logs will be saved to {save_dir}/")

    for epoch in range(start_epoch, epochs + 1):
        loss_G_total = 0.0
        loss_G_GAN = 0.0
        loss_G_cyc = 0.0
        loss_G_id = 0.0
        loss_D_total = 0.0
        epoch_start_time = time.time()  # Start Epoch Timer
        # Timing Accumulators
        data_time_sum = 0.0
        batch_time_sum = 0.0
        
        # 'end' marks the timestamp when the PREVIOUS batch finished.
        # The time from 'end' to inside the loop is the Data Loading time.
        end = time.time()
        for i, batch in enumerate(dataloader):
            # 1. Measure Data Loading Time
            data_time = time.time() - end
            data_time_sum += data_time

            
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            #print(f"Domain A Shape: {real_A.shape}")  # Should be [4, 2, 128, 128]
            #print(f"Domain B Shape: {real_B.shape}")
            # Define Valid/Fake targets based on D output shape
            # We run one forward pass to get the shape dynamically
            target_shape = D_A(real_A).shape
            valid = torch.ones(target_shape, requires_grad=False).to(device)
            fake = torch.zeros(target_shape, requires_grad=False).to(device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            rec_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A)
            rec_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss (Weighted)
            loss_G = loss_GAN + (lambda_cyc * loss_cycle) + (lambda_id* loss_identity)
            loss_G.backward()
            optimizer_G.step()
            loss_G_total += loss_G.item()
            loss_G_GAN += loss_GAN.item()
            loss_G_cyc += loss_cycle.item()
            loss_G_id += loss_identity.item()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            loss_D_total += (loss_D_A.item() + loss_D_B.item())

            # --- END OF EPOCH ACTIONS ---
            # 2. Measure Total Batch Time
            # (Time from start of data load -> end of training step)
            batch_time = time.time() - end
            batch_time_sum += batch_time
            
            # Reset timer for next iteration
            end = time.time()
        # 1. Calculate Average Losses
        avg_loss_G = loss_G_total / len(dataloader)
        avg_loss_G_GAN = loss_G_GAN / len(dataloader)
        avg_loss_G_cyc = loss_G_cyc / len(dataloader)
        avg_loss_G_id = loss_G_id / len(dataloader)
        avg_loss_D = loss_D_total / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
    
        psd_err, hist_err = calculate_physics_metrics(G_AB, dataloader, device)
        print(f"[Epoch {epoch}/{epochs}] Loss G cyc: {avg_loss_G_cyc:.4f}|Loss G id: {avg_loss_G_id:.4f}  | Loss D: {avg_loss_D:.4f} |  PSD Error: {psd_err:.4f} | Hist Error: {hist_err:.4f}")
        # 3. Log to CSV

        manager.log_metrics(epoch, 
                            avg_loss_G, 
                            avg_loss_D, 
                            avg_loss_G_GAN, 
                            avg_loss_G_cyc, 
                            avg_loss_G_id, 
                            psd_err, hist_err)
    
        # 4. Save Checkpoint (Weights + Optimizer state)
        if epoch%10==0 or epoch ==epochs  : 
            checkpoint_state = {
                'epoch': epoch,
                'G_AB': G_AB.state_dict(),
                'G_BA': G_BA.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'opt_G': optimizer_G.state_dict(),
                'opt_D_A': optimizer_D_A.state_dict(),
                'opt_D_B': optimizer_D_B.state_dict()
            }
            manager.save_checkpoint(checkpoint_state, epoch)
            # --- END OF EPOCH STATS ---
        
            avg_data_time = data_time_sum / len(dataloader)
            avg_batch_time = batch_time_sum / len(dataloader)
            
            # Calculate Percentage of time spent loading data (Diagnostics)
            # If this is high (>10-20%), you need more num_workers in DataLoader
            load_overhead = (avg_data_time / avg_batch_time) * 100
    
            print(f"--------------------------------------------------")
            print(f"Epoch {epoch} Completed in {epoch_duration:.2f} sec")
            print(f"  > Avg Data Load Time: {avg_data_time:.4f} sec ({load_overhead:.1f}%)")
            print(f"  > Avg Batch Proc Time: {avg_batch_time:.4f} sec")
            print(f"--------------------------------------------------")
        
        

        

if __name__ == "__main__":
    # Replace these with your actual .pt file paths
    # If files don't exist, the script will generate random noise to demonstrate functionality.
    workdir ="/pscratch/sd/g/giuspugl/workstation/CO_network/extending_CO" 
    import glob 
    import re
    chks = (glob . glob (f"{workdir}/new_experiment/checkpoint_epoch_*" )) 
    # Regex to extract the epoch number
    def extract_epoch(filename):
        match = re.search(r"checkpoint_epoch_(\d+).pth", filename)
        return int(match.group(1)) if match else -1

    # Sort based on epoch number (Descending)
    latest_chkp  = max(chks, key=extract_epoch)

    
    train(f"{workdir}/fileA.pt", f"{workdir}/fileB.pt", 
          epochs=96, save_dir = f"{workdir}/new_experiment" ,resume_path=latest_chkp ) 

    #train(  epochs=1, save_dir = f"{workdir}/new_experiment",
    #      resume_path=f"{workdir}/new_experiment/checkpoint_epoch_1.pth")