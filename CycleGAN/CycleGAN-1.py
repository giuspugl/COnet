import torch
import torch.nn as nn
import random

class ReplayBuffer:
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

        
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x) # The "Skip Connection"

class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=6): # Default changed to 6
        super(Generator, self).__init__()
        channels = input_shape[0] # This will now extract '2'

        # --- Initial Convolution Block ---
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # --- Downsampling (Encoding) ---
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # --- Residual Blocks ---
        # Using 6 blocks for 128x128 resolution
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # --- Upsampling (Decoding) ---
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # --- Output Layer ---
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, channels, 7), # Outputs 2 channels
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # First block does not use Instance Normalization
            *discriminator_block(channels, 64, normalize=False),
            
            # Subsequent blocks increase channels
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            
            # Zero padding to maintain shape for the final classification
            nn.ZeroPad2d((1, 0, 1, 0)),
            
            # Collapses to 1 channel prediction map
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

def train_one_epoch(dataloader):
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        # ---------------------------------------------------------
        # DYNAMIC SHAPE FIX
        # Run a dummy pass or check shape of D(real_A) 
        # to ensure targets match the patch output size (e.g., 1x8x8)
        # ---------------------------------------------------------
        out_shape = D_A(real_A).shape 
        
        valid = torch.ones(out_shape, requires_grad=False).to(device)
        fake = torch.zeros(out_shape, requires_grad=False).to(device)
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # 1. Identity Loss
        # G_BA(A) should equal A if fed A
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        # G_AB(B) should equal B if fed B
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # 2. GAN Loss (Adversarial)
        # GAN loss for G_AB (A -> B) trying to fool D_B
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        
        # GAN loss for G_BA (B -> A) trying to fool D_A
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # 3. Cycle Consistency Loss
        # A -> Fake B -> Rec A
        rec_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(rec_A, real_A)
        
        # B -> Fake A -> Rec B
        rec_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(rec_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total Generator Loss
        loss_G = loss_GAN + (lambda_cyc * loss_cycle) + (lambda_id * loss_identity)
        
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (using buffer to sample past fakes)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake) # .detach() helps stop gradients flowing to G
        
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        
        if i % 100 == 0:
            print(f"[Batch {i}] [D loss: {loss_D_A.item() + loss_D_B.item()}] [G loss: {loss_G.item()}]")





import torch
import itertools
from torch.autograd import Variable

input_shape = (2, 128, 128)

# Initialize Generators
G_AB = Generator(input_shape) # Transforms Domain A -> Domain B
G_BA = Generator(input_shape) # Transforms Domain B -> Domain A

# Initialize Discriminators
D_A = Discriminator(input_shape) # Checks Real A vs Generated A
D_B = Discriminator(input_shape) # Checks Real B vs Generated B




# --- Hyperparameters ---
lr = 0.0002
b1 = 0.5
b2 = 0.999
lambda_cyc = 10.0  # Cycle loss weight (very high importance)
lambda_id = 5.0    # Identity loss weight

# --- Loss Functions ---
criterion_GAN = torch.nn.MSELoss() # For adversarial loss
criterion_cycle = torch.nn.L1Loss() # For cycle consistency
criterion_identity = torch.nn.L1Loss() # For identity preservation

# --- Optimizers ---
# We combine the parameters of both Generators into one optimizer
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)

# Discriminators have their own optimizers
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))



# Initialize buffers
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to device
G_AB.to(device); G_BA.to(device); D_A.to(device); D_B.to(device)

from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticCycleGANDataset(Dataset):
    def __init__(self, length=1000, shape=(2, 128, 128)):
        """
        Args:
            length (int): How many virtual 'images' to define per epoch.
            shape (tuple): The (Channels, Height, Width) of the data.
        """
        self.length = length
        self.shape = shape
        self.channels, self.h, self.w = shape

    def _generate_domain_A_sample(self):
        """Generates Domain A: Vertical Gradients in Channel 0, Noise in Channel 1"""
        # Channel 0: Vertical Gradient (0 to 1)
        y = torch.linspace(0, 1, self.h).view(self.h, 1).expand(self.h, self.w)
        
        # Channel 1: Random Gaussian Noise
        noise = torch.randn(self.h, self.w)
        
        return torch.stack([y, noise])

    def _generate_domain_B_sample(self):
        """Generates Domain B: Horizontal Gradients in Channel 0, Noise in Channel 1"""
        # Channel 0: Horizontal Gradient (0 to 1)
        x = torch.linspace(0, 1, self.w).view(1, self.w).expand(self.h, self.w)
        
        # Channel 1: Random Uniform Noise
        noise = torch.rand(self.h, self.w)
        
        return torch.stack([x, noise])

    def __getitem__(self, index):
        # Since CycleGAN is unpaired, index A and index B don't need to relate.
        # In a real file-based loader, you might randomize the index for B.
        
        data_A = self._generate_domain_A_sample()
        data_B = self._generate_domain_B_sample()

        # Return dictionary as expected by the training loop
        return {'A': data_A, 'B': data_B}

    def __len__(self):
        return self.length

# --- How to Instantiate ---

# 1. Create Dataset
dataset = SyntheticCycleGANDataset(length=100, shape=(2, 128, 128))

# 2. Create DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=4,       # Standard batch size for CycleGAN is 1 or 4
    shuffle=True, 
    num_workers=0       # Set to >0 if using real files/GPU
)

# --- Verification Test ---
# Let's pull one batch to verify shapes and verify the pipeline is ready.
for i, batch in enumerate(dataloader):
    real_A = batch['A']
    real_B = batch['B']
    
    print(f"Batch {i} Loaded Successfully")
    print(f"Domain A Shape: {real_A.shape}")  # Should be [4, 2, 128, 128]
    print(f"Domain B Shape: {real_B.shape}")  # Should be [4, 2, 128, 128]
    
    # Run a quick pass through the generator we defined earlier to ensure no crashes
    # Assuming G_AB is already instantiated from previous steps
    if 'G_AB' in globals():
        fake_B = G_AB(real_A)
        print(f"Generator Pass Successful. Output Shape: {fake_B.shape}")
    
    break # Stop after one batch