import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3*32*32),
            nn.Tanh()   # Output range [-1,1]
        )

    def forward(self, z):
        img_flat = self.model(z)
        return img_flat.view(z.size(0), 3, 32, 32)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)  # Scalar output
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        return self.model(flat)

def load_generator():
    gen = Generator()
    if os.path.exists('generator.pth'):
        try:
            gen.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
            print("Loaded default generator points.")
        except RuntimeError:
            print("Error loading generator.pth structure. Proceeding with initialized Generator.")
    return gen.to(DEVICE)