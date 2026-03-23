import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from model import Generator, Critic, DEVICE, LATENT_DIM, BATCH_SIZE
import base64
import io
import time
from PIL import Image

# Default Hyperparameters for training
DEFAULT_EPOCHS = 100
LR = 0.00005
CLIP_VALUE = 0.01
CRITIC_ITER = 5

class WGANTrainer:
    def __init__(self):
        self.generator = Generator().to(DEVICE)
        self.critic = Critic().to(DEVICE)
        
        # Load weights if they exist
        if os.path.exists("checkpoints/generator.pth"):
            try:
                self.generator.load_state_dict(torch.load("checkpoints/generator.pth", map_location=DEVICE))
            except Exception as e:
                print(f"Warning: Could not load generator checkpoint - {e}")
                
        if os.path.exists("checkpoints/critic.pth"):
            try:
                self.critic.load_state_dict(torch.load("checkpoints/critic.pth", map_location=DEVICE))
            except Exception as e:
                print(f"Warning: Could not load critic checkpoint - {e}")

        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=LR)
        self.optimizer_C = optim.RMSprop(self.critic.parameters(), lr=LR)

        # Dataset initialization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        self.data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Training state
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = DEFAULT_EPOCHS
        self.current_step = 0
        self.total_steps = len(self.data_loader)
        self.latest_loss_C = 0.0
        self.latest_loss_G = 0.0
        self.latest_prob_real = 0.0
        self.latest_prob_fake = 0.0
        self.latest_samples_base64 = ""
        
        # Training history for charts
        self.loss_history_C = []
        self.loss_history_G = []
        self.prob_history_real = []
        self.prob_history_fake = []
        self.history_labels = []
        self.history_counter = 0
        
        # Timing for ETA
        self.epoch_start_time = 0
        self.training_start_time = 0
        self.eta_seconds = 0
        self.elapsed_seconds = 0
        
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("samples", exist_ok=True)

    def reset_history(self):
        """Clear training history for a fresh run"""
        self.loss_history_C = []
        self.loss_history_G = []
        self.prob_history_real = []
        self.prob_history_fake = []
        self.history_labels = []
        self.history_counter = 0

    def train_loop(self, num_epochs=None):
        """ Blocking training loop meant to run in a background thread """
        if num_epochs is not None:
            self.total_epochs = num_epochs
        
        self.is_training = True
        self.reset_history()
        self.training_start_time = time.time()
        
        for epoch in range(1, self.total_epochs + 1):
            if not self.is_training:
                break
                
            self.current_epoch = epoch
            self.epoch_start_time = time.time()
            
            for i, (real_imgs, _) in enumerate(self.data_loader, 1):
                if not self.is_training:
                    break
                    
                self.current_step = i
                real_imgs = real_imgs.to(DEVICE)
                bs = real_imgs.size(0)

                # Train Critic
                for _ in range(CRITIC_ITER):
                    z = torch.randn(bs, LATENT_DIM, device=DEVICE)
                    fake_imgs = self.generator(z).detach()

                    real_score = self.critic(real_imgs).mean()
                    fake_score = self.critic(fake_imgs).mean()
                    loss_C = -(real_score - fake_score)

                    self.optimizer_C.zero_grad()
                    loss_C.backward()
                    self.optimizer_C.step()

                    for p in self.critic.parameters():
                        p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

                # Train Generator
                z = torch.randn(bs, LATENT_DIM, device=DEVICE)
                gen_imgs = self.generator(z)
                loss_G = -self.critic(gen_imgs).mean()

                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()
                
                self.latest_loss_C = loss_C.item()
                self.latest_loss_G = loss_G.item()
                
                # For WGAN, the critic outputs a raw unbounded score, not a probability.
                # To display a "probability distribution" (0 to 1), we apply a Sigmoid.
                self.latest_prob_real = torch.sigmoid(self.critic(real_imgs)).mean().item()
                self.latest_prob_fake = torch.sigmoid(self.critic(gen_imgs.detach())).mean().item()

                # Record history every 50 steps for smooth charts without memory overload
                if i % 50 == 0:
                    self.history_counter += 1
                    self.loss_history_C.append(round(self.latest_loss_C, 4))
                    self.loss_history_G.append(round(self.latest_loss_G, 4))
                    self.prob_history_real.append(round(self.latest_prob_real, 4))
                    self.prob_history_fake.append(round(self.latest_prob_fake, 4))
                    self.history_labels.append(f"E{epoch}S{i}")

                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{self.total_epochs}][Step {i}/{self.total_steps}] "
                          f"Critic Loss: {loss_C.item():.3f} | Gen Loss: {loss_G.item():.3f}")
                    # Generate sample for frontend visualization
                    self.generate_sample_for_frontend()
            
            # Calculate ETA
            elapsed = time.time() - self.training_start_time
            self.elapsed_seconds = elapsed
            if epoch > 0:
                avg_epoch_time = elapsed / epoch
                remaining_epochs = self.total_epochs - epoch
                self.eta_seconds = avg_epoch_time * remaining_epochs

            # End of epoch: save checkpoints
            torch.save(self.generator.state_dict(), "checkpoints/generator.pth")
            torch.save(self.critic.state_dict(), "checkpoints/critic.pth")
            print(f"Checkpoint saved for Epoch {epoch}/{self.total_epochs}")

        self.is_training = False
        
    def stop_training(self):
        self.is_training = False
        
    def generate_sample_for_frontend(self):
        """Generates a batch of images and saves them as a base64 string for the UI"""
        with torch.no_grad():
            z = torch.randn(16, LATENT_DIM, device=DEVICE)
            samples = self.generator(z)
            samples = samples * 0.5 + 0.5 # Rescale to [0, 1]
            
            # Save to memory buffer
            grid = utils.make_grid(samples, nrow=4)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            self.latest_samples_base64 = img_str

trainer = WGANTrainer()
