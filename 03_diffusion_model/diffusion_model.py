import torch
import torch.nn as nn
import torch.optim as optim

class DiffusionModel(nn.Module):
    def __init__(self, input_size, latent_dim, num_layers):
        super(DiffusionModel, self).__init__()
        
        # Encoder (Contracting path)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),  # Latent space
        )
        
        # Decoder (Expanding path)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),  # Reconstructed output
        )
        
        # Denoising block (diffusion reverse process)
        self.denoise = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),  # Map to latent space
        )

        # Noise generation (forward diffusion)
        self.noise_layer = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        # Forward pass (diffusion process: add noise)
        noisy_data = self.add_noise(x)
        
        # Encoder: compress to latent space
        latent = self.encoder(noisy_data)
        
        # Denoising block (reverse diffusion)
        denoised_latent = self.denoise(latent)
        
        # Decoder: reconstruct the data
        output = self.decoder(denoised_latent)
        
        return output
    
    def add_noise(self, x):
        # Add random noise during the forward diffusion step
        noise = torch.randn_like(x) * 0.1  # Gaussian noise with scale 0.1
        noisy_data = x + noise  # Add noise to input data
        return noisy_data
    
    def denoise_step(self, x):
        # Denoising step for reversing the noise
        denoised = self.denoise(x)
        return denoised

# Example: Let's say we're working with 1D polymer SMILES sequences (for simplicity)
input_size = 100  # e.g., SMILES length (you can adjust this)
latent_dim = 64  # Latent space dimension
num_layers = 3  # Number of layers in encoder-decoder

# Create model
model = DiffusionModel(input_size=input_size, latent_dim=latent_dim, num_layers=num_layers)

# Example input (random polymer SMILES encoded as integers)
x = torch.randn(1, input_size)  # 1 sample, 100 input features (e.g., SMILES length)

# Forward pass
output = model(x)
print(output.shape)  # Should output the same shape as the input (1, 100)

# Loss & Optimization
criterion = nn.MSELoss()  # Assuming we use Mean Squared Error (MSE) for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy target for demonstration (real target would be your original SSE structure)
target = torch.randn_like(x)

# Training step
optimizer.zero_grad()
loss = criterion(output, target)
loss.backward()
optimizer.step()

print(f"Training Loss: {loss.item()}")