import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation
from scipy.ndimage import gaussian_filter

# Settings
os.makedirs('results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Simulation grid (intentionally kept small for speed)
N = 64  # spatial grid for optical propagation
physical_size = 2e-3  # 2 mm aperture
dx = physical_size / N

# Fresnel/Angular spectrum parameters
z = 5e-3  # propagation distance (5 mm)
wavelength = 550e-9  # central (use single-wavelength simulation for speed)

# Training parameters
batch_size = 32
epochs = 20  # Increased for better convergence
lr = 1e-3

# 1. Dataset (CIFAR-10 -> grayscale 64x64)
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((N, N)),
    T.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Slightly larger subsets for better training
train_subset = torch.utils.data.Subset(train_set, list(range(0, 5000)))
val_subset = torch.utils.data.Subset(val_set, list(range(0, 1000)))

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)


# 2. Optical Forward Model

# Precompute transfer function for angular spectrum (single wavelength, fully in Torch)

fx = torch.fft.fftfreq(N, d=dx).to(device)
FX, FY = torch.meshgrid(fx, fx, indexing='ij')
k = 2 * torch.pi / wavelength
term = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
term = torch.clamp(term, min=0.0)
H = torch.exp(1j * k * z * torch.sqrt(term))

def angular_spectrum(u_in):
    """Propagate complex field u_in (torch complex tensor) using precomputed H."""
    # u_in: complex tensor, shape (B, N, N) stored as complex64
    U = torch.fft.fft2(u_in)
    U = U * H
    u_out = torch.fft.ifft2(U)
    return u_out

def forward_model_intensity(obj_batch, mask):
    """Given object intensity (B,1,N,N) and mask (B,N,N), return sensor intensity (B,1,N,N).
    Model: field ~ sqrt(obj) * exp(1j * mask_phase) -> propagate -> intensity
    """
    B = obj_batch.shape[0]
    # amplitude field
    amp = torch.sqrt(torch.clamp(obj_batch, min=1e-6))  # (B,1,N,N)
    amp = amp.squeeze(1)  # (B,N,N)
    # apply phase mask
    coded_complex = amp * torch.exp(1j * mask.type(torch.complex64))  # (B,N,N) complex
    # propagate
    propagated = angular_spectrum(coded_complex)
    intensity = (torch.abs(propagated) ** 2).unsqueeze(1)
    # normalize per image
    intensity = intensity / (intensity.amax(dim=(2,3), keepdim=True) + 1e-12)
    return intensity


# 3. Create Random Coded Phase Masks

# Create a small bank of smooth phase masks ([-pi, pi]) to augment training
num_masks = 8
masks = []
for i in range(num_masks):
    m = np.random.rand(N, N)
    m = gaussian_filter(m, sigma=3)  # Smooth for physical plausibility
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)  # Normalize to [0,1]
    m = m * 2 * np.pi - np.pi  # Scale to [-pi, pi]
    masks.append(torch.from_numpy(m.astype(np.float32)).to(device))
masks = torch.stack(masks)  # (M,N,N)


# 4. Simple CNN (U-Net-lite)

class UNetLite(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, base,3,padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2,3,padding=1), nn.ReLU(), nn.Conv2d(base*2, base*2,3,padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base,3,padding=1), nn.ReLU(), nn.Conv2d(base, base,3,padding=1), nn.ReLU())
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u1 = self.up1(e2)
        cat = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat)
        return torch.sigmoid(self.outc(d1))

model = UNetLite().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# 5. Training Loop

train_losses = []
val_losses = []
sample_vis = []  # store reconstruction snapshots for animation

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_count = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    for batch in pbar:
        imgs, _ = batch  # imgs: (B,1,N,N)
        imgs = imgs.to(device)
        B = imgs.shape[0]
        # sample different masks per image
        idx = np.random.randint(0, num_masks, size=B)
        mask = masks[idx]  # (B, N, N)
        # forward sim
        sensor = forward_model_intensity(imgs, mask)
        # model input: sensor intensity
        pred = model(sensor)
        loss = criterion(pred, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * B
        train_count += B
        pbar.set_postfix({'train_loss': loss.item()})

    train_losses.append(train_loss / train_count)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs, _ = batch
            imgs = imgs.to(device)
            B = imgs.shape[0]
            idx = np.random.randint(0, num_masks, size=B)
            mask = masks[idx]
            sensor = forward_model_intensity(imgs, mask)
            pred = model(sensor)
            loss = criterion(pred, imgs)
            val_loss += loss.item() * B
            val_count += B

    val_losses.append(val_loss / val_count)
    print(f'Epoch {epoch+1}: Val Loss = {val_losses[-1]:.4f}')

    # Validation snapshot: take first 6 validation images and record reconstructions
    with torch.no_grad():
        batch = next(iter(val_loader))
        imgs_val, _ = batch
        imgs_val = imgs_val[:6].to(device)
        idx = np.random.randint(0, num_masks, size=6)
        mask = masks[idx]
        sensor_val = forward_model_intensity(imgs_val, mask)
        recon = model(sensor_val)
        # store for animation: stack (input, recon, target)
        stacked = torch.cat([sensor_val.cpu(), recon.cpu(), imgs_val.cpu()], dim=0)
        sample_vis.append(stacked.numpy())
        # save example reconstruction images for this epoch
        fig, axes = plt.subplots(3,6, figsize=(12,6))
        for i in range(6):
            axes[0,i].imshow(sensor_val[i,0].cpu().numpy(), cmap='gray')
            axes[0,i].axis('off')
            axes[1,i].imshow(recon[i,0].cpu().numpy(), cmap='gray')
            axes[1,i].axis('off')
            axes[2,i].imshow(imgs_val[i,0].cpu().numpy(), cmap='gray')
            axes[2,i].axis('off')
        axes[0,0].set_title('Sensor')
        axes[1,0].set_title('Recon')
        axes[2,0].set_title('Target')
        plt.suptitle(f'Epoch {epoch+1} Reconstructions')
        plt.tight_layout()
        plt.savefig(f'results/recon_epoch_{epoch+1}.png', dpi=200, bbox_inches='tight')
        plt.close()



plt.figure()
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/loss_curve.png', dpi=300)
plt.close()

# 7. Make Animation of Reconstructions

frames = []
fig = plt.figure(figsize=(10,6))
for snap in sample_vis:
    # snap shape: (3*6,1,N,N) -> first 6 are sensors, next 6 recon, next 6 targets
    grid = np.zeros((3*N, 6*N))
    for col in range(6):
        sensor = snap[col,0]
        recon = snap[6+col,0]
        target = snap[12+col,0]
        grid[0:N, col*N:(col+1)*N] = sensor
        grid[N:2*N, col*N:(col+1)*N] = recon
        grid[2*N:3*N, col*N:(col+1)*N] = target
    im = plt.imshow(grid, cmap='gray', animated=True)
    plt.axis('off')
    frames.append([im])
ani = animation.ArtistAnimation(fig, frames, interval=700, blit=True)
ani.save('results/reconstructions_evolution.gif', writer='pillow')
plt.close()


# 8. Save model weights and loss history

torch.save(model.state_dict(), 'results/unetlite_recon.pth')
np.save('results/loss_history.npy', np.array([train_losses, val_losses]))

print('Done. Results saved in /results: sample recon images, loss curve, animation, and model.')