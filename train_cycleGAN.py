import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import SARToEODataset
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

# ---------- Generator (U-Net Architecture) ----------
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(Generator, self).__init__()
        
        def down(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)
        
        def up(in_feat, out_feat, dropout=False):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1),
                     nn.BatchNorm2d(out_feat),
                     nn.ReLU(True)]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder (downsampling)
        self.down1 = down(in_channels, features, normalize=False)  # 256->128
        self.down2 = down(features, features*2)                   # 128->64
        self.down3 = down(features*2, features*4)                 # 64->32
        self.down4 = down(features*4, features*8)                 # 32->16
        self.down5 = down(features*8, features*8)                 # 16->8
        self.down6 = down(features*8, features*8)                 # 8->4
        self.down7 = down(features*8, features*8)                 # 4->2
        self.down8 = down(features*8, features*8, normalize=False) # 2->1

        # Decoder (upsampling with skip connections)
        self.up1 = up(features*8, features*8, dropout=True)       # 1->2
        self.up2 = up(features*16, features*8, dropout=True)      # 2->4 (16 = 8+8 from skip)
        self.up3 = up(features*16, features*8, dropout=True)      # 4->8
        self.up4 = up(features*16, features*8)                    # 8->16
        self.up5 = up(features*16, features*4)                    # 16->32
        self.up6 = up(features*8, features*2)                     # 32->64
        self.up7 = up(features*4, features)                       # 64->128
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final(torch.cat([u7, d1], 1))

# ---------- Discriminator (PatchGAN) ----------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(in_channels, features, normalize=False),    # 256->128
            discriminator_block(features, features*2),                      # 128->64
            discriminator_block(features*2, features*4),                    # 64->32
            discriminator_block(features*4, features*8),                    # 32->16
            nn.Conv2d(features*8, 1, 4, 2, 1),                            # 16->8 (changed stride to 2)
        )

    def forward(self, x):
        return self.model(x)

# ---------- S-CycleGAN Training Loop ----------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SARToEODataset(
        root_dir="C:/Users/ahuja/OneDrive/Desktop/coding/python/summer/WHU-SEN-City/train",
        transform=None
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize networks
    G_SAR2EO = Generator(in_channels=3, out_channels=3).to(device)  # SAR to Optical
    G_EO2SAR = Generator(in_channels=3, out_channels=3).to(device)  # Optical to SAR
    D_EO = Discriminator(in_channels=3).to(device)                  # Discriminator for optical images
    D_SAR = Discriminator(in_channels=3).to(device)                 # Discriminator for SAR images

    # Loss functions
    criterion_GAN = nn.MSELoss()  # Using MSE instead of BCE for better stability
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(G_SAR2EO.parameters(), G_EO2SAR.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D_EO = optim.Adam(D_EO.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_SAR = optim.Adam(D_SAR.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Learning rate schedulers
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    lr_scheduler_D_EO = optim.lr_scheduler.LambdaLR(
        optimizer_D_EO, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    lr_scheduler_D_SAR = optim.lr_scheduler.LambdaLR(
        optimizer_D_SAR, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    checkpoint_path = "Project1_SAR_to_EO/s_cyclegan_checkpoint.pth"
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        print("🔁 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        G_SAR2EO.load_state_dict(checkpoint['G_SAR2EO'])
        G_EO2SAR.load_state_dict(checkpoint['G_EO2SAR'])
        D_EO.load_state_dict(checkpoint['D_EO'])
        D_SAR.load_state_dict(checkpoint['D_SAR'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D_EO.load_state_dict(checkpoint['optimizer_D_EO'])
        optimizer_D_SAR.load_state_dict(checkpoint['optimizer_D_SAR'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming from epoch {start_epoch}")

    # Hyperparameters from paper
    lambda_cycle = 100  # λ = 100
    lambda_identity = 100  # β = 100

    for epoch in range(start_epoch, 201):  # Train for 200 epochs as mentioned in paper
        for i, (sar_img, eo_img, file_path) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            sar_img, eo_img = sar_img.to(device), eo_img.to(device)
            
            # Get discriminator output size dynamically for the first batch
            if i == 0 and epoch == start_epoch:
                with torch.no_grad():
                    sample_output = D_EO(sar_img)
                    patch_h, patch_w = sample_output.shape[2], sample_output.shape[3]
                print(f"Discriminator output size: {patch_h}x{patch_w}")
            
            # Helper function to create labels with correct size
            def create_labels(batch_size, device):
                with torch.no_grad():
                    sample_output = D_EO(sar_img)
                    h, w = sample_output.shape[2], sample_output.shape[3]
                real_label = torch.ones((batch_size, 1, h, w), device=device, requires_grad=False)
                fake_label = torch.zeros((batch_size, 1, h, w), device=device, requires_grad=False)
                return real_label, fake_label
            
            real_label, fake_label = create_labels(sar_img.size(0), device)

            # ------------------
            # Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss
            # G_SAR2EO should be identity if real EO images are passed
            identity_eo = G_SAR2EO(eo_img)
            loss_identity_eo = criterion_identity(identity_eo, eo_img) * lambda_identity
            
            # G_EO2SAR should be identity if real SAR images are passed
            identity_sar = G_EO2SAR(sar_img)
            loss_identity_sar = criterion_identity(identity_sar, sar_img) * lambda_identity

            # GAN losses
            fake_eo = G_SAR2EO(sar_img)
            pred_fake_eo = D_EO(fake_eo)
            loss_GAN_SAR2EO = criterion_GAN(pred_fake_eo, real_label)

            fake_sar = G_EO2SAR(eo_img)
            pred_fake_sar = D_SAR(fake_sar)
            loss_GAN_EO2SAR = criterion_GAN(pred_fake_sar, real_label)

            # Cycle consistency losses
            recovered_sar = G_EO2SAR(fake_eo)
            loss_cycle_SAR = criterion_cycle(recovered_sar, sar_img) * lambda_cycle

            recovered_eo = G_SAR2EO(fake_sar)
            loss_cycle_EO = criterion_cycle(recovered_eo, eo_img) * lambda_cycle

            # Total generator loss
            loss_G = (loss_identity_eo + loss_identity_sar + 
                     loss_GAN_SAR2EO + loss_GAN_EO2SAR + 
                     loss_cycle_SAR + loss_cycle_EO)

            loss_G.backward()
            optimizer_G.step()

            # ------------------
            # Train Discriminator EO
            # ------------------
            optimizer_D_EO.zero_grad()

            # Real loss
            pred_real = D_EO(eo_img)
            loss_D_real = criterion_GAN(pred_real, real_label)

            # Fake loss
            pred_fake = D_EO(fake_eo.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake_label)

            # Total loss
            loss_D_EO = (loss_D_real + loss_D_fake) * 0.5

            loss_D_EO.backward()
            optimizer_D_EO.step()

            # ------------------
            # Train Discriminator SAR
            # ------------------
            optimizer_D_SAR.zero_grad()

            # Real loss
            pred_real = D_SAR(sar_img)
            loss_D_real = criterion_GAN(pred_real, real_label)

            # Fake loss
            pred_fake = D_SAR(fake_sar.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake_label)

            # Total loss
            loss_D_SAR = (loss_D_real + loss_D_fake) * 0.5

            loss_D_SAR.backward()
            optimizer_D_SAR.step()

            # ------------------
            # Logging and Saving
            # ------------------
            if i % 10 == 0:
                file_path = str(file_path[0])  # Convert tuple to string
                city_name = os.path.basename(os.path.dirname(file_path))
                file_id = os.path.splitext(os.path.basename(file_path))[0]

                # Create sample directory
                sample_dir = os.path.join(
                    "Project1_SAR_to_EO", "s_cyclegan_samples", f"epoch_{epoch}", city_name, file_id
                )
                os.makedirs(sample_dir, exist_ok=True)

                # Save images
                save_image(sar_img[0], os.path.join(sample_dir, "sar_real.png"), normalize=True)
                save_image(eo_img[0], os.path.join(sample_dir, "eo_real.png"), normalize=True)
                save_image(fake_eo[0], os.path.join(sample_dir, "eo_fake.png"), normalize=True)
                save_image(fake_sar[0], os.path.join(sample_dir, "sar_fake.png"), normalize=True)

                print(f"[Epoch {epoch} | Step {i}] "
                      f"G Loss: {loss_G.item():.4f} | "
                      f"D_EO Loss: {loss_D_EO.item():.4f} | "
                      f"D_SAR Loss: {loss_D_SAR.item():.4f}")

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_EO.step()
        lr_scheduler_D_SAR.step()

        # Save checkpoint every epoch
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'G_SAR2EO': G_SAR2EO.state_dict(),
                'G_EO2SAR': G_EO2SAR.state_dict(),
                'D_EO': D_EO.state_dict(),
                'D_SAR': D_SAR.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_EO': optimizer_D_EO.state_dict(),
                'optimizer_D_SAR': optimizer_D_SAR.state_dict(),
            }, checkpoint_path)

if __name__ == "__main__":
    train()