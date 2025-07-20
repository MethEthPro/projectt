import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import SARToEODataset
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Generator ----------
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(Generator, self).__init__()
        def down(in_feat, out_feat): return nn.Sequential(
            nn.Conv2d(in_feat, out_feat, 4, 2, 1), nn.BatchNorm2d(out_feat), nn.ReLU(True))
        def up(in_feat, out_feat): return nn.Sequential(
            nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1), nn.BatchNorm2d(out_feat), nn.ReLU(True))

        self.encoder = nn.Sequential(
            down(in_channels, features),
            down(features, features*2),
            down(features*2, features*4),
            down(features*4, features*8),
        )
        self.decoder = nn.Sequential(
            up(features*8, features*4),
            up(features*4, features*2),
            up(features*2, features),
            nn.ConvTranspose2d(features, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---------- Discriminator ----------
class Discriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features*2, 4, 2, 1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2, features*4, 4, 2, 1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*4, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# ---------- Training Loop ----------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SARToEODataset(
        root_dir="C:/Users/ahuja/OneDrive/Desktop/coding/python/summer/WHU-SEN-City/train",
        transform=None
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    checkpoint_path = "Project1_SAR_to_EO/checkpoint.pth"
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        print("🔁 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        opt_G.load_state_dict(checkpoint['opt_G'])
        opt_D.load_state_dict(checkpoint['opt_D'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, 31):
        for i, (s1, s2, file_path) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            file_path = str(file_path)  # 🔒 Force it to plain string
            city_name = os.path.basename(os.path.dirname(file_path))
            file_id = os.path.splitext(os.path.basename(file_path))[0]


            s1, s2 = s1.to(device), s2.to(device)

            # Train Discriminator
            opt_D.zero_grad()
            real_pred = D(s1, s2)
            real_loss = criterion_GAN(real_pred, torch.ones_like(real_pred))

            fake_s2 = G(s1)
            fake_pred = D(s1, fake_s2.detach())
            fake_loss = criterion_GAN(fake_pred, torch.zeros_like(fake_pred))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            fake_pred = D(s1, fake_s2)
            g_adv = criterion_GAN(fake_pred, torch.ones_like(fake_pred))
            g_l1 = criterion_L1(fake_s2, s2)
            g_loss = g_adv + 100 * g_l1
            g_loss.backward()
            opt_G.step()

            # Save image samples
            if i % 10 == 0:
                city_name = os.path.basename(os.path.dirname(file_path))  # e.g., changsha
                file_id = os.path.splitext(os.path.basename(file_path))[0]  # e.g., 0

                # Folder: generated_samples/epoch_1/changsha/0/
                sample_dir = os.path.join(
                    "Project1_SAR_to_EO", "generated_samples", f"epoch_{epoch}", city_name, file_id
                )
                os.makedirs(sample_dir, exist_ok=True)

                save_image(s1[0], os.path.join(sample_dir, "input.png"))
                save_image(s2[0], os.path.join(sample_dir, "target.png"))
                save_image(fake_s2[0], os.path.join(sample_dir, "fake.png"))

                print(f"[Epoch {epoch} | Step {i}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'opt_G': opt_G.state_dict(),
                    'opt_D': opt_D.state_dict(),
                }, checkpoint_path)

if __name__ == "__main__":
    train()
