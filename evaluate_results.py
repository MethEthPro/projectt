import os
import torch
from torchvision.utils import save_image
from preprocess import SARToEODataset
from train_cycleGAN import Generator
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim(pred, target):
    pred = pred.squeeze().permute(1, 2, 0).cpu().numpy()
    target = target.squeeze().permute(1, 2, 0).cpu().numpy()
    return ssim(pred, target, data_range=1.0, channel_axis=2)

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SARToEODataset(root_dir="C:/Users/ahuja/OneDrive/Desktop/coding/python/summer/WHU-SEN-City/train", transform=None)
    G = Generator().to(device)
    G.load_state_dict(torch.load("generator.pth"))  # Load saved model
    G.eval()

    os.makedirs("Project1_SAR_to_EO/generated_samples", exist_ok=True)

    with torch.no_grad():
        for i in range(5):
            s1, s2 = dataset[i]
            s1, s2 = s1.unsqueeze(0).to(device), s2.unsqueeze(0).to(device)
            fake = G(s1)
            save_image(fake[0], f"Project1_SAR_to_EO/generated_samples/sample_{i}_fake.png")
            save_image(s1[0], f"Project1_SAR_to_EO/generated_samples/sample_{i}_input.png")
            save_image(s2[0], f"Project1_SAR_to_EO/generated_samples/sample_{i}_target.png")

            score = compute_ssim(fake[0], s2[0])
            print(f"Sample {i}: SSIM = {score:.4f}")

if __name__ == "__main__":
    evaluate()
