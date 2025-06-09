from PIL import Image
from torch.optim import Adam, SGD
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from .point_splatting import Points, scale_then_crop, sobol_filter, float_to_uint8
import torchvision.transforms.functional as F
from torchvision.io import write_png


device = torch.device("mps")

CANVAS_HEIGHT_PX = 192
CANVAS_WIDTH_PX = 128
NUM_POINTS = 1024

# Ensure the output directory exists
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

target_image = Image.open("monalisa.jpg").convert("RGB")
target_image = F.to_tensor(target_image)  # Add batch dimension

target_image = scale_then_crop(
    target_image, CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX
)  # Resize and crop to desired size
target_image = target_image.to(device)
target_sobol = sobol_filter(target_image)  # Apply Sobol filter

points = Points(
    num_points=NUM_POINTS,
    canvas_height=CANVAS_HEIGHT_PX,
    canvas_width=CANVAS_WIDTH_PX,
    device=device,
)
points.train()

optimizer = Adam(points.parameters(), lr=0.001, betas=(0.9, 0.99))
num_iterations = 10_000
pbar = tqdm(range(num_iterations), desc="Loss: 0.0000")
last_saved_loss = float("inf")
improvements = 0
steps_since_last_save = 0
for i in pbar:
    optimizer.zero_grad()
    canvas = points.render(CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX)
    recon_l2_loss = torch.nn.functional.mse_loss(canvas, target_image)

    canvas_sobol = sobol_filter(canvas)  # Apply Sobol filter to the rendered canvas
    recon_sobol_l2_loss = torch.nn.functional.mse_loss(
        canvas_sobol, target_sobol
    )  # Compare Sobol filtered images

    total_loss = recon_l2_loss + recon_sobol_l2_loss
    total_loss.backward()
    optimizer.step()
    pbar.set_description(
        f"L2 Loss: {recon_l2_loss.item():.6f} | Sobol Loss: {recon_sobol_l2_loss.item():.6f} | Steps Since Last Save: {steps_since_last_save}"
    )

    if total_loss.item() < last_saved_loss * 0.99:
        improvements += 1
        steps_since_last_save = 0
        last_saved_loss = total_loss.item()

        left = torch.cat(
            (
                float_to_uint8(target_image),
                float_to_uint8(target_sobol),
            ),
            dim=1,
        )
        center = torch.cat(
            (
                float_to_uint8(canvas),
                float_to_uint8(canvas_sobol),
            ),
            dim=1,
        )
        right = torch.cat(
            (
                float_to_uint8(torch.abs(target_image - canvas)),
                float_to_uint8(
                    torch.abs(target_sobol - canvas_sobol),
                ),
            ),
            dim=1,
        )
        combined = torch.cat((left, center, right), dim=2)

        write_png(combined, f"outputs/{improvements:05d}.png")
        write_png(combined, f"latest.png")

    else:
        steps_since_last_save += 1
