from PIL import Image
from torch.optim import Adam, SGD
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from .point_splatting import Points, scale_then_crop, sobol_filter, float_to_uint8
import torchvision.transforms.functional as F
from torchvision.io import write_png, write_jpeg
import shutil


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

CANVAS_HEIGHT_PX = 384
CANVAS_WIDTH_PX = 256
NUM_POINTS = 2048
NUM_ITERATIONS = 8192
IMAGE = "monalisa.jpg"  # Path to the target image
TIMELAPSE = True

# Ensure the output directory exists
output_dir = Path("outputs")
shutil.rmtree(output_dir, ignore_errors=True)  # Clear previous outputs
output_dir.mkdir(parents=True, exist_ok=True)


target_image = Image.open(IMAGE).convert("RGB")
target_image = F.to_tensor(target_image)  # Add batch dimension


print(
    f"height: {CANVAS_HEIGHT_PX}, width: {CANVAS_WIDTH_PX}, num_points: {NUM_POINTS}, image: {IMAGE}"
)

target_image = scale_then_crop(
    target_image, CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX
)  # Resize and crop to desired size
target_image = target_image.to(device)
write_png(float_to_uint8(target_image), "resized_target.png")  # Save the target image
write_jpeg(
    float_to_uint8(target_image), "resized_target.jpg"
)  # Save the target image in JPEG format
target_grad = sobol_filter(target_image)  # Apply Sobol filter
target_double_grad = sobol_filter(target_grad)  # Apply Sobol filter again

points = Points(
    num_points=NUM_POINTS,
    canvas_height=CANVAS_HEIGHT_PX,
    canvas_width=CANVAS_WIDTH_PX,
    device=device,
)
points.train()

optimizer = Adam(points.parameters(), lr=0.01, betas=(0.9, 0.99))
pbar = tqdm(range(NUM_ITERATIONS), desc="Loss: 0.0000")
last_saved_loss = float("inf")
improvements = 0
steps_since_last_save = 0
for i in pbar:
    optimizer.zero_grad()
    canvas = points.render(CANVAS_HEIGHT_PX, CANVAS_WIDTH_PX)
    recon_l2_loss = torch.nn.functional.mse_loss(canvas, target_image)

    canvas_grad = sobol_filter(canvas)  # Apply Sobol filter to the rendered canvas
    canvas_double_grad = sobol_filter(
        canvas_grad
    )  # Apply Sobol filter again to the gradient
    recon_grad_l2_loss = torch.nn.functional.mse_loss(
        canvas_grad, target_grad
    )  # Compare Sobol filtered images
    recon_double_grad_l2_loss = torch.nn.functional.mse_loss(
        canvas_double_grad, target_double_grad
    )  # Compare Sobol filtered gradients

    total_loss = recon_l2_loss + recon_grad_l2_loss + recon_double_grad_l2_loss
    total_loss.backward()
    optimizer.step()
    pbar.set_description(
        f"ReconL2: {recon_l2_loss.item():.6f} | GradL2: {recon_grad_l2_loss.item():.6f} | DoubleGradL2: {recon_double_grad_l2_loss.item():.6f} Last Save: {steps_since_last_save}"
    )

    if total_loss.item() < last_saved_loss * 0.99:
        improvements += 1
        steps_since_last_save = 0
        last_saved_loss = total_loss.item()

        if TIMELAPSE:
            top = torch.cat(
                (
                    float_to_uint8(target_image),
                    float_to_uint8(canvas),
                ),
                dim=2,
            )
            center = torch.cat(
                (
                    float_to_uint8(target_grad),
                    float_to_uint8(canvas_grad),
                ),
                dim=2,
            )
            bottom = torch.cat(
                (
                    float_to_uint8(target_double_grad),
                    float_to_uint8(canvas_double_grad),
                ),
                dim=2,
            )
            # right = torch.cat(
            #     (
            #         float_to_uint8(torch.abs(target_image - canvas)),
            #         float_to_uint8(
            #             torch.abs(target_grad - canvas_grad),
            #         ),
            #         float_to_uint8(
            #             torch.abs(target_double_grad - canvas_double_grad),
            #         ),
            #     ),
            #     dim=1,
            # )
            combined = torch.cat((top, center, bottom), dim=1)

            write_jpeg(top, f"outputs/{improvements:05d}.jpg")
            write_png(combined, f"latest.png")

    else:
        steps_since_last_save += 1
