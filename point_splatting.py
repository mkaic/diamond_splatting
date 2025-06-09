import torch
from torch import Tensor
import torch.nn as nn
import math


def squared_euclidean_distance_from_origin(points: Tensor):
    """
    Computes the Euclidean distance between each pair of points in a batch.

    Args:
        points (Tensor): A tensor of shape (..., 2) representing the points.

    Returns:
        Tensor: A tensor of shape (...) containing the points' distances from the origin.
    """

    return torch.sum(points**2, dim=-1, keepdim=True)


def manhattan_distance_from_origin(points: Tensor):
    """
    Computes the Manhattan distance between each pair of points in a batch.

    Args:
        points (Tensor): A tensor of shape (..., 2) representing the points.

    Returns:
        Tensor: A tensor of shape (...) containing the points' distances from the origin.
    """

    return torch.sum(torch.abs(points), dim=-1, keepdim=True) / 2


class Points(nn.Module):
    def __init__(self, num_points, device="cpu"):
        super().__init__()
        self.device = device
        self.num_points = num_points
        self.locations = nn.Parameter(
            torch.rand(1, 1, num_points, 2, device=self.device) * 2 - 1
        )  # relative to canvas size

        # Initialize 2D transformation matrices with random rotations
        random_angles = (
            torch.rand(num_points, device=self.device, dtype=torch.float32)
            * 2
            * torch.pi
        )
        rotation_matrices = torch.stack(
            [
                torch.cos(random_angles),
                -torch.sin(random_angles),
                torch.sin(random_angles),
                torch.cos(random_angles),
            ],
            dim=-1,
        ).view(num_points, 2, 2)
        self.matrix_offsets = nn.Parameter(
            rotation_matrices.detach()
            - torch.eye(2, device=self.device, dtype=torch.float32)
        )
        self.matrix_scale_factor_offsets = nn.Parameter(
            torch.zeros(num_points, 1, 1, device=self.device, dtype=torch.float32)
        )
        self.colors = nn.Parameter(
            torch.rand(1, 1, num_points, 3, device=self.device)
        )  # Values to add or subtract from each channel of the canvas.
        self.alphas = nn.Parameter(
            torch.zeros(1, 1, num_points, 1, device=self.device, dtype=torch.float32)
        )

    def render(self, canvas_height: int, canvas_width: int) -> Tensor:
        """
        Render points on a canvas.

        Args:
            canvas_height (int): Height of the canvas.
            canvas_width (int): Width of the canvas.

        Returns:
            Tensor: A tensor representing the rendered points on the canvas.
        """

        untransformed_coordinates = torch.meshgrid(
            torch.linspace(
                -1, 1, canvas_height, dtype=torch.float32, device=self.device
            ),
            torch.linspace(
                -1, 1, canvas_width, dtype=torch.float32, device=self.device
            ),
            indexing="ij",
        )

        untransformed_coordinates = torch.stack(untransformed_coordinates, dim=-1)

        # Broadcast copy of coordinates for each point
        untransformed_coordinates = untransformed_coordinates.unsqueeze(-2).expand(
            -1, -1, self.num_points, 2
        )

        shifted_coordinates = untransformed_coordinates - self.locations

        # Coordinates are of shape (canvas_height, canvas_width, num_points, 2)
        # Rotation matrices are of shape (num_points, 2, 2)
        # Expand dimensions to allow batched matrix multiplication
        shifted_coordinates = shifted_coordinates.unsqueeze(
            -2
        )  # (H, W, num_points, 1, 2)
        transform_matrices = self.matrix_offsets + (
            torch.eye(2, device=self.device, dtype=torch.float32)
        ) * (
            math.sqrt(self.num_points)
            / 2  # the /2 is bc the canvas is 2 units wide (-1 to 1)
        ) * (
            torch.exp(self.matrix_scale_factor_offsets)
        )
        transform_matrices = transform_matrices.view(1, 1, num_points, 2, 2)
        transformed_coordinates = torch.matmul(
            shifted_coordinates, transform_matrices
        ).squeeze(
            -2
        )  # (H, W, num_points, 2)

        distances = manhattan_distance_from_origin(transformed_coordinates)

        # distances = squared_euclidean_distance_from_origin(transformed_coordinates)

        # mapped_distances = 1 / (1 + torch.pow(torch.sqrt(distances), 3))
        mapped_distances = torch.relu(1 - distances)

        canvas = self.colors * self.alphas * mapped_distances

        canvas = canvas.sum(dim=-2)  # (H, W, 3)
        canvas = torch.sigmoid(canvas * 4)

        return canvas


if __name__ == "__main__":
    device = torch.device("mps")
    from PIL import Image
    import torchvision.transforms.functional as F
    from torch.optim import Adam, SGD
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm

    canvas_height = 256
    canvas_width = 256
    num_points = 512

    # Ensure the output directory exists
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_image = Image.open("branos.jpg").convert("RGB")
    target_image = F.to_tensor(target_image)  # Add batch dimension

    target_image = F.resize(target_image, (canvas_height, canvas_width))
    target_image = target_image.permute(1, 2, 0)  # (Height, Width, Channels)
    target_image = target_image.to(device)

    points = Points(num_points=num_points, device=device)
    points.train()

    optimizer = Adam(points.parameters(), lr=0.001, betas=(0.9, 0.99))
    num_iterations = 10_000
    pbar = tqdm(range(num_iterations), desc="Loss: 0.0000")
    for i in pbar:
        optimizer.zero_grad()
        canvas = points.render(canvas_height, canvas_width)
        recon_l2_loss = torch.nn.functional.mse_loss(canvas, target_image)

        # ratio of scales should be close to 1 (log of ratio should be close to 0)
        # reg_loss = torch.mean(
        #     torch.abs(torch.log((points.scales[..., 0] / points.scales[..., 1])))
        # )
        total_loss = recon_l2_loss
        total_loss.backward()
        optimizer.step()
        pbar.set_description(f"L2 Loss: {recon_l2_loss.item():.6f}")

        if i % 100 == 0:
            canvas = canvas.detach().cpu()
            canvas = canvas * 255
            canvas = canvas.numpy().astype(np.uint8)
            image = Image.fromarray(canvas)
            image.save(f"outputs/{i:05d}.png")
            image.save(f"latest.png")
