import torch
from torch import Tensor
import torch.nn as nn
import math
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class Points(nn.Module):
    def __init__(self, num_points, canvas_height, canvas_width, device="cpu"):
        super().__init__()
        self.device = device
        self.num_points = num_points

        self.canvas_height = canvas_height
        self.canvas_width = canvas_width

        self.width_to_height_ratio = canvas_width / canvas_height

        self.minimum_width = -1 * self.width_to_height_ratio
        self.maximum_width = 1 * self.width_to_height_ratio
        self.minimum_height = -1
        self.maximum_height = 1

        locations_y = torch.rand(1, 1, num_points, device=self.device) * 2 - 1
        locations_x = (
            torch.rand(1, 1, num_points, device=self.device)
            * (self.maximum_width - self.minimum_width)
            + self.minimum_width
        )
        self.locations = nn.Parameter(
            torch.stack(
                (locations_y, locations_x), dim=-1
            )  # Shape: (1, 1, num_points, 2)
        )  # relative to canvas size

        self.matrix_offsets = nn.Parameter(
            torch.randn(num_points, 2, 2, device=self.device, dtype=torch.float32) * 2
        )

        self.matrix_scale_exponents = nn.Parameter(
            torch.zeros(num_points, 1, 1, device=self.device, dtype=torch.float32)
        )
        self.colors = nn.Parameter(
            torch.zeros(1, 1, num_points, 3, device=self.device, dtype=torch.float32)
        )  # Values to add or subtract from each channel of the canvas.

    def render(self, canvas_height_px: int, canvas_width_px: int) -> Tensor:
        """
        Render points on a canvas.

        Args:
            canvas_height_px (int): Height of the canvas in pixels.
            canvas_width_px (int): Width of the canvas in pixels.

        Returns:
            Tensor: A tensor representing the rendered points on the canvas.
        """

        untransformed_coordinates = torch.meshgrid(
            torch.linspace(
                self.minimum_height,
                self.maximum_height,
                canvas_height_px,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.linspace(
                self.minimum_width,
                self.maximum_width,
                canvas_width_px,
                dtype=torch.float32,
                device=self.device,
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
        ) / (
            torch.exp(self.matrix_scale_exponents)
        )
        transform_matrices = transform_matrices.view(1, 1, self.num_points, 2, 2)
        transformed_coordinates = torch.matmul(
            shifted_coordinates, transform_matrices
        ).squeeze(
            -2
        )  # (H, W, num_points, 2)

        distances = torch.mean(torch.abs(transformed_coordinates), dim=-1, keepdim=True)

        # distances = torch.sqrt(
        #     torch.sum(torch.square(transformed_coordinates), dim=-1, keepdim=True)
        # )

        # distances = torch.sum(
        #     torch.square(transformed_coordinates), dim=-1, keepdim=True
        # )

        mapped_distances = torch.relu(1 - distances)

        canvas = self.colors * mapped_distances

        canvas = canvas.sum(dim=-2)  # (H, W, 3)
        canvas = torch.sigmoid(canvas * 4)

        return canvas.permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)


def scale_then_crop(image: Tensor, desired_height: int, desired_width: int):
    """
    Resizes smaller dimension of the image to the desired size, then crops the larger dimension to match the aspect ratio.
    Args:
        image (Tensor): Input image tensor of shape (C, H, W).
        desired_height (int): Desired height of the output image.
        desired_width (int): Desired width of the output image.
    Returns:
        Tensor: Resized and cropped image tensor of shape (C, desired_height, desired_width).
    """
    # Get the size of the image
    C, H, W = image.shape

    # Determine which dimension is smaller and scale accordingly.
    if H < W:
        # Scale height to desired_height
        scale_factor = desired_height / H
        new_width = int(round(W * scale_factor))
        resized = TF.resize(image, (desired_height, new_width))
        # Crop width to desired_width
        cropped = TF.center_crop(resized, (desired_height, desired_width))
    else:
        # Scale width to desired_width (also handles H == W)
        scale_factor = desired_width / W
        new_height = int(round(H * scale_factor))
        resized = TF.resize(image, (new_height, desired_width))
        # Crop height to desired_height
        cropped = TF.center_crop(resized, (desired_height, desired_width))

    return cropped


def sobol_filter(image: Tensor) -> Tensor:
    """
    Applies a Sobel filter for edge detection on an input image tensor.

    This function accepts an image in CHW format and applies the Sobel operator separately to
    each channel using group convolution. It computes the gradient in the x and y directions,
    calculates the gradient magnitude, and clamps the results to the range [0.0, 1.0].
    This approach maintains the color information by processing each channel independently.

    Parameters:
        image (torch.Tensor): A tensor representing the image in CHW format (Channels x Height x Width).

    Returns:
        torch.Tensor: A tensor in CHW format containing the edge-detected image
                      with gradient magnitudes in each channel.
    """
    # Assume image is in CHW format.
    # Add batch dimension: shape (1, C, H, W)
    image_unsqueezed = image.unsqueeze(0)
    C = image.shape[0]
    device = image.device

    # Define Sobel kernels for x and y directions.
    base_sobel_x = (
        torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            device=device,
            dtype=torch.float32,
        )
        / 4
    )
    base_sobel_y = (
        torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            device=device,
            dtype=torch.float32,
        )
        / 4
    )

    # Expand the kernels to apply them separately on each channel using groups convolution.
    sobel_x = base_sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = base_sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # Apply convolution using groups equal to number of channels.
    grad_x = F.conv2d(
        image_unsqueezed,
        sobel_x,
        padding=1,
        groups=C,
    )
    grad_y = F.conv2d(
        image_unsqueezed,
        sobel_y,
        padding=1,
        groups=C,
    )

    # Compute gradient magnitude.
    grad_magnitude = (torch.abs(grad_x) + torch.abs(grad_y)) / 2

    # Remove batch dimension.
    edge_image = grad_magnitude.squeeze(0)  # shape (C, H, W)
    return edge_image


def float_to_uint8(image: Tensor) -> Tensor:
    """
    Convert a float tensor image to uint8 format.

    Args:
        image (Tensor): Input image tensor of shape (C, H, W) with values in [0, 1].

    Returns:
        Tensor: Converted image tensor of shape (C, H, W) with values in [0, 255].
    """
    return (image.detach().cpu() * 255.0).clamp(0, 255).to(torch.uint8)
