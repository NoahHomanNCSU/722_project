"""
Datasets and dataset utilities

[1] Joel Janek Dabrowski, Daniel Edward Pagendam, James Hilton, Conrad Sanderson, 
    Daniel MacKinlay, Carolyn Huston, Andrew Bolt, Petra Kuhnert, "Bayesian 
    Physics Informed Neural Networks for Data Assimilation and Spatio-Temporal 
    Modelling of Wildfires", Spatial Statistics, Volume 55, June 2023, 100746
    https://www.sciencedirect.com/science/article/pii/S2211675323000210
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

__author__      = "Joel Janek Dabrowski"
__license__     = "MIT license"
__version__     = "0.0.0"

def level_set_function(x, y, x0=0.0, y0=0.0, offset=0.0):
    """
    Generate the level set function in the form of a signed distance function.
    Positive values: Outside the firefront.
    Negative values: Inside the firefront.
    Zero values: Firefront boundary.

    :param x: grid over the x-dimension with shape [Nx]
    :param y: grid over the y-dimension with shape [Ny]
    :param x0: location of the centre of the signed distance function on x
    :param y0: location of the centre of the signed distance function on y
    :param offset: offset of the signed distance function below the zero level
        set plane.
    :return: the level set function with shape [Nx, Ny, 1]
    """
    with torch.no_grad():
        Nx = x.shape[0]
        Ny = y.shape[0]
        X, Y = torch.meshgrid(x, y)
        u = torch.zeros((Nx, Ny, 1))
        # Signed distance function
        u[:, :, 0] = torch.sqrt((X-x0)**2 + (Y-y0)**2) - offset
    return u

def level_set_palisades(X):
    """
    Generate the level set function given the Palisades fire damage data.
    Positive values: Outside the firefront.
    Negative values: Inside the firefront.
    Zero values: Firefront boundary.

    :param X: raster grid over the x-dimension with shape [height, width]
    :return: the level set function with shape [height, width, 1]
    """
    with torch.no_grad():
        # Initialize level set (1 for unburned, -1 for burned)
        X = torch.from_numpy(X).float().clone()
        u = torch.where(X > 0, -1.0, 1.0)
        
        # Create a kernel for checking 8-neighborhood connectivity
        kernel = torch.ones(3, 3, device=X.device)
        kernel[1,1] = 0  # Ignore center pixel
        
        # Pad the input to handle edge pixels
        padded = F.pad(u.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='constant', value=1)
        
        # Convolve to count neighboring unburned pixels (value=1)
        neighbor_count = F.conv2d(padded, kernel.unsqueeze(0).unsqueeze(0))
        
        # Find boundary pixels:
        # 1) Current pixel is burned (-1)
        # 2) Has at least one unburned neighbor (neighbor_count > 0)
        boundary_mask = (u == -1) & (neighbor_count.squeeze() > 0)
        
        # Set boundary pixels to 0
        u[boundary_mask] = 0.0
        
        # Add channel dimension
        u = u.unsqueeze(-1)

    # Convert to numpy if it's a torch tensor
    level_set = u.squeeze().numpy()

    plt.figure(figsize=(8, 6))

    # Create binary mask where boundary=0 becomes black (0), everything else white (1)
    boundary_only = np.where(level_set == 0, 0, 1)

    # Display with grayscale colormap (0=black, 1=white)
    plt.imshow(boundary_only, cmap='gray', vmin=0, vmax=1)

    plt.title("Fire Boundary Outline")
    plt.axis('off')

    # Remove colorbar since we're using binary values
    plt.show()

    return u

def c_wind_obstruction_complex(t, x, y):
    """
    Compute the speed constant c in the level-set equation. This constant 
    comprises both the wind speed and the fire-front speed

    :param t: grid over the time-dimension with shape [Nt]
    :param x: grid over the x-dimension with shape [Nx]
    :param y: grid over the x-dimension with shape [Ny]
    :return: the firefront speed s, and the wind speed in the x, and y 
        directions. These are returned as tensors with shape [Nx, Ny, Nt]
    """
    with torch.no_grad():
        if torch.is_tensor(t):
            X, Y, T = torch.meshgrid(x, y, t)
        else:
            X, Y = torch.meshgrid(x, y)
        
        # Firefront speed
        s = 0.15 * torch.ones_like(X)
        mask = X<0.5
        s[mask] = 0.25
        
        # Wind
        w_x = 1e-6 * torch.ones_like(X)
        w_y = 0.1 * torch.ones_like(X)
        
        if t.shape[0] > 1:
            np.random.seed(0)
            for ti in range(1,t.shape[0]):
                w_x[:,:,ti] = w_x[:,:,ti-1] + 0.001 * np.random.randn(1)
                w_y[:,:,ti] = w_y[:,:,ti-1] + 0.005 * np.random.randn(1)
        
        # Obstructions
        x1_1 = -0.2
        x1_2 = 0.3
        y1_1 = 0.2
        y1_2 = 0.8
        x2_1 = 0.7
        x2_2 = 0.8
        y2_1 = 0.4
        y2_2 = 0.5
        x3_1 = 0.7
        x3_2 = 0.8
        y3_1 = 0.6
        y3_2 = 0.7
        mask = ((X>x1_1) & (X<x1_2) & (Y>y1_1) & (Y<y1_2)) | ((X>x2_1) & (X<x2_2) & (Y>y2_1) & (Y<y2_2)) | ((X>x3_1) & (X<x3_2) & (Y>y3_1) & (Y<y3_2))
        s[mask] = 1e-9
        w_x[mask] = 1e-9
        w_y[mask] = 1e-9
    return s, w_x, w_y


class DatasetScaler(nn.Module):
    def __init__(self, x_min, x_max):
        """
        Utility for scaling a dataset to a given minimum and maximum value.

        :param x_min: desired minimum value of the scaled data
        :param x_max: desired maximum value of the scaled data
        """
        super(DatasetScaler, self).__init__()
        self.x_min = x_min
        self.x_max = x_max
    
    def forward(self, x):
        """
        Apply the scaling to an input tensor x

        :param x: input tensor to be scaled
        :return: scaled tensor with range [self.x_min, self.x_max]
        """
        if self.x_min == self.x_max:
            return x
        else:
            with torch.no_grad():
                x = (x - self.x_min) / (self.x_max - self.x_min)
            return x