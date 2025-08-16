import numpy as np

def normal_distribution(x, mean, std):
    """Generate 1D normal distribution values."""
    coeff = 1 / (std * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coeff * np.exp(exponent)

def joint_distribution(x_dist, y_dist):
    """Compute 2D joint distribution assuming independence."""
    X, Y = np.meshgrid(x_dist, y_dist)
    Z = X * Y
    return X, Y, Z