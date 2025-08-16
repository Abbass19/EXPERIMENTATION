import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Normal Distribution Function
# ----------------------------
def normal_distribution(x, mean, std):
    """Returns the probability density of a normal distribution for array x."""
    coeff = 1 / (std * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coeff * np.exp(exponent)


# ----------------------------
# Outlier Calculations
# ----------------------------
def calculate_Outliers_IQR(data):
    """Calculate lower and upper outlier boundaries using IQR method."""
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    IQR = q3 - q1
    lower_outlier = q1 - 1.5 * IQR
    higher_outlier = q3 + 1.5 * IQR
    return lower_outlier, higher_outlier


def calculate_Outliers_std(data, mean, std, k=2):
    """Calculate lower and upper outlier boundaries using mean ± k*std."""
    lower_outlier = mean - k * std
    upper_outlier = mean + k * std
    return lower_outlier, upper_outlier


def count_outliers(data, lower_bound, upper_bound):
    """Count the number of outliers outside the provided bounds. Returns count and mask."""
    outliers_mask = (data < lower_bound) | (data > upper_bound)
    return np.sum(outliers_mask), outliers_mask


# ----------------------------
# Outliers as a Function of Variance
# ----------------------------
def outliers_as_function_of_variance(mean=50, final_variance=40, size=1000):
    """Plots the percentage of outliers as variance increases."""
    x = np.linspace(0, 100, size)
    percentage_outlier = []

    for current_variance in range(1, final_variance + 1):
        x_distribution = normal_distribution(x, mean, current_variance)
        lower_outlier, higher_outlier = calculate_Outliers_std(x_distribution, mean, current_variance)
        outlier_count, _ = count_outliers(x, lower_outlier, higher_outlier)
        outlier_percentage = (outlier_count / len(x)) * 100
        percentage_outlier.append(outlier_percentage)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, final_variance + 1), percentage_outlier, marker='o', color='b')
    plt.title(f"Percentage of Outliers as Variance Changes from 1 to {final_variance}")
    plt.xlabel("Variance")
    plt.ylabel("Percentage of Outliers")
    plt.grid(True)
    plt.show()
