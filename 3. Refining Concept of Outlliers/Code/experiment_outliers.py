import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functions import normal_distribution, calculate_Outliers_std, count_outliers, outliers_as_function_of_variance

# ----------------------------
# Initial Parameters
# ----------------------------
mean, std = 50, 10
size = 1000
x = np.linspace(0, 100, size)
x_distribution = normal_distribution(x, mean, std)

# Initial outlier boundaries
lower_outlier, higher_outlier = calculate_Outliers_std(x_distribution, mean, std)

# ----------------------------
# Set up Figure and Plot
# ----------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.3)
line_x, = ax.plot(x, x_distribution, label="Normal Distribution")
line_1 = ax.axvline(lower_outlier, color='r', linestyle='--', label="Lower Outlier")
line_2 = ax.axvline(higher_outlier, color='r', linestyle='--', label="Higher Outlier")
text_box = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# ----------------------------
# Slider Setup
# ----------------------------
slider_mean_ax = plt.axes([0.4, 0.2, 0.35, 0.05])
slider_std_ax = plt.axes([0.4, 0.1, 0.35, 0.05])
slider_mean = Slider(slider_mean_ax, 'Mean', 0, 100, valinit=mean)
slider_std = Slider(slider_std_ax, 'Std Dev', 1, 50, valinit=std)

# ----------------------------
# Update Function
# ----------------------------
def update(val):
    new_mean = slider_mean.val
    new_std = slider_std.val

    # Update distribution
    new_x_distribution = normal_distribution(x, new_mean, new_std)
    line_x.set_ydata(new_x_distribution)

    # Update outlier boundaries
    lower_out, upper_out = calculate_Outliers_std(new_x_distribution, new_mean, new_std)
    line_1.set_xdata([lower_out])
    line_2.set_xdata([upper_out])

    # Update text with outlier percentage
    outlier_count, _ = count_outliers(x, lower_out, upper_out)
    outlier_percentage = (outlier_count / len(x)) * 100
    text_box.set_text(f"Outliers: {outlier_percentage:.2f}%")

    fig.canvas.draw_idle()

# Connect sliders
slider_mean.on_changed(update)
slider_std.on_changed(update)

# Final plot
plt.legend()
plt.title("Interactive Normal Distribution with Outlier Boundaries")
plt.show()

# ----------------------------
# Optional: Plot Outliers as Function of Variance
# ----------------------------
outliers_as_function_of_variance()
