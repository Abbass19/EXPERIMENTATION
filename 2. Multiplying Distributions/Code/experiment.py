import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from distributions import normal_distribution, joint_distribution
from plotting import create_figure, plot_1d_distribution, plot_3d_surface

# -------------------- Parameters --------------------
boundary = 20
size = 100
(x_mean_init, x_std_init) = (10, 2)
(y_mean_init, y_std_init) = (10, 2)

x = np.linspace(0, boundary, size)
y = np.linspace(0, boundary, size)

# -------------------- Initial Distributions --------------------
x_dist = normal_distribution(x, x_mean_init, x_std_init)
y_dist = normal_distribution(y, y_mean_init, y_std_init)
X, Y, Z = joint_distribution(x_dist, y_dist)

# -------------------- Figure and Axes --------------------
fig, ax = create_figure()

line_x = plot_1d_distribution(ax[0], x, x_dist, 'Normal Distribution of X', 'x', 'Probability')
line_y = plot_1d_distribution(ax[2], y, y_dist, 'Normal Distribution of Y', 'y', 'Probability')

ax[1] = fig.add_subplot(1, 3, 2, projection='3d')
plot_3d_surface(ax[1], X, Y, Z)

# -------------------- Sliders --------------------
ax_x_mean = plt.axes([0.1, 0.2, 0.35, 0.05])
ax_x_std = plt.axes([0.1, 0.1, 0.35, 0.05])
ax_y_mean = plt.axes([0.55, 0.2, 0.35, 0.05])
ax_y_std = plt.axes([0.55, 0.1, 0.35, 0.05])

slider_x_mean = Slider(ax_x_mean, 'X Mean', 0, 20, valinit=x_mean_init)
slider_x_std = Slider(ax_x_std, 'X Std', 0.1, 10, valinit=x_std_init)
slider_y_mean = Slider(ax_y_mean, 'Y Mean', 0, 20, valinit=y_mean_init)
slider_y_std = Slider(ax_y_std, 'Y Std', 0.1, 10, valinit=y_std_init)

# -------------------- Update Function --------------------
def update(val):
    x_mean = slider_x_mean.val
    x_std = slider_x_std.val
    y_mean = slider_y_mean.val
    y_std = slider_y_std.val

    x_dist_new = normal_distribution(x, x_mean, x_std)
    y_dist_new = normal_distribution(y, y_mean, y_std)

    line_x.set_ydata(x_dist_new)
    line_y.set_ydata(y_dist_new)

    X, Y, Z = joint_distribution(x_dist_new, y_dist_new)
    plot_3d_surface(ax[1], X, Y, Z)

    fig.canvas.draw_idle()

slider_x_mean.on_changed(update)
slider_x_std.on_changed(update)
slider_y_mean.on_changed(update)
slider_y_std.on_changed(update)

plt.show()
