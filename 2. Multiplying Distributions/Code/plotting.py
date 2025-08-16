import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider

def create_figure():
    """Create figure and axes for 1D and 3D plots."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)
    return fig, ax

def plot_1d_distribution(ax, x, dist, title, xlabel, ylabel):
    line, = ax.plot(x, dist, lw=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return line

def plot_3d_surface(ax, X, Y, Z, title='Joint 3D Surface'):
    ax.cla()
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('X Vector')
    ax.set_ylabel('Y Vector')
    ax.set_zlabel('Joint Probability')