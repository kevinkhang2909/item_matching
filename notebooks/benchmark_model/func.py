import matplotlib.pyplot as plt
from PIL import Image
import textwrap
import math


def plot_img(
        image_paths: list,
        titles: list = None,
        ncols: int = 1,
        figsize: tuple = (6, 6),
        wrap_width: int = 30
):
    # Create figure
    num_images = len(image_paths)
    nrows = int(math.ceil(num_images / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Make sure axes is a flat list
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    if not titles:
        titles = [""] * len(image_paths)

    # Plot images and wrapped titles
    for i, (path, title) in enumerate(zip(image_paths, titles)):
        img = Image.open(path)
        axes[i].imshow(img)
        axes[i].axis('off')

        wrapped_title = '\n'.join(textwrap.wrap(str(title), wrap_width))
        axes[i].set_title(wrapped_title, fontsize=10)

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
