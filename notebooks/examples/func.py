import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path


class Draw:
    def __init__(self):
        pass

    @staticmethod
    def plot_one_image(file_path: Path, fig_size: tuple = (7, 3)):
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.imshow(Image.open(file_path))
        fig.tight_layout()
        fig.show()

    @staticmethod
    def plot_one_image_long_caption(file_path: Path, caption: str, fig_size: tuple = (12, 5), vertical: bool = False):
        nrows, ncols = 1, 2
        if vertical:
            nrows, ncols = 2, 1
            fig_size = (5, 12)

        fig, (ax1, ax2) = plt.subplots(nrows, ncols, figsize=fig_size)
        ax1.imshow(Image.open(file_path))
        ax1.axis('off')
        ax2.text(0, 0.5, caption, va='center', ha='left', wrap=True)
        ax2.axis('off')
        fig.tight_layout()
        fig.show()

    @staticmethod
    def plot_multiple_images(file_paths: list, fig_size: tuple = (20, 10)):
        # layout
        num_images, ncol = len(file_paths), min(len(file_paths), 5)
        nrow = int(np.ceil(num_images / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
        axes = axes.flatten()

        # plot
        for i, image in enumerate(file_paths):
            axes[i].imshow(Image.open(image))
            axes[i].axis('off')

        # remove axes
        for _ in range(num_images, ncol * nrow, 1):
            axes[_].remove()
        fig.tight_layout()
