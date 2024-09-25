import matplotlib.pyplot as plt
from PIL import Image
import polars as pl
import numpy as np


def draw_images(data, test_id):
    # search
    sample = data.filter(pl.col('q_item_id') == test_id)
    org_path = sample['q_file_path'][0]
    query_path = sample['db_file_path'].to_list()

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.imshow(Image.open(org_path))
    fig.tight_layout()

    num_images, ncol = 5, 5
    nrow = int(np.ceil(num_images / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 10))
    axes = axes.flatten()
    for i, image in enumerate(query_path):
        axes[i].imshow(Image.open(image))
        axes[i].axis('off')

    for _ in range(num_images, ncol * nrow, 1):
        axes[_].remove()
    fig.tight_layout()
