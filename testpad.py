# sphinx_gallery_thumbnail_path = "../../gallery/assets/transforms_thumbnail.png"

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T


plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open('1.png')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)
print(orig_img.size)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.savefig("test.jpg")

def padf(size=100):
    block_diff = []
    for percent in range(3,4):
        for location in range(0, percent ** 2):
            block_size = size
            #location 0 -- percent*percent-1
            target_block_row = location//percent
            target_block_col = location%percent
            left = target_block_col * block_size
            right = (percent-target_block_col-1) * block_size
            top = target_block_row * block_size
            bot = (percent-target_block_row-1) * block_size

            block_diff.append([left,top,right,bot])
    return block_diff
bd = padf()
print(bd)
padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in bd]
for x in padded_imgs:
    print(x.size)
plot(padded_imgs)