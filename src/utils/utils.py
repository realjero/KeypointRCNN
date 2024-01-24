import torch
from matplotlib import pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def plot_keypoints(kpts):
    for p in kpts:
        for kpt in p:
            x, y, v = kpt
            if v >= 0.9:
                plt.scatter(x, y, color='green', marker='o', s=1)
