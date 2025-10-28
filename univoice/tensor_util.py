import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



def spec_to_figure(spec, title="", file_name=""):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    H = spec.shape[1] // 2
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        spec.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.title(title)
    plt.savefig(file_name)  
    plt.close()
    return fig


def spec_to_figure_single(spec, title="", file_name=""):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    H = spec.shape[1] // 2
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(
        spec.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.title(title)
    plt.savefig(file_name)  
    plt.close()
    return fig

