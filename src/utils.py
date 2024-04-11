import torch
import pandas as pd
import sklearn.metrics
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
import numpy as np
from config import Config
import matplotlib.pyplot as plt
import librosa as lb


def show_batch(img_ds, num_items, num_rows, num_cols, predict_arr=None):
    fig = plt.figure(figsize=(12, 6))
    img_index = np.random.randint(0, len(img_ds)-1, num_items)
    for index, img_index in enumerate(img_index):  # list first 9 images
        img, lb = img_ds[img_index]
        ax = fig.add_subplot(num_rows, num_cols, index +
                             1, xticks=[], yticks=[])
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy()
        if isinstance(img, np.ndarray):
            img = img.transpose(1, 2, 0)
            ax.imshow(img)

        title = f"Spec"
        ax.set_title(title)


def get_optimizer(lr, params):
    model_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, params),
        lr=lr,
        weight_decay=Config.weight_decay
    )
    interval = "epoch"

    lr_scheduler = CosineAnnealingWarmRestarts(
        model_optimizer,
        T_0=Config.epochs,
        T_mult=1,
        eta_min=1e-6,
        last_epoch=-1
    )

    return {
        "optimizer": model_optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": interval,
            "monitor": "val_loss",
            "frequency": 1
        }
    }


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution  # .drop(['row_id'], axis=1, errors='ignore')
    submission = submission  # .drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat(
        [solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat(
        [submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score


def map_score(solution, submission):
    solution = solution  # .drop(['row_id'], axis=1, errors='ignore')
    submission = submission  # .drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',
    )
    return score


def compute_melspec(y, sr, n_mels, fmin, fmax):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = lb.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )

    melspec = lb.power_to_db(melspec).astype(np.float32)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])

        n_repeats = length // len(y)
        epsilon = length % len(y)

        y = np.concatenate([y]*n_repeats + [y[:epsilon]])

    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y
